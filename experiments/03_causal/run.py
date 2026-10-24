'''03_causal: part edits in groups of heads of pretrained models (`sec:stability`). In the chosen query heads every context logit keeps only some parts (c, p, epsilon) of the counterfactual matrix at its query; the groups come from 01_measure's calibration; the effect is measured on the solved short and long tasks and on WikiText-2. Results: <model>__intact, <model>__<edit>__<group>.'''
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.analysis import counterfactual as cf
from src.analysis.calibration import head_table, windows
from src.core.models import Model, for_each_model
from src.core.suite import prompts
from src.eval.text_tasks import metric, nll
from src.utils.results import EXPERIMENTS, figures_dir, read, result_path, save, write_summary
from src.utils.stats import bootstrap_mean

from parts import check_part_edit, check_part_edit_identity, part_edit

REMOVED = {'c': ('var_p', 'var_e'), 'pe': ('var_c',), 'cp': ('var_e',)}  # variances removed, by the parts kept
TOLERANCE = {'parts': 1e-9, 'identity': 1e-6}  # parts: float64 closed form against the explicit matrix; identity: float32 logits


def table(cfg: dict, name: str) -> dict[str, np.ndarray]:
    '''Per query head, flat (layer * heads + head), from 01's calibration: the variances of the parts averaged over queries, the positional ratio, the interaction share and the sink weight; shape, (layers, heads).'''
    cal = read(EXPERIMENTS / cfg['calibration_experiment'], name, f'calibration_{cfg["calibration_tokens"]}')
    if cal is None:
        raise SystemExit(f'{name}: no calibration_{cfg["calibration_tokens"]} in {cfg["calibration_experiment"]}; run it first')
    return {**{k: np.asarray(cal[k], float).mean(0).ravel() for k in ('var_c', 'var_p', 'var_e')}, **{k: v.ravel() for k, v in head_table(cal).items()}, 'shape': np.asarray(cal['var_c']).shape[1:]}


def removed(t: dict, keep: str) -> np.ndarray:
    '''Per head, the variance the edit keeping keep removes.'''
    return sum(t[k] for k in REMOVED[keep])


def groups(t: dict, frac: float, n_layers: int, n_heads: int, keep: str, seed: int, draws: int) -> dict[str, np.ndarray]:
    '''Sorted flat indices of each group of round(frac * non-sink heads) non-sink heads: the most positional, symbolic and interacting; random heads (random_0); and the closest-variance controls, the random group of non-sink heads among draws whose removed variance (summed over its heads) is closest to that of the positional (matched_positional) or symbolic group (matched_symbolic).'''
    ok = np.flatnonzero(t['sink'] <= cf.SINK)
    k = max(1, round(frac * ok.size))
    top = lambda score: ok[np.argsort(-score[ok], kind='stable')][:k]
    rng = np.random.default_rng([seed, round(frac * 1000)])
    out = {'positional': top(t['ratio']), 'symbolic': top(-t['ratio']), 'interaction': top(t['interaction']), 'random_0': rng.choice(ok, k, replace=False)}
    for third in (ok // n_heads < n_layers / 3, ok // n_heads >= 2 * n_layers / 3):
        rng.choice(ok[third], min(k, third.sum()), replace=False)  # two draws within the first and last third of the layers precede the candidates, so that every group is the one its stored result was computed with
    rv = removed(t, keep)
    cand = np.array([rng.choice(ok, k, replace=False) for _ in range(draws)])
    tot = rv[cand].sum(1)
    for name in ('positional', 'symbolic'):
        out[f'matched_{name}'] = cand[np.argmin(np.abs(tot - rv[out[name]].sum()))]
    return {name: np.sort(v) for name, v in out.items()}


def inputs(cfg: dict, m: Model) -> dict:
    '''The evaluation inputs of one model: the prompts of each short task (binding: context i keeps its question about pair i mod 16) and long task, and the text windows.'''
    seed = cfg['seed'] + 1
    short = lambda t, n: [prompts(t, m.tokenizer, 16 * n, seed)[16 * i + i % 16] for i in range(n)] if t == 'binding' else prompts(t, m.tokenizer, n, seed)
    tasks = {t: short(t, cfg['n']['short']) for t in cfg['tasks']['short']} | {t: prompts(t, m.tokenizer, cfg['n']['long'], seed, target=cfg['target']) for t in cfg['tasks']['long']}
    return {'tasks': tasks, 'text': windows(m, cfg['text'], cfg['text_n'], cfg['text_tokens'], seed)}


@torch.no_grad()
def evaluate(m: Model, inp: dict) -> dict:
    '''With whatever attention is in force: the margin of each prompt (task_<task>) and the NLL of each window (text).'''
    return {f'task_{t}': np.array([metric(m, p) for p in ps]) for t, ps in inp['tasks'].items()} | {'text': np.array([nll(m, ids) for ids in inp['text']])}


def compute(cfg: dict, exp: Path) -> None:
    def run(m: Model) -> None:
        g = m.geometry
        t, inp = table(cfg, m.name), inputs(cfg, m)
        save(result_path(exp, m.name, 'intact'), lambda: evaluate(m, inp))
        for edit, e in cfg['edits'].items():
            gs = groups(t, e['fraction'], g.n_layers, g.n_heads, e['keep'], cfg['seed'], cfg['matched_draws'])
            for name in e['groups']:
                idx = gs[name]
                heads = {l: idx[idx // g.n_heads == l] % g.n_heads for l in range(g.n_layers)}

                def body(heads: dict = heads, keep: str = e['keep']) -> dict:
                    with part_edit(m, heads, keep, cfg['chunk']):
                        return evaluate(m, inp)
                save(result_path(exp, m.name, f'{edit}__{name}'), body)

    for_each_model(cfg, cfg['models'], run)


def summarize(cfg: dict, exp: Path) -> None:
    '''Per model: the intact accuracy per task (intact) and the solved tasks (solved), the only ones in the statistics. Per <edit>__<group>: the change of the text NLL (nats per token); over the solved short and long tasks, the margin change standardized by each task's intact standard deviation and the accuracy change; for the first group of each edit, the paired standardized difference from its control (minus_matched_<group> when the edit has a matched control, else minus_random_0), with a 95% BCa interval, and the ratio of their text NLL changes (nll_over_<control>); with a matched control, also control: the variance it removes over the group's, each summed over the heads from the calibration table the groups are drawn from (variance_ratio), the heads it shares with the group out of size and as a fraction of it (shared_fraction), and the group against the reference of a cost proportional to the variance removed, the control's effects divided by that ratio (minus_scaled: the group's standardized margin change minus the control's divided by variance_ratio, per kind; nll_over_scaled: the group's text NLL change over the control's divided by variance_ratio).'''
    boot = lambda x: {'mean': float(x.mean()), 'ci': list(bootstrap_mean(x, cfg['n_boot'], cfg['seed']))}
    summary: dict = {'models': {}}
    for n in cfg['models']:
        if (base := read(exp, n, 'intact')) is None:
            continue
        accuracy = {k.removeprefix('task_'): float(np.mean(np.asarray(v) > 0)) for k, v in base.items() if k.startswith('task_')}
        solved = {kind: [f'task_{t}' for t in ts if accuracy[t] >= cfg['solved']] for kind, ts in cfg['tasks'].items()}
        solved = {kind: keys for kind, keys in solved.items() if keys}
        std = lambda d, ref, keys: np.concatenate([(np.asarray(d[k], float) - np.asarray(ref[k], float)) / (float(np.std(base[k])) or 1.0) for k in keys])
        out: dict = {'intact': accuracy, 'solved': [k.removeprefix('task_') for keys in solved.values() for k in keys]}
        t = table(cfg, n)
        layers, heads = (int(x) for x in t['shape'])
        for edit, e in cfg['edits'].items():
            res = {name: d for name in e['groups'] if (d := read(exp, n, f'{edit}__{name}')) is not None}
            gs, rv = groups(t, e['fraction'], layers, heads, e['keep'], cfg['seed'], cfg['matched_draws']), removed(t, e['keep'])
            target = e['groups'][0]
            control = f'matched_{target}' if f'matched_{target}' in e['groups'] else 'random_0'
            removed_variance = {name: float(rv[gs[name]].sum()) for name in res}
            for name, d in res.items():
                row = {'nll_change': float(np.mean(np.asarray(d['text']) - np.asarray(base['text'])))}
                for kind, keys in solved.items():
                    x = std(d, base, keys)
                    row[kind] = {'margin_change_std': float(x.mean()), 'accuracy_change': float(np.mean(np.concatenate([(np.asarray(d[k]) > 0).astype(float) - (np.asarray(base[k]) > 0) for k in keys])))}
                    if name == target and control in res:
                        row[kind][f'minus_{control}'] = boot(std(d, res[control], keys))
                out[f'{edit}__{name}'] = row
            if {target, control} <= res.keys():
                g, c = out[f'{edit}__{target}'], out[f'{edit}__{control}']
                g[f'nll_over_{control}'] = g['nll_change'] / c['nll_change']
                if control.startswith('matched_'):
                    ratio = removed_variance[control] / removed_variance[target]
                    shared = len(set(np.ravel(gs[target]).tolist()) & set(np.ravel(gs[control]).tolist()))
                    scaled = {'nll_change': c['nll_change'] / ratio, **{kind: c[kind]['margin_change_std'] / ratio for kind in solved}}
                    g['control'] = {'variance_ratio': ratio, 'shared_heads': shared, 'size': int(np.size(gs[target])), 'shared_fraction': shared / int(np.size(gs[target])),
                                    'minus_scaled': {kind: g[kind]['margin_change_std'] - scaled[kind] for kind in solved}, 'nll_over_scaled': g['nll_change'] / scaled['nll_change']}
        summary['models'][n] = out
    write_summary(exp, summary)


def figures(cfg: dict, exp: Path) -> None:
    from figures import draw
    draw(cfg, read(exp, 'summary'), figures_dir(exp))


def check(cfg: dict, exp: Path) -> None:
    '''Identity checks of the part edit on one text window per model: check_part_edit and check_part_edit_identity; fails above TOLERANCE.'''
    def run(m: Model) -> None:
        ids = windows(m, cfg['text'], 1, cfg['check_tokens'], cfg['seed'])[0]
        dev = {'parts': check_part_edit(m, ids), 'identity': check_part_edit_identity(m, ids)}
        for k, v in dev.items():
            print(f'{m.name}: {k} deviation {v:.2e} (tolerance {TOLERANCE[k]:.0e})', flush=True)
        if bad := [k for k, v in dev.items() if not v <= TOLERANCE[k]]:
            raise SystemExit(f'{m.name}: check failed for {", ".join(bad)}')

    for_each_model(cfg, cfg['models'], run)
