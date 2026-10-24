'''01_measure: the shared measurement. Per model, one forward pass per benchmark input gives the parts of every head and plane at the last query, the deviations of Urrutia et al. and the attention on the first token, the answer and the distractor; plus the induction-circuit roles on repeated random tokens, Chen and Yan's quantities, and the calibration passes on WikiText-2 that 03 and 04 read.'''
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import counterfactual as cf
from src.analysis.calibration import calibrate, windows
from src.core.models import Model, for_each_model
from src.core.suite import TASKS, prompts
from src.utils.results import figures_dir, read, result_path, save, write_summary
from src.utils.stats import spearman

from measure import head_table, interaction_bound, measure, split_half, stack
from roles import repeated_prompts, role_summary, roles

def cases(cfg: dict) -> list[tuple[str, str, int | None]]:
    '''(case name, task, target length) of every case: one per task, and one per target length for long tasks.'''
    out = []
    for t in cfg['tasks']:
        if TASKS[t]['category'] == 'long':
            out += [(f'{t}_{n}', t, n) for n in cfg['lengths']]
        else:
            out.append((t, t, None))
    return out


def compute(cfg: dict, exp: Path) -> None:
    from chen import pseudo_variances
    c = cfg['calibration']
    spec = {'source': c['source'], 'span': c['span']}

    def run(m: Model) -> None:
        g = {'theta': m.geometry.theta().numpy(), 'n_kv': m.geometry.n_kv_heads}

        def case_prompts(task: str, target: int | None) -> list:
            return prompts(task, m.tokenizer, cfg['n'][TASKS[task]['category']], cfg['seed'], cfg['max_tokens'], target=target)

        def case(task: str, target: int | None) -> dict:
            return {**stack([measure(m, p) for p in case_prompts(task, target)]), **g}

        def repeated() -> dict:
            prs = repeated_prompts(m, cfg['repeated'], cfg['seed'])
            return {**roles(m, [p.ids for p in prs], cfg['repeated']['length']), **stack([measure(m, p) for p in prs]), **g}

        for name, task, target in cases(cfg):
            save(result_path(exp, m.name, name), lambda task=task, target=target: case(task, target))
        save(result_path(exp, m.name, 'repeated'), repeated)
        save(result_path(exp, m.name, 'pseudo'), lambda: {**pseudo_variances(m, case_prompts('wiki', None)), **g})
        for tokens in c['tokens']:
            save(result_path(exp, m.name, f'calibration_{tokens}'), lambda tokens=tokens: calibrate(m, windows(m, spec, c['windows'][tokens], tokens, cfg['seed']), c['queries'], c['elitekv_from'] if tokens in c['elitekv_tokens'] else None))

    for_each_model(cfg, cfg['models'], run)


def case_summary(cfg: dict, d: dict, common: np.ndarray) -> dict:
    '''One case of one model, at the last query: per head the positional ratio and the behavior (`def:behavior-and-positional-ratio`), non_sink, the mean prompt length, the split-half reliability of P_h over the heads in common (non-sink in every case of the model, cal(H)^* of `sec:stability`) and interaction_bound; per plane the head means over non-sink heads of Var(p^f), Var(eps^f), |w~^f|^2 and sigma_w,f^2 (plane_means) and the period over the context.'''
    from analyses import plane_means
    h = head_table(d)
    keep = h['sink'] <= cf.SINK
    pl = plane_means(d, keep)
    return {'heads': {'ph': h['ratio'], 'behavior': h['behavior']}, 'non_sink': keep, 'length_mean': float(np.mean(d['length'])), 'split_half': split_half(d, common, cfg['n_split'], cfg['seed']), 'interaction_bound': interaction_bound(d), 'plane': {k: pl[k] for k in ('var_p', 'var_e', 'w2', 'sigma_w2', 'period_over_context')}}


def deviation_summary(d: dict) -> dict:
    '''The deviations of Urrutia et al. (`def:deviations`) at the last query, averaged over prompts before any ratio: per head the positional ratio and their normalized measure Dsym / (Dpos + Dsym); per head the variance V of the observed logits over Dpos + Dsym (bound_theirs); over non-sink heads, their agreement (Spearman, fraction on opposite sides of 1/2) and the median of bound_theirs.'''
    h = head_table(d)
    v = {k: np.asarray(d['head'][k], float)[:, -1].mean(0) for k in ('var_obs', 'd_pos', 'd_sym')}
    ratio, keep = h['ratio'], h['sink'] <= cf.SINK
    normalized, theirs = v['d_sym'] / (v['d_pos'] + v['d_sym']), v['var_obs'] / (v['d_pos'] + v['d_sym'])
    return {'ph': ratio, 'measure': normalized, 'bound_theirs': theirs, 'non_sink': keep, 'spearman': spearman(ratio[keep], normalized[keep]), 'opposite_sides': float(((ratio > 0.5) != (normalized > 0.5))[keep].mean()), 'bound_theirs_median': float(np.median(theirs[keep]))}


def summarize(cfg: dict, exp: Path) -> None:
    from analyses import agreement, answer, head_extras
    from chen import pseudo_summary
    from planes import plane_extras
    summary: dict = {'cases': {}, 'tasks': {c: {'task': t, 'category': TASKS[t]['category'], 'target': n} for c, t, n in cases(cfg)}, 'questions': {}, 'urrutia': {}, 'roles': {}, 'chen': {}}
    rp = cfg['repeated']
    for n in cfg['models']:
        raw = {c: d for c, _, _ in cases(cfg) if (d := read(exp, n, c)) is not None}
        if raw:
            common = np.logical_and.reduce([head_table(d)['sink'] <= cf.SINK for d in raw.values()])
            cs = summary['cases'][n] = {c: case_summary(cfg, d, common) for c, d in raw.items()}
            summary['urrutia'][n] = {c: deviation_summary(d) for c, d in raw.items() if summary['tasks'][c]['category'] != 'long'}
            summary['questions'][n] = answer(raw, cs, summary['tasks'], cfg, int(next(iter(raw.values()))['n_kv']))
        if (rep := read(exp, n, 'repeated')) is not None:
            keys = ('var_c', 'var_p', 'var_e', 'sink')
            planes = lambda d: {'norm': np.asarray(d['plane']['plane_w2']) + np.asarray(d['plane']['plane_sigma_w2']), 'position': np.asarray(d['plane']['plane_var_p']) + np.asarray(d['plane']['plane_var_e']), 'sigma_zeta2': d['sigma_zeta2']}
            parts = {'repeated': {k: rep['head'][k] for k in keys}}
            if 'wiki' in raw:
                parts['wiki'] = {**{k: np.asarray(raw['wiki']['head'][k])[:, -1] for k in keys}, **planes(raw['wiki'])}
            summary['roles'][n] = role_summary(rep, parts, rp['prev_threshold'], rp['induction_threshold'])
        if (ps := read(exp, n, 'pseudo')) is not None and 'wiki' in summary['cases'].get(n, {}):
            w = summary['cases'][n]['wiki']
            summary['chen'][n] = pseudo_summary(ps, np.asarray(w['heads']['ph'], float), np.asarray(w['non_sink'], bool))
    if (main := cfg['figures']['main_model']) in summary['cases']:
        summary['agreement'] = {'model': main, **agreement(summary['cases'][main], summary['tasks'])}
    summary['planes_extra'] = plane_extras(cfg, exp)
    summary['heads_extra'] = head_extras(cfg, exp)
    write_summary(exp, summary)


def figures(cfg: dict, exp: Path) -> None:
    from figures import draw
    draw(cfg, read(exp, 'summary'), figures_dir(exp))


def check(cfg: dict, exp: Path) -> None:
    '''The identity checks of checks.py; fails on any deviation above its tolerance.'''
    from checks import run_checks
    for_each_model(cfg, cfg['models'], lambda m: run_checks(m, cfg))
