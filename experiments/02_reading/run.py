'''02_reading: heads at the three vertices of the simplex in toy models (`prop:three-pure-heads`): hand-built and trained one-layer RoPE transformers on the positional, symbolic and interaction tasks, their parts at the query and the part edits. Results: toy__hand_<task>, toy__trained_<task>_<seed>, toy__multi_<seed>.'''
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from src.core.models import resolve_device
from src.utils.results import figures_dir, read, result_path, save, write_summary

from toy import EDITS, Toy, Vocab, accuracy, edits, hand_built, make, measure, mixture, part_edit, train


def toy_setup(cfg: dict) -> tuple[Vocab, dict[str, dict], torch.device]:
    '''The vocabulary, one held-out evaluation set per task (fixed seeds, shared by every toy model) and the device.'''
    tm = cfg['toy_model']
    voc = Vocab(tm['length'], tm['symbols'], tm['keys'])
    evals = {t: make(t, voc, cfg['n_eval'], np.random.default_rng(20_000 + i)) for i, t in enumerate(cfg['toy_tasks'])}
    return voc, evals, resolve_device(cfg['device'])


def compute(cfg: dict, exp: Path) -> None:
    tm = cfg['toy_model']
    voc, evals, device = toy_setup(cfg)

    def hand(t: str) -> dict:
        m = hand_built(t, voc, tm['head_dim'], tm['base'], cfg['hand']['beta'], cfg['hand']['gamma'], cfg['seed']).to(device)
        return {'measure': measure(m, evals[t]), 'accuracy': accuracy(m, evals[t])}

    def trained(heads: int, sample: Callable, seed: int) -> tuple[Toy, np.ndarray]:
        torch.manual_seed(seed)
        m = Toy(voc.size, tm['d_model'], heads, tm['head_dim'], tm['base']).to(device)
        return m, np.array(train(m, sample, seed=seed, n_eval=cfg['n_eval'], **cfg['train']))

    def single(t: str, seed: int) -> dict:
        m, log = trained(1, lambda n, rng: make(t, voc, n, rng), seed)
        return {'log': log, 'measure': measure(m, evals[t])}

    def multi(seed: int) -> dict:
        m, _ = trained(cfg['multi_heads'], lambda n, rng: mixture(voc, n, rng), seed)
        return {t: {'measure': measure(m, evals[t]), 'edits': edits(m, evals[t])} for t in cfg['toy_tasks']}

    for t in cfg['toy_tasks']:
        save(result_path(exp, 'toy', f'hand_{t}'), lambda t=t: hand(t))
    for seed in cfg['seeds']:
        for t in cfg['toy_tasks']:
            save(result_path(exp, 'toy', f'trained_{t}_{seed}'), lambda t=t, seed=seed: single(t, seed))
        save(result_path(exp, 'toy', f'multi_{seed}'), lambda seed=seed: multi(seed))


def head_summary(res: dict) -> dict:
    '''Per head (layers, heads), variances averaged over examples: the behavior b_h (`def:behavior-and-positional-ratio`) and the mean attention on the answer.'''
    c, p, e = (np.asarray(res[k], float).mean(0) for k in ('var_c', 'var_p', 'var_e'))
    return {'behavior': np.stack([c, p, e], -1) / (c + p + e)[..., None], 'att_answer': np.asarray(res['att_answer'], float).mean(0)}


def toy_summary(cfg: dict, exp: Path) -> dict:
    '''hand.<task>: behavior and accuracy of the hand-built head. trained.<task>: per seed the final held-out accuracy, solved (accuracy at least solved) and the behavior of its head. multi.<task>: per seed the behavior of the reader head (most attention on the answer), the mean attention on the answer of the reader and of the second head (reader_attention, second_attention), and the mean over seeds of the accuracy with each edit (edits_mean).'''
    out: dict = {'hand': {}, 'trained': {}, 'multi': {}}
    for t in cfg['toy_tasks']:
        if (d := read(exp, 'toy', f'hand_{t}')) is not None:
            out['hand'][t] = {'behavior': head_summary(d['measure'])['behavior'][0, 0], 'accuracy': d['accuracy']}
        if rows := [d for s in cfg['seeds'] if (d := read(exp, 'toy', f'trained_{t}_{s}')) is not None]:
            hs = [head_summary(d['measure']) for d in rows]
            acc = np.array([d['log'][-1][2] for d in rows])
            out['trained'][t] = {'accuracy': acc, 'solved': acc >= cfg['solved'], 'behavior': np.array([h['behavior'][0, 0] for h in hs])}
    if multi := [d for s in cfg['seeds'] if (d := read(exp, 'toy', f'multi_{s}')) is not None]:
        for t in cfg['toy_tasks']:
            hs = [head_summary(d[t]['measure']) for d in multi]
            reader = [int(np.argmax(h['att_answer'][0])) for h in hs]
            att = np.sort(np.array([h['att_answer'][0] for h in hs]), 1)[:, ::-1]
            ed = {e: np.array([d[t]['edits'][e] for d in multi]) for e in EDITS}
            out['multi'][t] = {'reader_behavior': np.array([h['behavior'][0, r] for h, r in zip(hs, reader)]), 'reader_attention': att[:, 0], 'second_attention': att[:, 1], 'edits_mean': {e: float(v.mean()) for e, v in ed.items()}}
    return out


def summarize(cfg: dict, exp: Path) -> None:
    write_summary(exp, {'toy': toy_summary(cfg, exp)})


def figures(cfg: dict, exp: Path) -> None:
    from figures import draw
    draw(cfg, read(exp, 'summary'), figures_dir(exp))


def check(cfg: dict, exp: Path) -> None:
    '''On the hand-built head of every task: the logits with every part kept (part_edit with cpe, which rebuilds the diagonal from cf.matrix) equal the intact logits within check_tolerance, and the head solves its task; on the CPU instead of MPS, which has no float64.'''
    tm, tol = cfg['toy_model'], cfg['check_tolerance']
    voc, evals, device = toy_setup(cfg)
    device = torch.device('cpu') if device.type == 'mps' else device
    for t in cfg['toy_tasks']:
        m = hand_built(t, voc, tm['head_dim'], tm['base'], cfg['hand']['beta'], cfg['hand']['gamma'], cfg['seed']).to(device)
        ids = torch.as_tensor(evals[t]['ids'][:100], device=device)
        with torch.no_grad():
            dev = float((m(ids) - m(ids, part_edit(m, 'cpe'))).abs().max())
        acc = accuracy(m, evals[t])
        print(f'{t}: every part kept against the intact logits {dev:.2e}; hand-built accuracy {acc:.3f}', flush=True)
        if dev > tol:
            raise SystemExit(f'{t}: check failed, deviation {dev:.2e} above {tol:.0e}')
        if acc < cfg['solved']:
            raise SystemExit(f'{t}: check failed, the hand-built head does not solve its task ({acc:.3f} < {cfg["solved"]})')
