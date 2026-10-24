'''Summaries of single RoPE planes for figures.py and `sec:rope`, from the raw results of 01_measure; summarize stores them under summary['planes_extra']. Per (head, plane) quantities are averaged over prompts at the last query; head means are over the non-sink heads of every layer (mean attention on the first token at most SINK), variances averaged before any ratio.'''
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from src.analysis import counterfactual as cf
from src.core.suite import TASKS
from src.utils.results import read

from analyses import phase_pseudo
from measure import head_table

KEYS = ('var_c', 'var_p', 'var_e', 'w2', 'sigma_w2')
SLOW, FAST = 0.1, 0.9  # sigma_zeta^2 below SLOW: slow plane; above FAST: fast plane
TRANSITION = (2.0, 13.0)  # period over the context of the transition planes in the ratio quartiles
BOUND_TARGETS = (0.2, 0.5, 0.85)  # sigma_zeta^2 of the three planes of the bound figure
BUDGET_TURNS = (0.01, 0.05, 0.2, 0.6, 2.0, 20.0)  # turns at the shortest length of the planes of the budget figure
BINS = np.linspace(-2.5, 4.5, 29)  # log10 of the period over the context: bins of the pooled quartiles over heads


def planes(d: dict) -> dict[str, np.ndarray]:
    '''Per non-sink head (rows) and plane (columns), averaged over prompts: KEYS (squared logits); per plane sigma_zeta2 (over prompts), theta, turns (T - 1) theta_f / 2 pi and the period over the context (its inverse), with T - 1 the mean prompt length minus 2; and the mean length.'''
    keep = head_table(d)['sink'] <= cf.SINK
    out = {k: np.asarray(d['plane'][f'plane_{k}'], float).mean(0)[keep] for k in KEYS}
    length = float(np.mean(d['length']))
    theta = np.asarray(d['theta'], float)
    out |= {'sigma_zeta2': np.asarray(d['sigma_zeta2'], float).mean(0), 'theta': theta, 'turns': (length - 2) * theta / (2 * np.pi), 'period_over_context': 2 * np.pi / theta / (length - 2), 'length': length}
    return out


def head_mean(p: dict) -> dict[str, np.ndarray]:
    return {k: p[k].mean(0) for k in KEYS}


def bounds(p: dict) -> np.ndarray:
    '''The bounds of `cor:where-each-part-can-live` (content, position, interaction) per head and plane: sigma_w^2 (1 - sigma_zeta^2), |w~|^2 sigma_zeta^2, sigma_w^2 sigma_zeta^2.'''
    sz = p['sigma_zeta2']
    return np.stack([p['sigma_w2'] * (1 - sz), p['w2'] * sz, p['sigma_w2'] * sz])


def phase_center(theta: np.ndarray, length: float) -> np.ndarray:
    '''|zeta~^f| = |delta_n(theta_f)| of `prop:phase-statistics` on n = length - 2 positions (the context of measure).'''
    n = int(round(length)) - 2
    return abs(np.sin(n * theta / 2) / (n * np.sin(theta / 2)))


def phase_covariance(theta: np.ndarray, n: int) -> np.ndarray:
    '''The covariance E_j[zr_j^f conj(zr_j^g)] of the centered phases zeta_j^f = e^{i (T - j) theta_f} over the n = T - 1 positions j = 1, ..., T - 1 (F, F), complex (`prop:head-as-a-sum-of-planes`); its diagonal is sigma_zeta,f^2.'''
    z = np.exp(1j * np.arange(1, n + 1)[:, None] * theta[None, :])
    zr = z - z.mean(0)
    return zr.T @ zr.conj() / n


def fast_covariance_median(theta: np.ndarray, length: float) -> float:
    '''The median over pairs f != g of fast planes (closed-form sigma_zeta^2 > FAST) of |E_j[zr_j^f conj(zr_j^g)]| on n = length - 2 positions (the context of measure).'''
    G = abs(phase_covariance(np.asarray(theta, float), int(round(length)) - 2))
    fast = np.diag(G) > FAST
    return float(np.median(G[np.ix_(fast, fast)][~np.eye(int(fast.sum()), dtype=bool)]))


def bound_mosaic(p: dict) -> dict:
    '''Three planes of one model and case (planes: 0-based indices; the document and figures number planes from 1), the closest to sigma_zeta^2 = BOUND_TARGETS (distinct), from the head means: sigma_zeta2, the squared mean over the budget, |w~^f|^2 / (|w~^f|^2 + sigma_w^2) (w2_share), and each variance over its bound of `cor:where-each-part-can-live` (ratio_c, ratio_p, ratio_e).'''
    h, sz = head_mean(p), p['sigma_zeta2']
    idx = sorted({int(np.argmin(abs(sz - t))) for t in BOUND_TARGETS}, key=lambda i: sz[i])
    b = bounds(h | {'sigma_zeta2': sz})
    out = {'planes': np.array(idx), 'sigma_zeta2': sz[idx], 'w2_share': (h['w2'] / (h['w2'] + h['sigma_w2']))[idx]}
    for k, part in enumerate('cpe'):
        out[f'ratio_{part}'] = h[f'var_{part}'][idx] / b[k][idx]
    return out


def fractions(per_model: dict[str, dict]) -> dict:
    '''With D = Var(c^f) + Var(p^f) + Var(eps^f) of each (non-sink head, plane): per model (models), the period over the context, the median over heads of each part's fraction of D (fraction_q50; rows content, position, interaction) and the quartiles over (head, plane) pairs of each part's variance over its bound of `cor:where-each-part-can-live` (ratio_q, rows content, position, interaction; columns the quartiles) in the fast (sigma_zeta^2 > FAST), transition (period between TRANSITION times the context) and slow (sigma_zeta^2 < SLOW) planes; pooled over models in bins of log10 of the period over the context (pooled), the bin centers and the quartiles of each part's fraction of D (fraction_q25, fraction_q50, fraction_q75) and of its variance over its bound (ratio_q25, ratio_q50, ratio_q75).'''
    out: dict = {'models': {}}
    pool_x, pool_f, pool_r = [], [], []
    quartiles = lambda a: np.nanquantile(a, [0.25, 0.5, 0.75], axis=1)
    for m, p in per_model.items():
        v = np.stack([p['var_c'], p['var_p'], p['var_e']])
        total = np.where(v.sum(0) > 0, v.sum(0), np.nan)
        b = bounds(p)
        frac = v / total
        ratio = v / np.where(b > 0, b, np.nan)
        sz, x = np.broadcast_to(p['sigma_zeta2'], ratio.shape[1:]), np.broadcast_to(p['period_over_context'], ratio.shape[1:])
        zones = {'fast': sz > FAST, 'transition': (x >= TRANSITION[0]) & (x <= TRANSITION[1]), 'slow': sz < SLOW}
        out['models'][m] = {'period_over_context': p['period_over_context'], 'fraction_q50': np.nanquantile(frac, 0.5, axis=1), 'ratio_q': {z: np.nanquantile(ratio[:, mask], [0.25, 0.5, 0.75], axis=1).T for z, mask in zones.items()}}
        pool_x.append(np.broadcast_to(np.log10(p['period_over_context']), frac.shape[1:]).ravel())
        pool_f.append(frac.reshape(3, -1)), pool_r.append(ratio.reshape(3, -1))
    x, f, r = np.concatenate(pool_x), np.concatenate(pool_f, 1), np.concatenate(pool_r, 1)
    which = np.digitize(x, BINS) - 1
    kept = [k for k in range(BINS.size - 1) if (which == k).sum() >= 20]
    fq, rq = (np.stack([quartiles(a[:, which == k]) for k in kept], -1) if kept else np.empty((3, 3, 0)) for a in (f, r))
    centers = np.array([10 ** ((BINS[k] + BINS[k + 1]) / 2) for k in kept])
    out['pooled'] = {'period_over_context': centers, **{f'fraction_q{k}': fq[i] for i, k in enumerate((25, 50, 75))}, **{f'ratio_q{k}': rq[i] for i, k in enumerate((25, 50, 75))}}
    return out


def budget(by_length: dict[int, dict]) -> dict:
    '''One model and task at several lengths (lengths), the planes (0-based indices; the document and figures number planes from 1) whose turns at the shortest length are the closest to BUDGET_TURNS (planes): per length the turns, the shares of the head-mean budget |w~|^2 + sigma_w^2 of Var(c^f), Var(p^f), Var(eps^f) and the rest (share_c, share_p, share_e, share_rest; the rest is the squared mean plus what the real part loses), the position-dependent share Phi^f (phi) and the interval (sigma_zeta^2 -+ |s_zeta^f|) / 2 of `cor:slow-planes-and-long-contexts` (lo, hi).'''
    lengths = sorted(by_length)
    turns0 = by_length[lengths[0]]['turns']
    idx = sorted({int(np.argmin(abs(np.log(turns0 / t)))) for t in BUDGET_TURNS}, key=lambda i: turns0[i])
    out: dict = {'planes': np.array(idx), 'lengths': np.array(lengths), 'by_length': {}}
    for L in lengths:
        p = by_length[L]
        h = head_mean(p)
        B = h['w2'] + h['sigma_w2']
        sz = p['sigma_zeta2'][idx]
        ps = phase_pseudo(p['period_over_context'], p['length'])[idx]
        sh = {part: (h[f'var_{part}'] / B)[idx] for part in 'cpe'}
        out['by_length'][str(L)] = {'turns': p['turns'][idx], **{f'share_{k}': v for k, v in sh.items()}, 'share_rest': 1 - sh['c'] - sh['p'] - sh['e'], 'phi': sh['p'] + sh['e'], 'lo': (sz - ps) / 2, 'hi': (sz + ps) / 2}
    return out


def ratio_limit(per_case: dict[str, dict]) -> dict:
    '''`cor:ratio-in-long-contexts` per long case of one model (cases): per plane the turns, sigma_zeta2, the measured plane positional ratio of the head means, P^f = (Var p + Var eps) / (Var p + Var c + 2 Var eps) (pf), and its limit (|w~|^2 + sigma_w^2) / (|w~|^2 + 2 sigma_w^2) from the same head means.'''
    out = {}
    for case, p in per_case.items():
        h = head_mean(p)
        pe, ce = h['var_p'] + h['var_e'], h['var_c'] + h['var_e']
        out[case] = {'turns': p['turns'], 'sigma_zeta2': p['sigma_zeta2'], 'pf': pe / (pe + ce), 'limit': (h['w2'] + h['sigma_w2']) / (h['w2'] + 2 * h['sigma_w2'])}
    return out


def limit_stats(limits: dict[str, dict], lengths: list[int]) -> dict:
    '''Per length (per_length), pooled over models and long tasks, over fast planes (sigma_zeta^2 > FAST): the number of (case, plane) pairs and the median of |P^f - limit|.'''
    def stats(rows: list[dict]) -> dict:
        r = np.concatenate([d['pf'][d['sigma_zeta2'] > FAST] for d in rows])
        l = np.concatenate([d['limit'][d['sigma_zeta2'] > FAST] for d in rows])
        return {'n': int(r.size), 'median_abs_error': float(np.median(abs(r - l))) if r.size else float('nan')}
    return {'per_length': {str(L): stats([d for m in limits.values() for c, d in m.items() if c.endswith(f'_{L}')]) for L in lengths}}


def plane_extras(cfg: dict, exp: Path) -> dict:
    '''The plane summaries: theta (per model, the RoPE frequencies theta_f), bounds (bound_mosaic of the main model on the plane case), zones (per model, on the plane case and on the single-needle task at the longest length: measured sigma_zeta2, |zeta~^f| and |s_zeta^f| at the mean length, the mean length, the number of fast planes and fast_covariance_median), zone_context (per case of zones, the mean over models of the positions T - 1 of the context, rounded, and whether every model has the same), fractions (fractions on the plane case), budget (budget of the main model on the single-needle task at the shortest and longest lengths), limit (ratio_limit of every model on every long case) and limit_stats.'''
    f = cfg['figures']
    main, case, lengths = f['main_model'], f['plane_case'], sorted(cfg['lengths'])
    long_tasks = [t for t in cfg['tasks'] if TASKS[t]['category'] == 'long']
    load = lambda m, c: planes(d) if (d := read(exp, m, c)) is not None else None
    out: dict = {'theta': {}, 'zones': {}, 'zone_context': {}, 'limit': {}}
    text = {}
    for m in cfg['models']:
        if (p := load(m, case)) is None:
            continue
        text[m] = p
        out['theta'][m] = p['theta']
        out['zones'][m] = {}
        long_case = f'niah_{lengths[-1]}'
        for c, q in ((case, p), (long_case, load(m, long_case))):
            if q is not None:
                out['zones'][m][c] = {'sigma_zeta2': q['sigma_zeta2'], 'center': phase_center(q['theta'], q['length']), 'pseudo': phase_pseudo(q['period_over_context'], q['length']), 'length_mean': q['length'], 'fast': int((q['sigma_zeta2'] > FAST).sum()), 'fast_covariance_median': fast_covariance_median(q['theta'], q['length'])}
        out['limit'][m] = ratio_limit({f'{t}_{L}': q for t in long_tasks for L in lengths if (q := load(m, f'{t}_{L}')) is not None})
    for c in dict.fromkeys(c for z in out['zones'].values() for c in z):
        ns = [int(round(z[c]['length_mean'])) - 2 for z in out['zones'].values() if c in z]
        out['zone_context'][c] = {'positions': int(round(np.mean(ns))), 'same': len(set(ns)) == 1}
    if main in text:
        out['bounds'] = bound_mosaic(text[main])
        if all(c in out['limit'][main] for c in (f'niah_{lengths[0]}', f'niah_{lengths[-1]}')):
            out['budget'] = {'model': main, **budget({L: load(main, f'niah_{L}') for L in (lengths[0], lengths[-1])})}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out['fractions'] = {'length_mean': next(iter(text.values()))['length'], **fractions(text)} if text else {}
    out['limit_stats'] = limit_stats(out['limit'], lengths)
    return out
