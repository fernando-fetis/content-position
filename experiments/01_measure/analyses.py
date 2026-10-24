'''The summaries of 01_measure about whole heads: the questions answered from the raw results and case summaries of one model (answer, summary['questions'][model]; heads non-sink in every case compared), and the whole-head quantities pooled over models (head_extras, summary['heads_extra']).'''
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import counterfactual as cf
from src.core.suite import TASKS
from src.utils.results import read
from src.utils.stats import spearman

SHORT = ('positional', 'symbolic', 'interaction', 'mixed')
NAN = float('nan')
QUANTILES = {'q10': 0.1, 'median': 0.5}


def _kind(category: str) -> str:
    return 'short' if category in SHORT else category


def _common(cs: dict, names: list[str]) -> np.ndarray:
    return np.logical_and.reduce([np.asarray(cs[c]['non_sink'], bool) for c in names])


def _thirds(x: np.ndarray) -> np.ndarray:
    return np.array([x[t].mean() if t.size else NAN for t in np.array_split(np.arange(x.size), 3)])


def _prompt_mean(x: np.ndarray) -> np.ndarray | None:
    '''The mean over prompts of a per-prompt attention, None when no prompt has one (no answer in the context).'''
    x = np.asarray(x, float)
    return np.nanmean(x, 0) if np.isfinite(x).any() else None


def plane_means(d: dict, keep: np.ndarray) -> dict[str, np.ndarray]:
    '''Per plane of one case at the last query: Var(c^f), Var(p^f), Var(eps^f), |w~^f|^2 and sigma_w,f^2 averaged over prompts and the heads in keep (squared logits), sigma_zeta,f^2 over prompts, and the period over the context, 2 pi / (theta_f (T - 1)) with T - 1 the mean prompt length minus 2.'''
    out = {k: np.asarray(d['plane'][f'plane_{k}'], float)[:, keep].mean((0, 1)) for k in ('var_c', 'var_p', 'var_e', 'w2', 'sigma_w2')}
    out['sigma_zeta2'] = np.asarray(d['sigma_zeta2'], float).mean(0)
    out['period_over_context'] = 2 * np.pi / np.asarray(d['theta'], float) / (np.mean(d['length']) - 2)
    return out


def phase_pseudo(period_over_context: np.ndarray, length: float) -> np.ndarray:
    '''|s_zeta^f| of the planes with these periods over the context, on a context of length - 2 positions (the context of measure, without the first token and the query).'''
    n = int(round(length)) - 2
    z = np.exp(2j * np.pi * np.outer(1 / (period_over_context * n), np.arange(1, n + 1)))
    zt = z.mean(1)
    return abs((z ** 2).mean(1) - zt ** 2)


def stability(cs: dict, tasks: dict, threshold: float) -> dict:
    '''The stability of P_h over the heads that are non-sink in every case (`sec:stability`): the minimum over cases of the split-half reliability; the mean over pairs of cases of the Spearman correlation of P_h; the fraction of heads that keep their class (symbolic P_h < 1 - threshold, positional > threshold, mixed) over the short tasks; the median over heads of the standard deviation of P_h across the short tasks.'''
    names = list(cs)
    short = [i for i, c in enumerate(names) if tasks[c]['category'] in SHORT]
    keep = _common(cs, names)
    R = np.stack([np.asarray(cs[c]['heads']['ph'], float)[keep] for c in names])
    S = np.array([[spearman(a, b) for b in R] for a in R])[~np.eye(len(names), dtype=bool)]
    cls = np.where(R > threshold, 2, np.where(R < 1 - threshold, 0, 1))
    return {'split_half_min': float(min(cs[c]['split_half'] for c in names)), 'spearman_mean': float(S.mean()) if S.size else NAN, 'same_class_short': float((cls[short] == cls[short][:1]).all(0).mean()), 'sd_short_median': float(np.median(R[short].std(0))) if short else NAN}


def agreement(cs: dict, tasks: dict) -> dict:
    '''The cases of one model ordered as short tasks, texts and long tasks (cases) and, over the heads non-sink in every case (cal(H)^* of `sec:stability`), the Spearman correlation of P_h between every pair of them (spearman, (C, C)).'''
    kind = lambda c: 0 if tasks[c]['category'] in SHORT else (1 if tasks[c]['category'] == 'text' else 2)
    names = sorted(cs, key=lambda c: (kind(c), list(cs).index(c)))
    keep = _common(cs, names)
    R = np.stack([np.asarray(cs[c]['heads']['ph'], float)[keep] for c in names])
    return {'cases': names, 'spearman': np.array([[spearman(x, y) for y in R] for x in R])}


def interaction(cs: dict) -> dict:
    '''Per case, over non-sink heads, the interaction share b_h3 = Var(eps) / Var(A) (`def:behavior-and-positional-ratio`): its median and the fraction of heads above 0.2.'''
    out = {}
    for c, d in cs.items():
        f = np.asarray(d['heads']['behavior'], float)[..., 2][np.asarray(d['non_sink'], bool)]
        out[c] = {'median': float(np.median(f)), 'above_0.2': float(np.mean(f > 0.2))}
    return out


def depth(d: dict, threshold: float) -> dict:
    '''In one case, the fractions of positional (P_h > threshold) and symbolic (P_h < 1 - threshold) heads among the non-sink heads of each layer, averaged over each third of the layers.'''
    r, keep = np.asarray(d['heads']['ph'], float), np.asarray(d['non_sink'], bool)
    per_layer = lambda sel: (sel & keep).sum(1) / np.maximum(keep.sum(1), 1)
    return {'positional_thirds': _thirds(per_layer(r > threshold)), 'symbolic_thirds': _thirds(per_layer(r < 1 - threshold))}


def planes(raw: dict, cs: dict, slow: float = 0.1) -> dict:
    '''Per case, from the plane variances averaged over prompts and non-sink heads (plane_means): the median over the slow planes (sigma_zeta^2 below slow) of the content variance over its bound of `cor:where-each-part-can-live` (bound_ratio_c); the share of E_i |w_i^f|^2 that is position-dependent variance, (Var(p^f) + Var(eps^f)) / (|w~^f|^2 + sigma_w,f^2), against the interval (sigma_zeta^2 -+ |s_zeta^f|) / 2 of `cor:slow-planes-and-long-contexts` at the mean prompt length: its largest excess over the interval (phi_excess; positive only through the spread of prompt lengths within a case) and its largest value (phi_max); the share is Phi^f of `sec:length`.'''
    out = {}
    for c, d in raw.items():
        pl = plane_means(d, np.asarray(cs[c]['non_sink'], bool))
        x = pl['period_over_context']
        parts = np.stack([pl[k] for k in ('var_c', 'var_p', 'var_e')])
        w2, s2, sz = pl['w2'], pl['sigma_w2'], pl['sigma_zeta2']
        half_s = phase_pseudo(x, cs[c]['length_mean']) / 2
        is_slow = sz < slow
        share = (parts[1] + parts[2]) / (w2 + s2)
        excess = np.maximum((sz - 2 * half_s) / 2 - share, share - (sz + 2 * half_s) / 2)
        out[c] = {'bound_ratio_c': float(np.median(parts[0][is_slow] / (s2 * (1 - sz))[is_slow])), 'phi_excess': float(excess.max()), 'phi_max': float(share.max())}
    return out


def reading(raw: dict, cs: dict, top: float, accuracy: float, margin: float) -> dict:
    '''Over the cases with an answer in the context, the reader heads of each (the fraction top of non-sink heads with the most mean attention on the answer). A case is read when its accuracy (answer logit above the distractor's) is at least accuracy and its readers attend to the answer at least margin times as much as to the distractor (0 when there is none). The read cases (read) and the mean over them of the mean P_h of all non-sink heads minus that of the readers (gap).'''
    rows = {}
    for c, d in raw.items():
        if (att := _prompt_mean(d['att_answer'])) is None:
            continue
        keep = np.asarray(cs[c]['non_sink'], bool)
        a, r = att[keep], np.asarray(cs[c]['heads']['ph'], float)[keep]
        readers = np.argsort(-a)[:max(1, round(top * a.size))]
        dis = _prompt_mean(d['att_distractor'])
        metric = np.asarray(d['metric'], float)
        rows[c] = {'accuracy': float(np.mean(metric > 0)) if np.isfinite(metric).all() else NAN, 'gap': float(r.mean()) - float(r[readers].mean()), 'att_answer': float(a[readers].mean()), 'att_distractor': float(dis[keep][readers].mean()) if dis is not None and np.isfinite(dis[keep]).any() else 0.0}
    read = [c for c, v in rows.items() if v['accuracy'] >= accuracy and v['att_answer'] >= margin * v['att_distractor']]
    return {'read': read, 'gap': float(np.mean([rows[c]['gap'] for c in read])) if read else NAN}


def groups(cs: dict, tasks: dict, n_kv: int, seed: int, n_perm: int = 200) -> dict:
    '''Over the short tasks, the fraction of the within-layer variance of P_h explained by the KV group (eta2), and its mean under n_perm random assignments of the heads to groups (eta2_chance); means over cases.'''
    eta, chance = [], []
    for c, d in cs.items():
        if tasks[c]['category'] not in SHORT:
            continue
        r, keep = np.asarray(d['heads']['ph'], float), np.asarray(d['non_sink'], bool)
        group = np.arange(r.shape[1]) // (r.shape[1] // n_kv)
        rng = np.random.default_rng(seed)
        total = between_groups = permuted = 0.0
        for l in range(r.shape[0]):
            x, g = r[l, keep[l]], group[keep[l]]
            if x.size < 2:
                continue
            between = lambda g: sum((x[g == k].mean() - x.mean()) ** 2 * (g == k).sum() for k in np.unique(g))
            total += ((x - x.mean()) ** 2).sum()
            between_groups += between(g)
            permuted += np.mean([between(rng.permutation(g)) for _ in range(n_perm)])
        if total:
            eta.append(between_groups / total)
            chance.append(permuted / total)
    return {'eta2': float(np.mean(eta)) if eta else NAN, 'eta2_chance': float(np.mean(chance)) if chance else NAN}


def interaction_bound(cs: dict, tasks: dict) -> dict:
    '''Pooled over the non-sink heads of each kind of input (short, text, long): the median of kappa_2 = Var(eps) / d_h^-1 sum_f sigma_w,f^2 sigma_zeta,f^2 (one half for evenly spread phases; `prop:head-as-a-sum-of-planes`).'''
    out = {}
    for kind in ('short', 'text', 'long'):
        names = [c for c in cs if _kind(tasks[c]['category']) == kind]
        if names:
            m, pb = (np.concatenate([np.asarray(cs[c]['interaction_bound'][k], float)[np.asarray(cs[c]['non_sink'], bool)] for c in names]) for k in ('measured', 'plane_bound_sum'))
            out[kind] = {'kappa2': float(np.median(m / np.maximum(pb, 1e-30)))}
    return {'kinds': out}


def decomposition(cs: dict, tasks: dict) -> dict:
    '''eta of `sec:stability` over the short tasks and the heads non-sink in every case (cal(H)^*): the share of the sum of squares of P_h over (head, short task) pairs explained by each head's mean over the short tasks.'''
    keep = _common(cs, list(cs))
    R = np.stack([np.asarray(cs[c]['heads']['ph'], float)[keep] for c in cs if tasks[c]['category'] in SHORT], 1)
    return {'eta': float(((R.mean(1) - R.mean()) ** 2).mean() / ((R - R.mean()) ** 2).mean())}


def _phase_variance(theta: np.ndarray, N: float) -> np.ndarray:
    x = np.asarray(theta, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        D = np.where(x > 0, np.sin(N * x / 2) / (N * np.sin(x / 2)), 1.0)
    return 1 - D ** 2


def length_prediction(raw: dict, cs: dict, tasks: dict) -> dict:
    '''Per long task, over the heads non-sink at every length: the P_h predicted when the content statistics of each plane (|w~^f|^2 and sigma_w,f^2 of the shortest length, averaged over prompts) are kept and only the variance of the phases is recomputed at each length (sigma_zeta^2 = 1 - delta_N(theta_f)^2, N the mean prompt length minus two; `prop:phase-statistics`), with Var(c^f) = sigma_w,f^2 (1 - sigma_zeta^2) / 2, Var(p^f) = |w~^f|^2 sigma_zeta^2 / 2 and Var(eps^f) = sigma_w,f^2 sigma_zeta^2 / 2 summed over planes (`prop:the-parts-of-a-plane` without its pseudo-variance terms and `prop:head-as-a-sum-of-planes` without its cross terms). The means over heads of the measured and predicted rise from the shortest to the longest length (delta, Delta_h; delta_hat, its prediction hat(Delta)_h), the Spearman correlation over heads of the two rises, and the mean absolute error of the prediction at the shortest length.'''
    out = {}
    for t in sorted({v['task'] for c, v in tasks.items() if c in cs and v['category'] == 'long'}):
        names = [c for _, c in sorted((tasks[c]['target'], c) for c in cs if tasks[c]['task'] == t)]
        keep = _common(cs, names)
        base = raw[names[0]]
        w2 = np.asarray(base['plane']['plane_w2'], float).mean(0)
        s2 = np.asarray(base['plane']['plane_sigma_w2'], float).mean(0)
        measured, predicted = [], []
        for c in names:
            sz = _phase_variance(base['theta'], float(np.mean(raw[c]['length'])) - 2)
            vc, vp, ve = (0.5 * (s2 * (1 - sz)).sum(-1), 0.5 * (w2 * sz).sum(-1), 0.5 * (s2 * sz).sum(-1))
            predicted.append(((vp + ve) / (vp + 2 * ve + vc))[keep])
            measured.append(np.asarray(cs[c]['heads']['ph'], float)[keep])
        rise_m, rise_p = measured[-1] - measured[0], predicted[-1] - predicted[0]
        out[t] = {'delta': float(rise_m.mean()), 'delta_hat': float(rise_p.mean()), 'spearman_rise': spearman(rise_m, rise_p), 'base_abs_error': float(np.abs(predicted[0] - measured[0]).mean())}
    return out


def answer(raw: dict, cs: dict, tasks: dict, cfg: dict, n_kv: int) -> dict:
    '''Every question for one model, from its raw results and case summaries; depth only on WikiText-2.'''
    thr = cfg['ratio_threshold']
    return {'stability': stability(cs, tasks, thr), 'b3': interaction(cs), 'depth': {'wiki': depth(cs['wiki'], thr)} if 'wiki' in cs else {}, 'planes': planes(raw, cs), 'reading': reading(raw, cs, cfg['reader_fraction'], cfg['read']['accuracy'], cfg['read']['margin']), 'groups': groups(cs, tasks, n_kv, cfg['seed']), 'bounds': {k: max(float(d['bound'][k]) for d in raw.values()) for k in ('bound_p', 'bound_e', 'bound_c')}, 'interaction_bound': interaction_bound(cs, tasks), 'decomposition': decomposition(cs, tasks), 'length_prediction': length_prediction(raw, cs, tasks)}


def group(task: str, target: int | None) -> str:
    '''The length group of a case: short (the nine short tasks), text (the three 1024-token texts) or long_<target>.'''
    category = TASKS[task]['category']
    return 'short' if category in SHORT else ('text' if category == 'text' else f'long_{target}')


def _quantiles(x: np.ndarray) -> dict[str, float]:
    return {k: float(np.quantile(x, q)) for k, q in QUANTILES.items()}


def _case(d: dict) -> dict[str, np.ndarray]:
    '''Per layer and head at the last query, averaged over prompts (squared logits): the head's Var(c), Var(p), Var(eps); head_position = Var(p) + Var(eps); the plane sums d_h^-1 sum_f Var(eps^f) (plane_e) and d_h^-1 sum_f (Var(p^f) + Var(eps^f)) (plane_position); non_sink.'''
    head = lambda k: np.asarray(d['head'][k], float)[:, -1].mean(0)
    plane = lambda k: np.asarray(d['plane'][f'plane_{k}'], float).sum(-1).mean(0)
    out = {'var_c': head('var_c'), 'var_p': head('var_p'), 'var_e': head('var_e'), 'plane_e': plane('var_e')}
    out['head_position'] = out['var_p'] + out['var_e']
    out['plane_position'] = plane('var_p') + out['plane_e']
    out['non_sink'] = head('sink') <= cf.SINK
    return out


def head_extras(cfg: dict, exp: Path) -> dict:
    '''Per model and case (cases), per head (layers, heads) at the last query: head_position and plane_position (`prop:head-as-a-sum-of-planes`, item 2). Per length group (pooled: short, text, long_<T>), pooled over models and over the non-sink heads of the group's cases: the mean length and, for kappa_0 = head_position / plane_position (kappa0) and kappa_1 = Var(eps) / d_h^-1 sum_f Var(eps^f) (kappa1), the 10% quantile and the median (q10, median). Per model and case (geometry), over non-sink heads: the fraction of heads with P_h above ratio_threshold.'''
    thr = cfg['ratio_threshold']
    out: dict = {'cases': {}, 'pooled': {}, 'geometry': {}}
    rows: dict[str, dict[str, list]] = {}
    for m in cfg['models']:
        per_group: dict[str, dict[str, list]] = {}
        for t in cfg['tasks']:
            for target in (cfg['lengths'] if TASKS[t]['category'] == 'long' else [None]):
                name = f'{t}_{target}' if target else t
                if (d := read(exp, m, name)) is None:
                    continue
                v = _case(d)
                keep = v['non_sink']
                out['cases'].setdefault(m, {})[name] = {k: v[k] for k in ('head_position', 'plane_position')}
                ratio = cf.positional_ratio(v['var_c'], v['var_p'], v['var_e'])[keep]
                out['geometry'].setdefault(m, {})[name] = {'positional_fraction': float((ratio > thr).mean())}
                g = per_group.setdefault(group(t, target), {k: [] for k in ('kappa0', 'kappa1', 'length')})
                g['kappa0'].append((v['head_position'] / v['plane_position'])[keep])
                g['kappa1'].append((v['var_e'] / v['plane_e'])[keep])
                g['length'].append(float(np.mean(d['length'])))
        for name, g in per_group.items():
            pooled = rows.setdefault(name, {k: [] for k in g})
            for k in g:
                pooled[k] += g[k]
    for name, g in rows.items():
        x = {k: np.concatenate(g[k]) for k in ('kappa0', 'kappa1')}
        out['pooled'][name] = {'length': float(np.mean(g['length'])), 'aggregation': {k: _quantiles(r) for k, r in x.items()}}
    return out
