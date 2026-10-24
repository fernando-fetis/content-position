'''The shared measurement: one forward pass per input gives, per query, layer and head, the parts of the counterfactual matrix (`def:content-positional-and-interaction-parts` and `prop:the-parts-of-a-plane`), the deviations of Urrutia et al., the attention on the first token and the task metric; and the per-head tables of a stack of measurements.'''
from __future__ import annotations

import numpy as np
import torch

from src.analysis import counterfactual as cf
from src.core.models import Model, stream_qk
from src.core.suite import Prompt
from src.utils.stats import spearman

HEAD = ('mu', 'var_c', 'var_p', 'var_e', 'var_obs', 'd_pos', 'd_sym')
PLANE = ('plane_var_c', 'plane_var_p', 'plane_var_e', 'plane_w2', 'plane_sigma_w2')
BOUNDS = ('bound_p', 'bound_e', 'bound_c')


@torch.no_grad()
def measure(m: Model, pr: Prompt) -> dict:
    '''One forward pass. Per query, layer and head (queries, L, H): the HEAD keys over the context without the first token (mu in logits, the rest in squared logits) and sink. At the last query: the PLANE keys (L, H, F); sigma_zeta2 (F,); the largest violation of each bound of `cor:where-each-part-can-live` relative to E_i |w_i^f|^2 (bound_*, zero up to rounding); the attention on the answer's and the distractor's tokens in the context (att_answer, att_distractor, (L, H); NaN when absent); metric, the answer logit minus the distractor's; length and queries.'''
    g = m.geometry
    head = {k: np.zeros((len(pr.queries), g.n_layers, g.n_heads)) for k in (*HEAD, 'sink')}
    plane = {k: np.zeros((g.n_layers, g.n_heads, g.n_freqs)) for k in PLANE}
    row = {k: np.zeros((g.n_layers, g.n_heads)) for k in ('att_answer', 'att_distractor')}
    bound = {k: -np.inf for k in BOUNDS}
    on_answer = np.array(pr.answer_pos, int) if pr.answer_pos else None
    on_distractor = np.array(pr.distractor_pos, int) if pr.distractor_pos else None
    out: dict = {}

    def on_layer(layer: int, Q: torch.Tensor, K: torch.Tensor) -> None:
        for i, T in enumerate(pr.queries):
            P = cf.parts(cf.factors(m, Q[T], K, T))
            for k in HEAD:
                head[k][i, layer] = P[k].cpu().numpy()
            att = cf.attention(m, Q[T], K, T)
            head['sink'][i, layer] = att[:, 0].cpu().numpy()
        a = att.cpu().numpy()
        for k, where in (('att_answer', on_answer), ('att_distractor', on_distractor)):
            row[k][layer] = a[:, where].sum(-1) if where is not None else np.nan
        for k in PLANE:
            plane[k][layer] = P[k].cpu().numpy()
        sz, scale = P['sigma_zeta2'], P['plane_w2'] + P['plane_sigma_w2']
        for k, gap in (('bound_p', P['plane_var_p'] - P['plane_w2'] * sz), ('bound_e', P['plane_var_e'] - P['plane_sigma_w2'] * sz), ('bound_c', P['plane_var_c'] - P['plane_sigma_w2'] * (1 - sz))):
            bound[k] = max(bound[k], float((gap / scale.clamp_min(1e-30)).max()))
        out['sigma_zeta2'] = sz.cpu().numpy()

    with stream_qk(m, on_layer):
        logits = m.model(pr.ids[None].to(m.device), logits_to_keep=1).logits[0, -1].float()
    metric = float(logits[pr.answer_id] - logits[pr.distractor_id]) if pr.answer_id is not None else float('nan')
    return {**head, **plane, **bound, **out, **row, 'metric': metric, 'length': len(pr.ids), 'queries': np.array(pr.queries)}


def stack(rows: list[dict]) -> dict:
    '''The measurements of one case stacked over prompts: head.<key> (prompts, queries, L, H), plane.<key> (prompts, L, H, F), bound.<key> (largest over prompts), the rest with a leading prompts axis; head, plane and att_* in float32.'''
    st = lambda k: np.stack([r[k] for r in rows])
    return {'head': {k: st(k).astype(np.float32) for k in (*HEAD, 'sink')}, 'plane': {k: st(k).astype(np.float32) for k in PLANE}, 'sigma_zeta2': st('sigma_zeta2'), 'bound': {k: max(r[k] for r in rows) for k in BOUNDS}, 'metric': st('metric'), 'att_answer': st('att_answer').astype(np.float32), 'att_distractor': st('att_distractor').astype(np.float32), 'length': st('length'), 'queries': st('queries')}


def head_table(d: dict, query: int = -1) -> dict[str, np.ndarray]:
    '''Per layer and head at one query (default the last) of a stacked result: the variances of the parts averaged over prompts, the positional ratio and the behavior (fractions of Var c, Var p, Var e; `def:behavior-and-positional-ratio`) and the mean attention on the first token.'''
    mean = lambda k: np.asarray(d['head'][k], float)[:, query].mean(0)
    c, p, e = mean('var_c'), mean('var_p'), mean('var_e')
    total = c + p + e
    return {'var_c': c, 'var_p': p, 'var_e': e, 'ratio': cf.positional_ratio(c, p, e), 'behavior': np.stack([c, p, e], -1) / total[..., None], 'sink': mean('sink')}


def interaction_bound(d: dict) -> dict[str, np.ndarray]:
    '''Per layer and head at the last query, averaged over prompts (squared logits): measured, Var(eps); plane_sum, sum_f Var(eps^f) (equal to it when the planes' interactions are uncorrelated); plane_bound_sum, sum_f sigma_w,f^2 sigma_zeta,f^2, the sum of the bounds of `cor:where-each-part-can-live`, item 2.'''
    ve = np.asarray(d['plane']['plane_var_e'], float)
    sw2 = np.asarray(d['plane']['plane_sigma_w2'], float)
    sz = np.asarray(d['sigma_zeta2'], float)[:, None, None, :]
    return {'measured': np.asarray(d['head']['var_e'], float)[:, -1].mean(0), 'plane_sum': ve.sum(-1).mean(0), 'plane_bound_sum': (sw2 * sz).sum(-1).mean(0)}


def split_half(d: dict, keep: np.ndarray, n_split: int, seed: int) -> float:
    '''Mean over n_split random halves of the prompts of the Spearman correlation, over the heads in keep, between the positional ratios of the two halves.'''
    rng, E = np.random.default_rng(seed), d['head']['var_c'].shape[0]
    ratio = lambda idx: cf.positional_ratio(*(np.asarray(d['head'][k], float)[idx, -1].mean(0) for k in ('var_c', 'var_p', 'var_e')))
    vals = []
    for _ in range(n_split):
        perm = rng.permutation(E)
        vals.append(spearman(ratio(perm[:E // 2])[keep], ratio(perm[E // 2:])[keep]))
    return float(np.nanmean(vals))
