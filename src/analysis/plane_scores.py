'''The rules that choose which planes of a KV head keep their rotation under partial RoPE: per layer, KV head and plane, the score of each rule, EliteKV's greedy order and the masks of kept planes. Query heads of a KV head are summed, since partial RoPE keeps the same planes for all of them.'''
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch

from src.analysis import counterfactual as cf
from src.core.models import Model, rotate

QUERY_SCORES = ('mkl', 'diagonal', 'bound', 'variance')


def plane_logits(m: Model, q: torch.Tensor, k: torch.Tensor, r: int) -> torch.Tensor:
    '''Per-plane logits (heads, r + 1, F) of the query at position r (unrotated q of shape (heads, F, 2)) on the positions 0, ..., r (unrotated k of shape (positions, heads, F, 2)), with the scaling.'''
    ang = (torch.arange(r + 1, dtype=torch.float64)[:, None] * m.geometry.theta()[None]).float().to(k.device)
    kr = rotate(k[:r + 1], ang[:, None, :])
    qr = rotate(q, ang[-1][None, :])
    return ((qr[None] * kr).sum(-1) * m.scaling).permute(1, 0, 2)


def norm_score(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    '''MHA2MLA's 2-norm rule (H, F): the mean over tokens of the plane's unrotated query norm times that of its key norm; Q, K (positions, H, F, 2).'''
    return (Q.norm(dim=-1).mean(0) * K.norm(dim=-1).mean(0)).double()


def scores_at_query(m: Model, Q: torch.Tensor, K: torch.Tensor, r: int, P: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    '''Plane scores at query r, per query head and plane (H, F, float64): mkl (MHA2MLA-VLM), KL(attention || attention with the plane zeroed in query and key); diagonal, the mean square change of the observed logits over the context when the plane loses its rotation; bound, (|w~^f|^2 + sigma_w,f^2) sigma_zeta,f^2, the sum of the bounds of `cor:where-each-part-can-live`, items 1 and 2; variance, Var(p^f) + Var(eps^f), the plane's position-dependent variance (`prop:the-cost-of-removing-a-rotation`, item 2). P: cf.parts at that query, computed when not given.'''
    P = P if P is not None else cf.parts(cf.factors(m, Q[r], K, r))
    lam = plane_logits(m, Q[r], K, r).double()
    total = lam.sum(-1)
    logp = torch.log_softmax(total, -1)
    masked = torch.log_softmax(total[..., None] - lam, 1)
    w = (Q[r][None, :, :, 0] * K[1:r, :, :, 0] + Q[r][None, :, :, 1] * K[1:r, :, :, 1]).transpose(0, 1).double() * m.scaling
    return {'mkl': (logp.exp()[..., None] * (logp[..., None] - masked)).sum(1), 'diagonal': ((lam[:, 1:r] - w) ** 2).mean(1), 'bound': (P['plane_w2'] + P['plane_sigma_w2']) * P['sigma_zeta2'][None], 'variance': P['plane_var_p'] + P['plane_var_e']}


def removal_matrices(m: Model, Q: torch.Tensor, K: torch.Tensor, start: int = 0) -> torch.Tensor:
    '''EliteKV's per-plane change of the scores, D^f_rs = Re[w^f_rs (zeta^f_(r-s) - 1)] times the scaling (full-RoPE score minus the score with plane f unrotated), for queries r = start, ..., T - 1 and keys s <= r (zero above the diagonal). Q, K (T, H', F, 2); output (H', F, T - start, T), float32. The scores with only the planes of E rotated differ from the full ones by the sum of D^f over f not in E.'''
    T = Q.shape[0]
    ang = (torch.arange(T, dtype=torch.float64)[:, None] * m.geometry.theta()[None]).float().to(Q.device)
    qr, kr = rotate(Q[start:], ang[start:, None, :]), rotate(K, ang[:, None, :])
    qq = torch.cat([qr, Q[start:]], -1)
    kk = torch.cat([kr, -K], -1)
    D = torch.einsum('rhfa,shfa->hfrs', qq, kk) * m.scaling
    causal = torch.arange(T, device=Q.device)[None, :] <= torch.arange(start, T, device=Q.device)[:, None]
    return D.masked_fill_(~causal, 0.0)


def _distances(D: torch.Tensor, kept: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    '''Per plane j not yet kept, the L1 norm over heads and pairs of the sum of D^f over the planes other than the kept ones and j; inf for kept planes. Output (F,), float64.'''
    F = D.shape[1]
    free = torch.nonzero(~kept).flatten()
    held = torch.nonzero(kept).flatten()
    R = D.index_select(1, free).sum(1) if free.numel() <= held.numel() else D.sum(1) - D.index_select(1, held).sum(1)
    out = torch.full((F,), float('inf'), dtype=torch.float64, device=D.device)
    for a in range(0, free.numel(), chunk):
        j = free[a:a + chunk]
        diff = R[:, None] - D.index_select(1, j)
        out[j] = diff.abs_().sum(dim=tuple(d for d in range(diff.dim()) if d != 1), dtype=torch.float64)
    return out


def elitekv_order(D: torch.Tensor | Callable[[], Iterable[torch.Tensor]], F: int | None = None) -> torch.Tensor:
    '''EliteKV's greedy order of the planes of one KV head (Zhou et al., 2025, Algorithm 1, RoPElite): from no plane, add at each step the plane that brings the scores closest in L1 to the full-RoPE scores over all pairs; the first r planes are EliteKV's r planes. D: removal_matrices of the head (query heads stacked on the first axis), or a callable yielding them in pieces, rebuilt at every step so that one piece is in memory at a time. Output (F,) long.'''
    pieces = (lambda: [D]) if isinstance(D, torch.Tensor) else D
    if F is None:
        F = next(iter(pieces())).shape[1]
    kept = torch.zeros(F, dtype=torch.bool)
    order = []
    for _ in range(F):
        dist = None
        for piece in pieces():
            d = _distances(piece, kept.to(piece.device)).cpu()
            dist = d if dist is None else dist + d
        j = int(torch.argmin(dist))
        kept[j] = True
        order.append(j)
    return torch.tensor(order)


def order_score(order: np.ndarray) -> np.ndarray:
    '''EliteKV's order (..., F) as a score: F minus the step at which each plane was added, so the r highest scores are the first r of the order.'''
    order = np.asarray(order, int)
    F = order.shape[-1]
    score = np.zeros(order.shape, float)
    np.put_along_axis(score, order, F - np.arange(F, dtype=float), -1)
    return score


def kept(scores: dict[str, np.ndarray], rule: str, r: int, F: int) -> np.ndarray:
    '''Boolean mask (layers, KV heads, F) of the r planes that keep their rotation under a rule: high, the r fastest planes; elitekv, the first r planes of the greedy order (scores['elitekv_order'], (layers, KV heads, F)); any other rule, the r planes of largest score in scores[rule].'''
    L, H = next(iter(scores.values())).shape[:2]
    out = np.zeros((L, H, F), bool)
    if rule == 'high':
        out[..., :r] = True
    elif rule == 'elitekv':
        np.put_along_axis(out, np.asarray(scores['elitekv_order'], int)[..., :r], True, -1)
    else:
        top = np.argsort(-scores[rule], -1)[..., :r]
        np.put_along_axis(out, top, True, -1)
    return out
