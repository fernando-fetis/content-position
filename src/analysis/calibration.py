'''The calibration pass on text windows that later experiments read: per query, layer and head, the parts of the counterfactual matrix and the attention on the first token; per KV head and plane, the plane scores and EliteKV's greedy order; and the helpers that read it.'''
from __future__ import annotations

import time
import warnings
from collections.abc import Iterator

import numpy as np
import torch

from src.analysis import counterfactual as cf
from src.analysis import plane_scores as ps
from src.core.models import Model, stream_qk
from src.core.tasks import make, render


def windows(m: Model, spec: dict, n: int, tokens: int, seed: int) -> list[torch.Tensor]:
    '''n text windows of exactly tokens tokens from a corpus span (spec: source and span).'''
    out = [render(ex, m.tokenizer, tokens) for ex in make({'task': 'corpus', 'n': 4 * n, 'chars': 8 * tokens + 4000, 'seed': seed, **spec})]
    return [w for w in out if len(w) == tokens][:n]


def query_positions(T: int, queries: int) -> np.ndarray:
    '''queries positions evenly spaced from min(64, T) to T, rounded and deduplicated.'''
    return np.unique(np.linspace(min(64, T), T, queries).round().astype(int))


def _clock(m: Model) -> float:
    '''Wall-clock seconds after the model's device has finished its queued work.'''
    if m.device.type == 'cuda':
        torch.cuda.synchronize(m.device)
    return time.perf_counter()


class _Stop(Exception):
    pass


@torch.no_grad()
def layer_qk(m: Model, ids: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    '''Unrotated queries and keys (positions, H, F, 2) of one layer, from a forward pass aborted once that layer's keys are computed.'''
    got: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def on_layer(l: int, Q: torch.Tensor, K: torch.Tensor) -> None:
        if l == layer:
            got['qk'] = (Q, K)
            raise _Stop

    with stream_qk(m, on_layer):
        try:
            m.model(ids[None].to(m.device), logits_to_keep=1)
        except _Stop:
            pass
    return got['qk']


@torch.no_grad()
def elitekv_orders(m: Model, windows: list[torch.Tensor], start: int, block: int = 256) -> np.ndarray:
    '''EliteKV's greedy order of the planes of every KV head, shape (L, KV, F): the L1 distance summed over the query heads of the KV head and all causal pairs (r >= start, s <= r) of every window. The removal matrices are rebuilt per row block at every greedy step, so at most (G, F, block, T) floats are in memory.'''
    g = m.geometry
    L, KV, G, F = g.n_layers, g.n_kv_heads, g.groups, g.n_freqs
    order = np.zeros((L, KV, F), int)
    for layer in range(L):
        qk = [layer_qk(m, ids, layer) for ids in windows]
        for h in range(KV):
            heads = slice(h * G, (h + 1) * G)

            def pieces() -> Iterator[torch.Tensor]:
                for Q, K in qk:
                    T = Q.shape[0]
                    for a in range(start, T, block):
                        b = min(T, a + block)
                        yield ps.removal_matrices(m, Q[:b, heads], K[:b, heads], a)

            order[layer, h] = ps.elitekv_order(pieces, F).numpy()
        qk.clear()
    return order


@torch.no_grad()
def calibrate(m: Model, windows: list[torch.Tensor], queries: int, elitekv_from: int | None = None) -> dict[str, np.ndarray]:
    '''Per query (query_positions of each window), layer and head, arrays (n queries, L, H): var_c, var_p, var_e; sink (attention on position 0); r (query positions). scores: the plane scores (L, KV, F), averaged over windows and queries and summed over the query heads of each KV head; elitekv_order when elitekv_from is given (cost quadratic in the window length). seconds: wall-clock cost of parts (which give variance and bound), observed (mkl, diagonal), norm and elitekv. Also theta, n_kv, tokens and queries.'''
    g = m.geometry
    F, G = g.n_freqs, g.groups
    per_query = ('var_c', 'var_p', 'var_e', 'sink')
    rows: dict[str, list] = {k: [] for k in (*per_query, 'r')}
    acc = {k: np.zeros((g.n_layers, g.n_heads, F)) for k in ('norm', *ps.QUERY_SCORES)}
    seconds = {'parts': 0.0, 'observed': 0.0, 'norm': 0.0}
    for ids in windows:
        T = len(ids) - 1
        qs = query_positions(T, queries)
        cur = {k: np.zeros((len(qs), g.n_layers, g.n_heads)) for k in per_query}

        def on_layer(layer: int, Q: torch.Tensor, K: torch.Tensor) -> None:
            t0 = _clock(m)
            acc['norm'][layer] += ps.norm_score(Q, K).cpu().numpy()
            seconds['norm'] += _clock(m) - t0
            for i, r in enumerate(qs):
                r = int(r)
                t0 = _clock(m)
                f = cf.factors(m, Q[r], K, r)
                P = cf.parts(f)
                seconds['parts'] += _clock(m) - t0
                for k in ('var_c', 'var_p', 'var_e'):
                    cur[k][i, layer] = P[k].cpu().numpy()
                cur['sink'][i, layer] = torch.softmax(cf.row_logits(m, Q[r], K, r).double(), -1)[:, 0].cpu().numpy()
                t0 = _clock(m)
                for k, v in ps.scores_at_query(m, Q, K, r, P).items():
                    acc[k][layer] += v.cpu().numpy() / len(qs)
                seconds['observed'] += _clock(m) - t0

        with stream_qk(m, on_layer):
            m.model(ids[None].to(m.device), logits_to_keep=1)
        for k in cur:
            rows[k].append(cur[k])
        rows['r'].append(qs)
    out: dict = {k: np.concatenate(v) for k, v in rows.items()}
    out['scores'] = {k: (v / len(windows)).reshape(g.n_layers, g.n_kv_heads, G, F).sum(2) for k, v in acc.items()}
    if elitekv_from is not None:
        t0 = _clock(m)
        out['elitekv_order'] = elitekv_orders(m, windows, elitekv_from)
        seconds['elitekv'] = _clock(m) - t0
    out.update(seconds=seconds, theta=g.theta().numpy(), n_kv=g.n_kv_heads, tokens=len(windows[0]), queries=queries)
    return out


def head_table(cal: dict) -> dict[str, np.ndarray]:
    '''Per layer and query head (L, H): positional ratio and interaction share (variances averaged over queries first) and mean attention on the first token.'''
    c, p, e = (np.asarray(cal[k], float).mean(0) for k in ('var_c', 'var_p', 'var_e'))
    return {'ratio': cf.positional_ratio(c, p, e), 'interaction': e / (c + p + e), 'sink': np.asarray(cal['sink'], float).mean(0)}


def kv(x: np.ndarray, n_kv: int, how: str, skip: np.ndarray | None = None, empty: float = np.inf) -> np.ndarray:
    '''Per-KV-head aggregate (flat, layer * n_kv + head) of an (L, H) array: min, max or mean over the query heads of each group, ignoring those in skip (empty when all are skipped).'''
    L, H = x.shape
    g = np.where(skip, np.nan, x) if skip is not None else x.astype(float)
    g = g.reshape(L, n_kv, H // n_kv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        r = {'min': np.nanmin, 'max': np.nanmax, 'mean': np.nanmean}[how](g, -1)
    return np.where(np.isnan(r), empty, r).ravel()
