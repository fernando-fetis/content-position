'''The part edit: in chosen query heads, at every query r, the logit of every context key s in [r-1] becomes mu_r plus some of the diagonal parts c_s, p_s, epsilon_ss of the counterfactual matrix at r (`def:content-positional-and-interaction-parts`); the first token and the query keep their logits. The parts come in closed form from running means (`prop:the-parts-of-a-plane`): with w_s = q_r conj(k_s) per plane, its center of mass w~_r over the context (q_r times the conjugate of the running mean of the keys) and the center of the phases zeta~_r, mu = Re[w~ zeta~], c_s = Re[w_s zeta~] - mu, p_s = Re[w~ zeta_s] - mu, epsilon_ss = lambda_s - mu - c_s - p_s, summed over planes and scaled; float64.'''
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analysis import counterfactual as cf
from src.core.models import Model, repeat_kv, rotate, stream_qk, use_attention
from src.edits.partial_rope import store_unrotated


def center_of_phases(theta: torch.Tensor, T: int) -> torch.Tensor:
    '''zt_r^f = mean over s = 1, ..., r - 1 of e^(i (r - s) theta_f), complex (T, F), float64 (one for r < 2, which has no context).'''
    r = torch.arange(T, dtype=torch.float64, device=theta.device)[:, None]
    n = (r - 1).clamp_min(1)
    z = torch.polar(torch.ones_like(theta), theta)
    frozen = theta == 0
    s = z * (1 - torch.polar(torch.ones_like(r * theta), (r - 1) * theta)) / torch.where(frozen, torch.ones_like(z), 1 - z)
    s = torch.where(frozen, n.to(s.dtype).expand_as(s), s)
    return torch.where(r >= 2, s / n, torch.ones_like(s))


class PartLogits:
    '''The parts of the logits of some heads at every query of one sequence, from unrotated qu, ku and rotated qr, kr, all (heads, T, head_dim); logits(a, b) gives lam, mu, c, p and e = lam - mu - c - p for the queries a, ..., b - 1 over the keys 0, ..., b - 1, (heads, b - a, b), float64.'''

    def __init__(self, qu: torch.Tensor, ku: torch.Tensor, qr: torch.Tensor, kr: torch.Tensor, theta: torch.Tensor, scaling: float) -> None:
        Fp, T = theta.shape[0], qu.shape[1]
        self.theta, self.scaling = theta.to(qu.device, torch.float64), scaling
        qu, ku = qu.double(), ku.double()
        self.qc, self.kc = torch.complex(qu[..., :Fp], qu[..., Fp:]), torch.complex(ku[..., :Fp], ku[..., Fp:])
        self.qr, self.kr = qr.double(), kr.double()
        self.r = torch.arange(T, device=qu.device)
        zt = center_of_phases(self.theta, T)
        cum = torch.cumsum(self.kc, 1)
        kbar = (cum[:, (self.r - 1).clamp_min(0)] - self.kc[:, :1]) / (self.r - 1).clamp_min(1)[None, :, None]  # mean key over the context 1, ..., r - 1
        self.wbar = self.qc * kbar.conj()
        self.mu = (self.wbar * zt[None]).real.sum(-1) * scaling  # (heads, T)
        self.qz = self.qc * zt[None]

    def logits(self, a: int, b: int) -> dict[str, torch.Tensor]:
        r, th = self.r, self.theta
        lam = self.qr[:, a:b] @ self.kr[:, :b].transpose(-1, -2) * self.scaling
        C = (self.qz[:, a:b, None] * self.kc[:, None, :b].conj()).real.sum(-1) * self.scaling
        ang = (r[a:b, None] - r[None, :b]).double()[..., None] * th[None, None]
        P = (self.wbar[:, a:b, None] * torch.polar(torch.ones_like(ang), ang)[None]).real.sum(-1) * self.scaling
        mu = self.mu[:, a:b, None]
        c, p = C - mu, P - mu
        return {'lam': lam, 'mu': mu.expand_as(lam), 'c': c, 'p': p, 'e': lam - mu - c - p}

    def edited(self, a: int, b: int, keep: str) -> torch.Tensor:
        '''The logits of the queries a, ..., b - 1 after the edit keeping the parts named in keep (letters c, p, e), -inf beyond the causal mask.'''
        parts = self.logits(a, b)
        new = parts['mu'] + sum((parts[x] for x in keep), torch.zeros_like(parts['lam']))
        r = self.r
        ctx = (r[None, :b] >= 1) & (r[None, :b] <= r[a:b, None] - 1) & (r[a:b, None] >= 2)
        return torch.where(ctx[None], new, parts['lam']).masked_fill(r[None, :b] > r[a:b, None], float('-inf'))


@contextmanager
def part_edit(m: Model, heads: dict[int, np.ndarray], keep: str, chunk: int = 64) -> Iterator[None]:
    '''Inside the context every forward pass (one sequence at a time) edits the query heads heads[layer] of each layer, keeping mu and the parts named in keep (letters c, p, e); the edited heads attend in query chunks, in float64.'''
    g = m.geometry
    if m.softcap or m.sliding_window:
        raise NotImplementedError(f'{m.name}: logit softcap or sliding-window layers are not handled by the part edit')
    if any(x not in 'cpe' for x in keep):
        raise ValueError(f'keep names the parts with the letters c, p, e, got {keep!r}')
    edited = {l: np.asarray(h, int) for l, h in heads.items() if len(h)}
    theta = g.theta()

    def attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None, dropout: float = 0.0, scaling: float | None = None, **kw: object) -> tuple[torch.Tensor, None]:
        layer = module.layer_idx
        B, H, T, D = query.shape
        scaling = scaling if scaling is not None else D ** -0.5
        G = H // key.shape[1]
        kr, v = repeat_kv(key, G), repeat_kv(value, G)
        out = F.scaled_dot_product_attention(query, kr, v, is_causal=True, scale=scaling)
        if layer not in edited:
            return out.transpose(1, 2).contiguous(), None
        if B != 1:
            raise ValueError('part_edit runs one sequence at a time')
        idx = torch.as_tensor(edited[layer], device=query.device)
        u = unrot.pop(layer)
        parts = PartLogits(u['q'][0, idx], repeat_kv(u['k'], G)[0, idx], query[0, idx], kr[0, idx], theta, scaling)
        vv = v[0, idx].double()
        o = torch.empty(len(idx), T, D, device=query.device, dtype=torch.float64)
        for a in range(0, T, chunk):
            b = min(T, a + chunk)
            o[:, a:b] = torch.softmax(parts.edited(a, b, keep), -1) @ vv[:, :b]
        out[:, idx] = o.to(out.dtype)[None]
        return out.transpose(1, 2).contiguous(), None

    with store_unrotated(m, set(edited)) as unrot, use_attention(m, attention):
        yield


@torch.no_grad()
def check_part_edit(m: Model, ids: torch.Tensor) -> float:
    '''Largest absolute deviation, over layers, heads and context keys of the last query of ids, between the closed-form parts of PartLogits and those read off the explicit matrix cf.matrix (row and column means), both in float64 (zero up to rounding, about 1e-12).'''
    g = m.geometry
    theta = g.theta()
    T = len(ids)
    r = T - 1
    worst = 0.0

    def on_layer(layer: int, Q: torch.Tensor, K: torch.Tensor) -> None:
        nonlocal worst
        q, k = Q.double(), K.double()  # (T, heads, F, 2)
        f = cf.factors(m, q[r], k, r)
        ang = torch.arange(r - 1, 0, -1, dtype=torch.float64)[:, None] * theta[None]
        A = cf.matrix(cf.Factors(f.u, torch.stack([torch.cos(ang), -torch.sin(ang)], -1)))  # (heads, N, N), N = r - 1 context positions
        mu = A.mean((1, 2))
        c, p = A.mean(2) - mu[:, None], A.mean(1) - mu[:, None]
        e = (A - mu[:, None, None] - c[:, :, None] - p[:, None, :]).diagonal(0, 1, 2)
        pos = (torch.arange(T, dtype=torch.float64)[:, None] * theta[None])[:, None, :]
        flat = lambda x: torch.cat([x[..., 0], x[..., 1]], -1).transpose(0, 1)  # (heads, T, head_dim)
        parts = PartLogits(flat(q), flat(k), flat(rotate(q, pos)), flat(rotate(k, pos)), theta, m.scaling).logits(r, r + 1)
        got = {k_: v[:, 0, 1:r] for k_, v in parts.items()}
        for ours, explicit in ((got['mu'], mu[:, None].expand(-1, r - 1)), (got['c'], c), (got['p'], p), (got['e'], e), (got['lam'], A.diagonal(0, 1, 2))):
            worst = max(worst, float((ours - explicit).abs().max()))

    with stream_qk(m, on_layer):
        m.model(ids[None].to(m.device), logits_to_keep=1, use_cache=False)
    return worst


@torch.no_grad()
def check_part_edit_identity(m: Model, ids: torch.Tensor) -> float:
    '''Largest absolute deviation of the logits of ids from the intact model's when every head keeps every part (zero up to float32 rounding, about 1e-7).'''
    g = m.geometry
    base = m.model(ids[None].to(m.device), use_cache=False).logits[0].double()
    with part_edit(m, {l: np.arange(g.n_heads) for l in range(g.n_layers)}, 'cpe', chunk=5):
        edited = m.model(ids[None].to(m.device), use_cache=False).logits[0].double()
    return float((edited - base).abs().max())
