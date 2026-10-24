'''Chen and Yan's quantities in the decomposition (`sec:chen`): per plane, the pseudo-variance of the content factors (`prop:pseudo-variance-and-pca` and `prop:the-parts-of-a-plane`) at the last query of a prompt, and the cosine between the two weight rows of the query and key projections that generate the plane (rotate_half pairing: coordinates f and f + F of a head).'''
from __future__ import annotations

import numpy as np
import torch

from src.analysis import counterfactual as cf
from src.core.models import Model, stream_qk
from src.core.suite import Prompt
from src.utils.stats import spearman


def weight_cosines(m: Model) -> dict[str, np.ndarray]:
    '''Per layer, head and plane, the cosine between the two weight rows that generate the plane: cos_q (L, H, F) from W_Q, cos_k (L, KV heads, F) from W_K.'''
    g, F = m.geometry, m.geometry.n_freqs
    out: dict[str, list] = {'cos_q': [], 'cos_k': []}
    for layer in range(g.n_layers):
        attn = m.attention(layer)
        for key, proj, heads in (('cos_q', attn.q_proj, g.n_heads), ('cos_k', attn.k_proj, g.n_kv_heads)):
            w = proj.weight.detach().double().reshape(heads, g.head_dim, -1)
            a, b = w[:, :F], w[:, F:]
            out[key].append(((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))).cpu().numpy())
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}


@torch.no_grad()
def pseudo_variances(m: Model, prompts: list[Prompt]) -> dict:
    '''Per prompt, at its last query and over the context without the first token (as the plane parts of measure): s_re, s_im, the real and imaginary parts of the pseudo-variance s_w^f = E_i[(w_i^f - w~^f)^2], and sigma_w2 = E_i |w_i^f - w~^f|^2, (prompts, L, H, F), squared logits; sigma_zeta2 (prompts, F); and the weight cosines.'''
    g = m.geometry
    rows = []
    for pr in prompts:
        T = int(pr.queries[-1])
        cur = {k: np.zeros((g.n_layers, g.n_heads, g.n_freqs)) for k in ('s_re', 's_im', 'sigma_w2')}
        sz: dict[str, np.ndarray] = {}

        def on_layer(layer: int, Q: torch.Tensor, K: torch.Tensor) -> None:
            f = cf.factors(m, Q[T], K, T)
            ut = f.u.double() - f.u.double().mean(1, keepdim=True)
            cu = torch.einsum('hifa,hifb->hfab', ut, ut) / ut.shape[1]
            cur['s_re'][layer] = (cu[..., 0, 0] - cu[..., 1, 1]).cpu().numpy()
            cur['s_im'][layer] = (2 * cu[..., 0, 1]).cpu().numpy()
            cur['sigma_w2'][layer] = (cu[..., 0, 0] + cu[..., 1, 1]).cpu().numpy()
            vt = f.v.double() - f.v.double().mean(0)
            sz['v'] = (vt ** 2).sum(-1).mean(0).cpu().numpy()

        with stream_qk(m, on_layer):
            m.model(pr.ids[None].to(m.device), logits_to_keep=1)
        rows.append({**cur, 'sigma_zeta2': sz['v']})
    st = lambda k: np.stack([r[k] for r in rows])
    return {**{k: st(k).astype(np.float32) for k in ('s_re', 's_im', 'sigma_w2')}, 'sigma_zeta2': st('sigma_zeta2'), **weight_cosines(m), 'groups': g.groups}


def pseudo_summary(d: dict, ratio: np.ndarray, keep: np.ndarray, improper: float = 0.9) -> dict:
    '''From pseudo_variances (averaged over prompts before any ratio) and, per head, the positional ratio and the non-sink mask of the same prompts: iota = |s_w^f| / sigma_w,f^2 of every (non-sink head, plane) and the share of those pairs with iota above improper (iota_share); the median absolute weight cosines; the Spearman correlation over (non-sink head, plane) of iota with the absolute key and query cosines; and over non-sink heads, of the positional ratio with the per-head mean absolute cosine (the mean of the query and key sides over planes, Chen and Yan's per-head proxy). For the figure, in deciles of the absolute key cosine over (non-sink head, plane), the median cosine and the median and quartiles of iota (cos_iota); in deciles of the positional ratio over non-sink heads, the median ratio and the median of the per-head proxy (ratio_cos).'''
    mean = lambda k: np.asarray(d[k], float).mean(0)
    iota = np.hypot(mean('s_re'), mean('s_im')) / np.maximum(mean('sigma_w2'), 1e-30)
    cos_q = np.abs(np.asarray(d['cos_q'], float))
    cos_k = np.repeat(np.abs(np.asarray(d['cos_k'], float)), int(d['groups']), axis=1)
    out: dict = {'iota_share': float(np.mean(iota[keep] > improper))}
    out['cos_q_median'], out['cos_k_median'] = float(np.median(cos_q[keep])), float(np.median(cos_k[keep]))
    out['spearman_iota_cos_k'] = spearman(iota[keep].ravel(), cos_k[keep].ravel())
    out['spearman_iota_cos_q'] = spearman(iota[keep].ravel(), cos_q[keep].ravel())
    proxy = 0.5 * (cos_q + cos_k).mean(-1)
    out['spearman_ratio_cos'] = spearman(ratio[keep], proxy[keep])
    out['cos_iota'] = deciles(cos_k[keep].ravel(), iota[keep].ravel())
    out['ratio_cos'] = deciles(ratio[keep], proxy[keep], quartiles=False)
    return out


def deciles(x: np.ndarray, y: np.ndarray, quartiles: bool = True) -> dict[str, list[float]]:
    '''In the ten deciles of x: the median of x and the 25%, 50% and 75% quantiles of y (only the 50% when not quartiles).'''
    bins = np.clip(np.searchsorted(np.quantile(x, np.linspace(0, 1, 11)[1:-1]), x, side='right'), 0, 9)
    qs = [0.25, 0.5, 0.75] if quartiles else [0.5]
    q = np.array([np.quantile(y[bins == b], qs) if (bins == b).any() else [np.nan] * len(qs) for b in range(10)])
    xs = [float(np.median(x[bins == b])) if (bins == b).any() else float('nan') for b in range(10)]
    return {'x': xs, 'q25': q[:, 0].tolist(), 'median': q[:, 1].tolist(), 'q75': q[:, 2].tolist()} if quartiles else {'x': xs, 'median': q[:, 0].tolist()}
