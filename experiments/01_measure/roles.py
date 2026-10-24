'''The induction circuit on repeated random tokens: previous-token and induction heads, the roles the positional ratio is compared with.'''
from __future__ import annotations

import numpy as np
import torch

from src.analysis import counterfactual as cf
from src.core.models import Model, override_attention, random_ids
from src.core.suite import Prompt

SHARES = {'norm': 'nu', 'position': 'pi'}  # the shares of a head's norm and position-dependent variance in its fast planes, nu_h and pi_h of `sec:barbero`


def repeated_prompts(m: Model, cfg: dict, seed: int) -> list[Prompt]:
    '''cfg['n_seq'] sequences of cfg['length'] random tokens repeated once (after BOS when there is one), with cfg['queries'] queries evenly spaced over the second copy, from its third token to its last.'''
    gen, L = torch.Generator().manual_seed(seed), cfg['length']
    seqs = [random_ids(m, gen, L, cfg['token_range'], repeats=2) for _ in range(cfg['n_seq'])]
    off = len(seqs[0]) - 2 * L
    queries = sorted({int(x) for x in np.linspace(off + L + 2, off + 2 * L - 1, cfg['queries']).round()})
    return [Prompt(ids, queries) for ids in seqs]


@torch.no_grad()
def roles(m: Model, seqs: list[torch.Tensor], length: int) -> dict[str, np.ndarray]:
    '''Per layer and head (L, H), the mean attention from the second copy to the previous token (prev_token) and to the token after the earlier occurrence of the query's token (induction).'''
    g = m.geometry
    prev, ind = np.zeros((g.n_layers, g.n_heads)), np.zeros((g.n_layers, g.n_heads))
    with override_attention(m):
        for ids in seqs:
            off = len(ids) - 2 * length
            q = torch.arange(off + length, off + 2 * length, device=m.device)
            out = m.model(ids[None].to(m.device), output_attentions=True)
            for l, a in enumerate(out.attentions):
                a = a[0].float()
                prev[l] += a[:, q, q - 1].mean(-1).cpu().numpy() / len(seqs)
                ind[l] += a[:, q, q - length + 1].mean(-1).cpu().numpy() / len(seqs)
    return {'prev_token': prev, 'induction': ind}


def role_summary(d: dict, parts: dict[str, dict[str, np.ndarray]], prev_threshold: float, induction_threshold: float, fast: float = 0.5) -> dict:
    '''From the role scores d (L, H) and, per input, the arrays var_c, var_p, var_e, sink (..., L, H): the number of previous-token heads (above prev_threshold) and, per input, the median positional ratio of the non-sink previous-token heads, induction heads (above induction_threshold, not previous-token) and other heads. When the input also has norm (|w~^f|^2 + sigma_w,f^2), position (Var(p^f) + Var(eps^f)), both (..., L, H, F), and sigma_zeta2 (..., F): per group, the median share of a head's norm and of its position-dependent variance in the planes with sigma_zeta^2 > fast (nu, pi), and the mean over the group's heads of each head's distribution of its norm and of its position-dependent variance over the planes (norm_profile, position_profile, (F,)).'''
    prev = np.asarray(d['prev_token'], float) > prev_threshold
    ind = (np.asarray(d['induction'], float) > induction_threshold) & ~prev
    out: dict = {'previous_token': int(prev.sum())}
    for name, arrays in parts.items():
        v = {k: np.asarray(arrays[k], float).reshape(-1, *prev.shape).mean(0) for k in ('var_c', 'var_p', 'var_e', 'sink')}
        ratio = cf.positional_ratio(v['var_c'], v['var_p'], v['var_e'])
        keep = v['sink'] <= cf.SINK
        shares, profiles = {}, {}
        if 'norm' in arrays:
            F = np.asarray(arrays['sigma_zeta2']).shape[-1]
            is_fast = np.asarray(arrays['sigma_zeta2'], float).reshape(-1, F).mean(0) > fast
            per_plane = {k: np.asarray(arrays[k], float).reshape(-1, *prev.shape, F).mean(0) for k in ('norm', 'position')}
            shares = {k: x[..., is_fast].sum(-1) / np.maximum(x.sum(-1), 1e-30) for k, x in per_plane.items()}
            profiles = {k: x / np.maximum(x.sum(-1, keepdims=True), 1e-30) for k, x in per_plane.items()}
        out[name] = {}
        for label, sel in (('previous_token', prev & keep), ('induction', ind & keep), ('other', keep & ~prev & ~ind)):
            x = ratio[sel]
            out[name][label] = {'median': float(np.median(x)) if x.size else float('nan'), **{SHARES[k]: float(np.median(s[sel])) if sel.any() else float('nan') for k, s in shares.items()}, **{f'{k}_profile': (x[sel].mean(0) if sel.any() else np.full(x.shape[-1], np.nan)).tolist() for k, x in profiles.items()}}
    return out
