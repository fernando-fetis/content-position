'''The counterfactual logit matrix of every head at one query in factored form, A_ij = sum_f u_i^f . v_j^f: u_i^f the content factor w_i^f = q^f conj(k^f(x_i)) as a real 2-vector (scaled), v_j^f the phase zeta_j^f = e^(i (T - j) theta_f) as (cos, -sin). Every part follows from the means and covariances of u and v, in O(N F^2) instead of O(N^2).'''
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.core.models import Model, rotate


@dataclass
class Factors:
    '''Content factors u (heads, N, F, 2) and phases v (N, F, 2) of the context positions 1, ..., T - 1 (N = T - 1) for a query at position T.'''
    u: torch.Tensor
    v: torch.Tensor


def phases(m: Model, T: int, device: torch.device | None = None) -> torch.Tensor:
    '''(cos, -sin) of (T - j) theta_f for j = 1, ..., T - 1, shape (N, F, 2); angles in float64, since float32 resolves them only to about 1e-3 rad at 30k tokens.'''
    device = device or m.device
    dev = 'cpu' if device.type == 'mps' else device
    ang = torch.arange(T - 1, 0, -1, dtype=torch.float64, device=dev)[:, None] * m.geometry.theta().to(dev)[None]
    return torch.stack([torch.cos(ang), -torch.sin(ang)], -1).float().to(device)


def factors(m: Model, q: torch.Tensor, k: torch.Tensor, T: int) -> Factors:
    '''Factors of the head's counterfactual matrix from the unrotated query q (heads, F, 2) at position T and the unrotated keys k (positions, heads, F, 2) of the input.'''
    kc = k[1:T]
    re = q[None, ..., 0] * kc[..., 0] + q[None, ..., 1] * kc[..., 1]
    im = q[None, ..., 1] * kc[..., 0] - q[None, ..., 0] * kc[..., 1]
    return Factors(torch.stack([re, im], -1).transpose(0, 1) * m.scaling, phases(m, T, k.device))


def matrix(f: Factors) -> torch.Tensor:
    '''The explicit matrix (heads, N, N), A[h, i, j] the logit of content 1 + i at position 1 + j; for checks and small contexts.'''
    return torch.einsum('hifa,jfa->hij', f.u.double(), f.v.double())


def parts(f: Factors) -> dict[str, torch.Tensor]:
    '''Per head (heads,): mu, var_c, var_p, var_e, the variance var_obs of the observed logits and the deviations d_pos, d_sym (`prop:deviations-in-terms-of-the-parts`, item 1). Per head and plane (heads, F): plane_var_c, plane_var_p, plane_var_e, plane_w2 = |w~^f|^2 and plane_sigma_w2. Per plane (F,): sigma_zeta2. mu in logits, the rest in squared logits; float64. The plane quantities of the content factors are those of the scaled u, scaling^2 times the document's (A = scaling sum_f A^f).'''
    u, v = f.u, f.v
    n = v.shape[0]
    ub, vb = u.double().mean(1), v.double().mean(0)
    ut, vt = u - ub[:, None].float(), v - vb.float()
    c = torch.einsum('hifa,fa->hi', ut, vb.float()).double()
    p = torch.einsum('hfa,jfa->hj', ub.float(), vt).double()
    e_diag = torch.einsum('hjfa,jfa->hj', ut, vt).double()
    gu = torch.einsum('hia,hib->hab', ut.flatten(2), ut.flatten(2)).double() / n
    gv = torch.einsum('ja,jb->ab', vt.flatten(1), vt.flatten(1)).double() / n
    out = {'mu': (ub * vb).sum((-2, -1)), 'var_c': (c ** 2).mean(1), 'var_p': (p ** 2).mean(1), 'var_e': (gu * gv).sum((-2, -1)), 'var_obs': (c + p + e_diag).var(1, unbiased=False)}
    out['d_pos'] = out['var_c'] + out['var_e'] + ((c + e_diag) ** 2).mean(1)
    out['d_sym'] = out['var_p'] + out['var_e'] + ((p + e_diag) ** 2).mean(1)
    cu = torch.einsum('hifa,hifb->hfab', ut, ut).double() / n
    cv = torch.einsum('jfa,jfb->fab', vt, vt).double() / n
    out['plane_var_c'] = torch.einsum('fa,hfab,fb->hf', vb, cu, vb)
    out['plane_var_p'] = torch.einsum('hfa,fab,hfb->hf', ub, cv, ub)
    out['plane_var_e'] = torch.einsum('hfab,fab->hf', cu, cv)
    out['plane_w2'] = (ub ** 2).sum(-1)
    out['plane_sigma_w2'] = cu.diagonal(0, -2, -1).sum(-1)
    out['sigma_zeta2'] = cv.diagonal(0, -2, -1).sum(-1)
    return out


def row_logits(m: Model, q: torch.Tensor, k: torch.Tensor, T: int, softcap: bool = True) -> torch.Tensor:
    '''Logits of the query at position T over the positions 0, ..., T, shape (heads, T + 1), with the model's softcap unless softcap is False.'''
    pos = (torch.arange(T + 1, dtype=torch.float64)[:, None] * m.geometry.theta()[None]).float().to(q.device)
    kr = rotate(k[:T + 1], pos[:, None, :])
    qr = rotate(q, pos[-1][None, :])
    lg = (qr[None] * kr).sum((-2, -1)).T * m.scaling
    return m.softcap * torch.tanh(lg / m.softcap) if softcap and m.softcap else lg


def attention(m: Model, q: torch.Tensor, k: torch.Tensor, T: int) -> torch.Tensor:
    '''Attention of the query at position T over the positions 0, ..., T, shape (heads, T + 1), float64.'''
    if m.sliding_window is not None and T >= m.sliding_window:
        raise ValueError(f'query at position {T} beyond the sliding window of {m.sliding_window}, which these logits do not mask')
    return torch.softmax(row_logits(m, q, k, T).double(), -1)


SINK = 0.8  # a head is a sink head when its attention on the first token exceeds SINK


def positional_ratio(var_c: object, var_p: object, var_e: object) -> object:
    '''P_h = (Var p + Var e) / (Var c + Var p + 2 Var e) (`def:behavior-and-positional-ratio`); arrays or tensors, elementwise.'''
    return (var_p + var_e) / (var_c + var_p + 2 * var_e)
