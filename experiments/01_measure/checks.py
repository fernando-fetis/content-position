'''The identity checks of 01_measure: the closed-form parts and deviations against the explicit counterfactual matrix, per head and per plane; the phase variance against its closed form; the query's attention against the model's; the permutation means of `prop:deviations-in-terms-of-the-parts` (items 2 and 3) over all 8! orders of a context of 8; and EliteKV's removal matrices and greedy order.'''
from __future__ import annotations

import itertools

import torch

from src.analysis import counterfactual as cf
from src.analysis import plane_scores as ps
from src.analysis.calibration import layer_qk
from src.core.models import Model, override_attention, stream_qk
from src.core.suite import prompts


def report(label: str, err: float, tol: float) -> None:
    print(f'{"ok  " if err <= tol else "FAIL"} {label}: {err:.2e} (tolerance {tol:.0e})', flush=True)
    if not err <= tol:
        raise SystemExit(f'check failed: {label}, deviation {err:.2e} above {tol:.0e}')


def _explicit(A: torch.Tensor) -> dict[str, torch.Tensor]:
    '''The parts of explicit matrices A (..., N, N) by their definitions: mu, var_c, var_p, var_e, var_obs, d_pos (from A_ij - A_jj), d_sym (from A_ij - A_ii) and gamma (their mean product); shapes (...,).'''
    mu = A.mean((-2, -1))
    c = A.mean(-1) - mu[..., None]
    p = A.mean(-2) - mu[..., None]
    e = A - mu[..., None, None] - c[..., :, None] - p[..., None, :]
    diag = A.diagonal(dim1=-2, dim2=-1)
    dp, ds = A - diag[..., None, :], A - diag[..., :, None]
    return {'mu': mu, 'var_c': (c ** 2).mean(-1), 'var_p': (p ** 2).mean(-1), 'var_e': (e ** 2).mean((-2, -1)), 'var_obs': diag.var(-1, unbiased=False), 'd_pos': (dp ** 2).mean((-2, -1)), 'd_sym': (ds ** 2).mean((-2, -1)), 'gamma': (dp * ds).mean((-2, -1))}


@torch.no_grad()
def parts_vs_matrix(m: Model, ids: torch.Tensor) -> dict[str, float]:
    '''At the last query of ids, over layers and heads, the largest deviation of: closed-form parts and deviations from the explicit matrix's, relative to Var(A) (parts); per-plane parts, |w~^f|^2 and sigma_w,f^2 from the explicit plane matrices, relative to E_i |w_i^f|^2 (planes); sigma_zeta,f^2 from 1 - delta_(T-1)(theta_f)^2 of `prop:phase-statistics` (phases); the diagonal from the model's row logits, relative to their largest magnitude (diagonal); cf.attention from the model's attention weights (attention).'''
    T, N = len(ids) - 1, len(ids) - 2
    th = m.geometry.theta()
    dirichlet = torch.sin(N * th / 2) / (N * torch.sin(th / 2))
    errs = {'parts': 0.0, 'planes': 0.0, 'phases': 0.0, 'diagonal': 0.0}
    att = {}

    def on_layer(layer: int, Q: torch.Tensor, K: torch.Tensor) -> None:
        f = cf.factors(m, Q[T], K, T)
        P, A = cf.parts(f), cf.matrix(f)
        X = _explicit(A)
        scale = X['var_c'] + X['var_p'] + X['var_e']
        errs['parts'] = max(errs['parts'], max(float(((P[k] - X[k]).abs() / scale).max()) for k in ('mu', 'var_c', 'var_p', 'var_e', 'var_obs', 'd_pos', 'd_sym')))
        u = f.u.double()
        Xf = _explicit(torch.einsum('hifa,jfa->hfij', u, f.v.double()))
        plane = {'plane_var_c': Xf['var_c'], 'plane_var_p': Xf['var_p'], 'plane_var_e': Xf['var_e'], 'plane_w2': (u.mean(1) ** 2).sum(-1), 'plane_sigma_w2': ((u - u.mean(1, keepdim=True)) ** 2).sum(-1).mean(1)}
        w2 = (u ** 2).sum(-1).mean(1)
        errs['planes'] = max(errs['planes'], max(float(((P[k] - v).abs() / w2).max()) for k, v in plane.items()))
        errs['phases'] = max(errs['phases'], float((P['sigma_zeta2'].cpu() - (1 - dirichlet ** 2)).abs().max()))
        lam = cf.row_logits(m, Q[T], K, T, softcap=False).double()[:, 1:T]
        errs['diagonal'] = max(errs['diagonal'], float((A.diagonal(dim1=-2, dim2=-1) - lam).abs().max() / lam.abs().max()))
        att[layer] = cf.attention(m, Q[T], K, T)

    with stream_qk(m, on_layer), override_attention(m):
        out = m.model(ids[None].to(m.device), logits_to_keep=1, output_attentions=True)
    errs['attention'] = max(float((att[l] - a[0, :, T, :T + 1].double()).abs().max()) for l, a in enumerate(out.attentions))
    return errs


@torch.no_grad()
def permutation_means(m: Model, ids: torch.Tensor, N: int = 8, chunk: int = 4096) -> tuple[float, float]:
    '''For the query at position N + 1 of ids, over layers, heads and all N! orders of the contents (rows of the explicit matrix permuted, float64): the largest deviation, relative to Var(A), of the identity V = (Dpos + Dsym) / 2 - Gamma over every order, with Gamma = E_ij[d_pos d_sym], and of E_tau[Dpos], E_tau[Dsym], E_tau[Gamma] from 2 (Var c + Var e), 2 (Var p + Var e) and N / (N - 1) Var e, which together give E_tau[V] of `prop:deviations-in-terms-of-the-parts`, item 3.'''
    perms = torch.tensor(list(itertools.permutations(range(N))), device=m.device)
    errs = {'identity': 0.0, 'means': 0.0}

    def on_layer(layer: int, Q: torch.Tensor, K: torch.Tensor) -> None:
        A = cf.matrix(cf.factors(m, Q[N + 1], K, N + 1))
        P = _explicit(A)
        scale = P['var_c'] + P['var_p'] + P['var_e']
        sums = {k: torch.zeros_like(scale) for k in ('d_pos', 'd_sym', 'gamma')}
        for a in range(0, perms.shape[0], chunk):
            B = A[:, perms[a:a + chunk]]
            X = _explicit(B)
            errs['identity'] = max(errs['identity'], float(((X['var_obs'] - 0.5 * (X['d_pos'] + X['d_sym']) + X['gamma']).abs() / scale[:, None]).max()))
            for k in sums:
                sums[k] += X[k].sum(1)
        means = {k: v / perms.shape[0] for k, v in sums.items()}
        targets = {'d_pos': 2 * (P['var_c'] + P['var_e']), 'd_sym': 2 * (P['var_p'] + P['var_e']), 'gamma': N / (N - 1) * P['var_e']}
        errs['means'] = max(errs['means'], max(float(((means[k] - targets[k]).abs() / scale).max()) for k in means))

    with stream_qk(m, on_layer):
        m.model(ids[:N + 2][None].to(m.device), logits_to_keep=1)
    return errs['identity'], errs['means']


@torch.no_grad()
def elitekv_checks(m: Model, ids: torch.Tensor, seed: int) -> float:
    '''EliteKV: returns the deviation of the last layer's removal matrices, summed over planes, from the full-RoPE minus the unrotated scores (relative to the largest score); fails when the greedy order on random matrices differs from a brute-force greedy, whole or in pieces.'''
    g = m.geometry
    Q, K = layer_qk(m, ids, g.n_layers - 1)
    T = Q.shape[0]
    D = ps.removal_matrices(m, Q, K, 0)
    err = 0.0
    for r in sorted({1, 5, T // 2, T - 1}):
        full = cf.row_logits(m, Q[r], K, r, softcap=False).double()
        plain = ((Q[r][None] * K[:r + 1]).sum((-2, -1)).T * m.scaling).double()
        err = max(err, float((D[:, :, r, :r + 1].double().sum(1) - (full - plain)).abs().max() / full.abs().max()))
    gen = torch.Generator().manual_seed(seed)
    D5 = torch.randn(2, 5, 9, 9, generator=gen)
    chosen: list[int] = []
    while len(chosen) < 5:
        free = [j for j in range(5) if j not in chosen]
        chosen.append(min(free, key=lambda j: float(D5[:, [f for f in free if f != j]].sum(1).abs().sum()) if len(free) > 1 else 0.0))
    whole, pieces = ps.elitekv_order(D5).tolist(), ps.elitekv_order(lambda: [D5[:, :, :4], D5[:, :, 4:]], 5).tolist()
    if whole != chosen or pieces != chosen:
        raise SystemExit(f'check failed: EliteKV greedy {whole} (pieces {pieces}) differs from the brute-force greedy {chosen}')
    return err


def run_checks(m: Model, cfg: dict) -> None:
    '''Every check on one model; tolerances 1e-5 for the closed forms (float32 intermediates), 1e-9 for the permutation identities (float64), 1e-4 for the float32 rotations.'''
    short = prompts('retrieval', m.tokenizer, 1, cfg['seed'])[0].ids
    text = prompts('wiki', m.tokenizer, 1, cfg['seed'], max_tokens=128)[0].ids
    errs = parts_vs_matrix(m, short)
    report(f'{m.name}: closed-form parts and deviations against the explicit matrix ({len(short) - 1} tokens), relative to Var(A)', errs['parts'], 1e-5)
    report(f'{m.name}: closed-form parts of every plane, |w~|^2 and sigma_w^2 against the explicit plane matrices, relative to E|w|^2', errs['planes'], 1e-5)
    report(f'{m.name}: variance of the phases against 1 - delta_(T-1)(theta)^2', errs['phases'], 1e-5)
    report(f'{m.name}: diagonal of the matrix against the row logits, relative', errs['diagonal'], 1e-4)
    report(f'{m.name}: attention of the query against the attention weights of the model', errs['attention'], 1e-5)
    identity, means = permutation_means(m, short)
    report(f'{m.name}: exclusion identity V = (Dpos + Dsym) / 2 - Gamma over all 8! orders, relative to Var(A)', identity, 1e-9)
    report(f'{m.name}: permutation means of Dpos, Dsym and Gamma against 2 (Var c + Var e), 2 (Var p + Var e), N / (N - 1) Var e', means, 1e-9)
    report(f'{m.name}: EliteKV removal matrices sum to the full change of the scores ({len(text)} tokens), relative; greedy order equal to the brute-force greedy', elitekv_checks(m, text, cfg['seed']), 1e-4)
