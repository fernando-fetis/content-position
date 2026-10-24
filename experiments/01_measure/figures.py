'''Figures of 01_measure, drawn from results/summary.json only, in the order the document uses them: single planes (bounds, phase moments, fractions, budget), from planes to heads (phase covariance, aggregation, interaction), behavior (simplex, stability), long contexts, roles, Urrutia et al. and Chen and Yan.'''
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyses import SHORT
from planes import FAST, SLOW, phase_covariance
from src.utils.plots import BACKGROUND, FS_SMALL, FS_TINY, GUIDE, INK, LIGHT, LW, LW_BOLD, LW_HAIR, LW_THIN, MESH_LW, MID, MS, MS_SMALL, NEUTRAL, PART_COLORS, SCATTER_SMALL, SEQ_CMAP, SOFT, figure, model_color, model_label, ordered, save_figure

VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
PARTS = ('content', 'positional', 'interaction')
TINT = 0.22  # alpha of a part's colour where it marks room (a bound) or a spread (quartiles over heads)
ZONE_ALPHA = 0.13
GROUPS = ('short', 'text', 'long_4000', 'long_16000', 'long_31000')
KINDS = {'short': 'short tasks', 'text': 'texts', 'long': 'long tasks'}
CASE_LABELS = {'index': 'index', 'retrieval': 'retrieval', 'fixed_point': 'fixed point', 'wiki': 'WikiText-2'}


def _depth(shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(np.linspace(0, 1, shape[0])[:, None], shape)


def curves(theta: np.ndarray, n: int) -> dict[str, np.ndarray]:
    '''`prop:phase-statistics` on n positions: |zeta~| = |delta_n(theta)|, sigma_zeta^2 = 1 - delta_n(theta)^2 and |s_zeta| = |delta_n(2 theta) - delta_n(theta)^2|.'''
    D = lambda t: np.sin(n * t / 2) / (n * np.sin(t / 2))
    d = D(theta)
    return {'center': abs(d), 'variance': 1 - d ** 2, 'pseudo': abs(D(2 * theta) - d ** 2)}


def zones(ax: plt.Axes, slow: tuple[float, float], fast: tuple[float, float]) -> None:
    '''Shades the slow (sigma_zeta^2 < 0.1, green) and fast (sigma_zeta^2 > 0.9, hatched) ranges of the x axis, as the synthetic dispersion figure.'''
    ax.axvspan(*slow, color=PART_COLORS['content'], alpha=ZONE_ALPHA, lw=0, zorder=0)
    ax.axvspan(*fast, fc=(*to_rgb(PART_COLORS['positional']), ZONE_ALPHA), ec=(*to_rgb(PART_COLORS['interaction']), 0.35), hatch='///', lw=0, zorder=0)


def zone_handles() -> list:
    return [plt.Rectangle((0, 0), 1, 1, color=PART_COLORS['content'], alpha=ZONE_ALPHA, lw=0), plt.Rectangle((0, 0), 1, 1, fc=(*to_rgb(PART_COLORS['positional']), ZONE_ALPHA), ec=(*to_rgb(PART_COLORS['interaction']), 0.35), hatch='///', lw=0)]


def row_major(handles: list, labels: list, ncols: int) -> tuple[list, list]:
    '''Reorders legend entries so that a legend of ncols columns reads by rows.'''
    rows = -(-len(handles) // ncols)
    idx = [r * ncols + c for c in range(ncols) for r in range(rows) if r * ncols + c < len(handles)]
    return [handles[i] for i in idx], [labels[i] for i in idx]


def turns_label(t: float) -> str:
    return f'{t:.0f}' if t >= 10 else f'{t:.2g}'


def crossings(x: np.ndarray, sz: np.ndarray, increasing: bool) -> tuple[float, float]:
    '''The first x where sigma_zeta^2 crosses SLOW and FAST (x increasing with the turns when increasing, else with the period).'''
    order = slice(None) if increasing else slice(None, None, -1)
    x, sz = x[order], sz[order]
    return x[np.argmax(sz > SLOW)], x[np.argmax(sz > FAST)]


def _group(task: dict) -> str:
    return 'short' if task['category'] in SHORT else ('text' if task['category'] == 'text' else f"long_{task['target']}")


def _kind(task: dict) -> str:
    return 'short' if task['category'] in SHORT else task['category']


def _density(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    '''Gaussian kernel density of x on grid (Silverman bandwidth), normalized to a peak of one.'''
    bw = 0.9 * min(x.std(), np.subtract(*np.quantile(x, [0.75, 0.25])) / 1.34) * x.size ** -0.2
    d = np.zeros_like(grid)
    for chunk in np.array_split(x, max(1, x.size // 4000)):
        d += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bw) ** 2).sum(1)
    return d / d.max()


def _pooled(s: dict, models: list[str], select, value) -> np.ndarray:
    '''value(case summary, heads_extra case) over the non-sink heads of every model and every case c with select(c).'''
    ex = s['heads_extra']['cases']
    return np.concatenate([np.asarray(value(s['cases'][m][c], ex[m][c]), float)[np.asarray(s['cases'][m][c]['non_sink'], bool)] for m in models for c in s['cases'][m] if select(c) and c in ex.get(m, {})])


def plane_bounds_real(e: dict, out: Path) -> None:
    '''`cor:where-each-part-can-live` on three measured planes of one model (head means over non-sink heads), the closest to sigma_zeta^2 = 0.2, 0.5 and 0.85, as a mosaic: columns |w~^f|^2 and sigma_w,f^2, rows |zeta~^f|^2 and sigma_zeta,f^2, all over the budget E_i |w_i^f|^2, so that each cell's area is the bound of one part (the constant's, of mu^2, in grey); inside each tinted cell, the measured variance of that part as a solid area from the bottom and, printed, its ratio to the bound.'''
    b = e['bounds']
    fig = figure(2.05)
    axs = fig.subplots(1, len(b['planes']))
    for i, ax in enumerate(axs):
        sz, wt2 = b['sigma_zeta2'][i], b['w2_share'][i]
        sw2 = 1 - wt2
        ax.add_patch(plt.Rectangle((0, 0), wt2, 1 - sz, fc=LIGHT, ec=BACKGROUND, lw=LW))
        for x, y, w, h, part, key in ((wt2, 0, sw2, 1 - sz, 'content', 'c'), (0, 1 - sz, wt2, sz, 'positional', 'p'), (wt2, 1 - sz, sw2, sz, 'interaction', 'e')):
            color, r = PART_COLORS[part], b[f'ratio_{key}'][i]
            ax.add_patch(plt.Rectangle((x, y), w, h, fc=(*to_rgb(color), TINT), ec=BACKGROUND, lw=LW))
            ax.add_patch(plt.Rectangle((x, y + h * (1 - r)), w, h * r, fc=color, ec='none', lw=0))
            ax.add_patch(plt.Rectangle((x, y), w, h, fc='none', ec=BACKGROUND, lw=LW))
            if h > 0.06:
                ax.text(x + w / 2, y + h - min(h * r, 0.5) / 2, f'{r:.2f}', ha='center', va='center', fontsize=FS_TINY, color=BACKGROUND)
        ax.set_xlim(0, 1), ax.set_ylim(1, 0), ax.set_aspect('equal')
        ax.set_xticks([wt2 / 2, wt2 + sw2 / 2], [r'$|\tilde w^f|^2$', r'$\sigma_{w,f}^2$'], fontsize=FS_SMALL)
        ax.set_yticks([(1 - sz) / 2, 1 - sz / 2], [r'$|\tilde\zeta^f|^2$', r'$\sigma_{\zeta,f}^2$'], fontsize=FS_SMALL)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        ax.set_title(f'({"abc"[i]}) $f = {b["planes"][i] + 1}$, ' + rf'$\sigma_{{\zeta,f}}^2 = {sz:.2f}$', fontsize=FS_SMALL)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=LIGHT), plt.Rectangle((0, 0), 1, 1, fc=(*to_rgb(NEUTRAL), TINT)), plt.Rectangle((0, 0), 1, 1, fc=NEUTRAL)]
    fig.legend(handles, ['bound of the constant', 'bound of a part (cell)', 'measured variance, with its ratio to the bound'], loc='outside lower center', ncols=3, fontsize=FS_SMALL, frameon=False, handlelength=1.0, columnspacing=1.5)
    save_figure(fig, out / 'plane_bounds_real.pdf')


def phase_moments_real(e: dict, models: list[str], out: Path) -> None:
    '''`prop:phase-statistics` on the planes of the seven models, with their real frequencies, as the synthetic dispersion figure: (a) the normalized Dirichlet kernel delta_N(u) for several N = T - 1; then one row per context (WikiText-2, about 1024 tokens; the single needle at 31000 tokens) and one column per statement of the lemma, against theta_f: |zeta~^f| = |delta_(T-1)(theta_f)| and |s_zeta^f| from theta_f and the mean length of the case, sigma_zeta^2 measured over the positions of each prompt (mean over prompts); the curves of the lemma at the mean length, 2 pi / (T - 1) (dashed) and the slow and fast zones.'''
    z = e['zones']
    cases = list(z[models[0]])
    fig = figure(3.2)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 1, 1, 1])
    k = fig.add_subplot(gs[:, 0])
    u = np.linspace(-2 * np.pi + 1e-3, 2 * np.pi - 1e-3, 2000)
    for N, c in zip((4, 8, 32), ordered(3)):
        k.plot(u, np.sin(N * u / 2) / (N * np.sin(u / 2)), color=c, lw=LW, label=f'$T - 1 = {N}$')
    k.axhline(0, color=GUIDE, lw=LW_THIN)
    k.set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi], [r'$-2\pi$', r'$-\pi$', '0', r'$\pi$', r'$2\pi$'])
    k.set_xlabel('$u$'), k.set_title(r'(a) $\delta_{T-1}(u)$', fontsize=FS_SMALL)
    k.legend(fontsize=FS_TINY, frameon=False, loc='upper right')
    lo = min(e['theta'][m].min() for m in models)
    grid = np.logspace(np.log10(lo / 3), 0, 1200)
    letters = iter('bcdefg')
    name = lambda c: 'WikiText-2' if c == 'wiki' else f'needle, {int(c.rsplit("_", 1)[1]) // 1000}k tokens'
    for r, c in enumerate(cases):
        n = e['zone_context'][c]['positions']
        cv = curves(grid, n)
        slow, fast = crossings(grid, cv['variance'], True)
        for col, (key, title) in enumerate((('center', r'$|\tilde\zeta^f|$'), ('sigma_zeta2', r'$\sigma_{\zeta,f}^2$'), ('pseudo', r'$|s_\zeta^f|$'))):
            ax = fig.add_subplot(gs[r, col + 1])
            zones(ax, (grid[0], slow), (fast, grid[-1]))
            ax.semilogx(grid, cv['variance' if key == 'sigma_zeta2' else key], color=NEUTRAL, lw=LW_THIN, zorder=2)
            for m in models:
                ax.plot(e['theta'][m], z[m][c][key], 'o', ms=MS_SMALL, mew=0, color=model_color(m), alpha=0.85, zorder=3)
            ax.axvline(2 * np.pi / n, color=GUIDE, lw=LW_THIN, ls='--', zorder=1)
            ax.set_xlim(grid[0], grid[-1]), ax.set_ylim(-0.03, 1.08)
            ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
            ax.set_title(f'({next(letters)}) {title}, $T - 1 {"=" if e['zone_context'][c]['same'] else chr(92) + "approx"} {n}$', fontsize=FS_SMALL)
            if col == 0:
                ax.set_ylabel(name(c))
            if r == len(cases) - 1:
                ax.set_xlabel(r'$\theta_f$')
    handles = [Line2D([], [], ls='', marker='o', ms=MS, mew=0, color=model_color(m)) for m in models] + [Line2D([], [], color=NEUTRAL, lw=LW_THIN)] + zone_handles()
    labels = [model_label(m) for m in models] + ['closed form', r'slow planes, $\sigma_{\zeta,f}^2 < 0.1$', r'fast planes, $\sigma_{\zeta,f}^2 > 0.9$']
    fig.legend(*row_major(handles, labels, 5), loc='outside lower center', ncols=5, fontsize=FS_SMALL, frameon=False, handletextpad=0.3, columnspacing=1.2)
    save_figure(fig, out / 'phase_moments_real.pdf')


def plane_fractions_ci(e: dict, models: list[str], out: Path) -> None:
    '''The parts of every plane on WikiText-2 against its period over the context, over the non-sink (head, plane) pairs of the seven models pooled in bins. (a) Each part's fraction of D = Var(c^f) + Var(p^f) + Var(eps^f): interquartile range (band), median (bold) and the median over heads of each model (thin). (b) Each part's variance over its bound of `cor:where-each-part-can-live`: interquartile range and median, with the interval (1 -+ |s_zeta^f| / sigma_zeta,f^2) / 2 of the positional and interaction ratios, from `prop:phase-statistics` at the mean length (grey).'''
    fr = e['fractions']
    pool = fr['pooled']
    n = int(round(fr['length_mean'])) - 2
    xs = np.logspace(-2.5, 4.5, 800)
    cv = curves(2 * np.pi / (xs * n), n)
    slow, fast = crossings(xs, cv['variance'], False)
    fig = figure(2.25)
    a, b = fig.subplots(1, 2)
    x = pool['period_over_context']
    half = np.clip(cv['pseudo'] / np.maximum(cv['variance'], 1e-12), 0, 1) / 2
    b.fill_between(xs, 0.5 - half, 0.5 + half, color=LIGHT, lw=0, zorder=0.6)
    for ax in (a, b):
        zones(ax, (slow, xs[-1]), (xs[0], fast))
    for k, part in enumerate(PARTS):
        color = PART_COLORS[part]
        for m in models:
            d = fr['models'][m]
            a.plot(d['period_over_context'], d['fraction_q50'][k], color=color, lw=LW_HAIR, alpha=0.45, zorder=2)
        for ax, key in ((a, 'fraction'), (b, 'ratio')):
            ax.fill_between(x, pool[f'{key}_q25'][k], pool[f'{key}_q75'][k], color=color, alpha=TINT, lw=0, zorder=1)
            ax.plot(x, pool[f'{key}_q50'][k], color=color, lw=LW_BOLD, zorder=3)
    for ax, title, ylabel in ((a, r'(a) fractions $\psi_K^f$', r'share of $\mathrm{Var}(c^f) + \mathrm{Var}(p^f) + \mathrm{Var}(\epsilon^f)$'), (b, '(b) ratios to the bounds $R_K^f$', 'variance / bound')):
        ax.set_xscale('log'), ax.set_xlim(xs[0], xs[-1]), ax.set_ylim(0, 1)
        ax.set_xlabel('period of the plane / context length'), ax.set_ylabel(ylabel), ax.set_title(title, fontsize=FS_SMALL)
    b.axhline(0.5, color=GUIDE, lw=LW_HAIR, ls=':', zorder=0.7)
    handles = [Line2D([], [], color=PART_COLORS[p], lw=LW_BOLD) for p in PARTS] + [plt.Rectangle((0, 0), 1, 1, fc=NEUTRAL, alpha=TINT, lw=0), Line2D([], [], color=NEUTRAL, lw=LW_HAIR), plt.Rectangle((0, 0), 1, 1, fc=LIGHT, lw=0)] + zone_handles()
    labels = ['content', 'position', 'interaction', 'quartiles over heads', 'median, each model', r'$\frac{1}{2}(1 \pm |s_\zeta^f| / \sigma_{\zeta,f}^2)$', 'slow planes', 'fast planes']
    fig.legend(handles, labels, loc='outside lower center', ncols=8, fontsize=FS_SMALL, frameon=False, handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
    save_figure(fig, out / 'plane_fractions_ci.pdf')


def budget_bars(ax: plt.Axes, e: dict, title: str) -> tuple[list, list]:
    '''`cor:slow-planes-and-long-contexts` in a short and a long context on measured planes of one model (head means over non-sink heads) on the single needle: per plane, from slow to fast, the budget E_i |w_i^f|^2 split into position, interaction, content and the rest (squared mean and what the real part loses) at the shortest and longest length, with the interval (sigma_zeta^2 -+ |s_zeta^f|) / 2 of the corollary under each bar and the position-dependent share printed; returns the legend handles and labels.'''
    b = e['budget']
    lengths = [str(L) for L in b['lengths']]
    planes = b['planes']
    bar, gap = 0.34, 0.5
    colors = (PART_COLORS['positional'], PART_COLORS['interaction'], PART_COLORS['content'], LIGHT)
    yticks, ylabels = [], []
    for k, f in enumerate(planes):
        for i, L in enumerate(lengths):
            d = b['by_length'][L]
            y = k * (2 * bar + gap + 0.2) + i * (bar + 0.2)
            left = 0.0
            for share, color in zip((d['share_p'][k], d['share_e'][k], d['share_c'][k], d['share_rest'][k]), colors):
                ax.barh(y, share, left=left, height=bar, color=color, lw=0)
                left += share
            lo, hi = d['lo'][k], d['hi'][k]
            yb = y + bar / 2 + 0.07
            ax.plot([lo, hi], [yb, yb], color=INK, lw=LW, solid_capstyle='butt')
            for x in (lo, hi):
                ax.plot([x, x], [yb - 0.05, yb + 0.05], color=INK, lw=LW_THIN)
            ax.text(1.02, y, f'{d["phi"][k]:.3f}', ha='left', va='center', fontsize=FS_TINY)
            yticks.append(y)
            ylabels.append(f'{int(L) // 1000}k: {turns_label(d["turns"][k])} turns')
        ax.text(-0.31, k * (2 * bar + gap + 0.2) + (bar + 0.2) / 2, f'$f = {f + 1}$', ha='right', va='center', fontsize=FS_SMALL, transform=ax.get_yaxis_transform())
    ax.set_yticks(yticks, ylabels, fontsize=FS_SMALL)
    ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1), ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.axvline(0.5, color=GUIDE, lw=LW_HAIR, ls=':', zorder=0)
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.set_xlabel(r'share of the budget $\mathbb{E}_i |w_i^f|^2$')
    ax.text(1.02, -0.6, r'$\Phi^f$', ha='left', va='center', fontsize=FS_SMALL)
    ax.set_title(title, fontsize=FS_SMALL)
    return [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors] + [Line2D([], [], color=INK, lw=LW)], ['position', 'interaction', 'content', 'constant and real part', r'interval $\frac{1}{2}(\sigma_{\zeta,f}^2 \pm |s_\zeta^f|)$']


def budget_share(s: dict, models: list[str], model: str, out: Path) -> None:
    '''`cor:slow-planes-and-long-contexts` on measured planes. (a) The budget of six planes of one model split into its parts on the single needle at two lengths (budget_bars). (b) The share Phi^f of E_i |w_i^f|^2 that is position-dependent variance, (Var(p^f) + Var(eps^f)) / (|w~^f|^2 + sigma_w^2), averaged over prompts and non-sink heads, of every model on the five long tasks at the three lengths, against the turns of the plane over the context, (T - 1) theta_f / 2 pi, with the planes of the same model on the single needle joined across the lengths (grey); the interval (sigma_zeta^2 +- |s_zeta^f|) / 2 of the corollary (shaded), its short-context bound ((T - 1) theta_f)^2 / 12 (dashed), the long-context limit 1/2 (dotted) and the unreachable range above it (hatched).'''
    fig = figure(2.9)
    bars, c = fig.subplots(1, 2, width_ratios=[1, 1])
    e = s.get('planes_extra', {})
    if 'budget' in e:
        lengths = e['budget']['lengths']
        handles, labels = budget_bars(bars, e, f'(a) {model_label(e["budget"]["model"])}, needle at {lengths[0] // 1000}k and {lengths[-1] // 1000}k tokens')
        bars.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.4, -0.14), ncols=3, fontsize=FS_SMALL, frameon=False, handlelength=1.0, columnspacing=0.8)
    turns = np.logspace(-3, 3, 600)
    n = 4000
    z = np.exp(2j * np.pi * np.outer(turns / n, np.arange(1, n + 1)))
    zt = z.mean(1)
    sz, ps = 1 - abs(zt) ** 2, abs((z ** 2).mean(1) - zt ** 2)
    short = (2 * np.pi * turns) ** 2 / 12

    def point(cs: dict, case: str) -> tuple[np.ndarray, np.ndarray]:
        pl = cs[case]['plane']
        x = np.asarray(pl['period_over_context'], float)
        share = (np.asarray(pl['var_p'], float) + np.asarray(pl['var_e'], float)) / (np.asarray(pl['w2'], float) + np.asarray(pl['sigma_w2'], float))
        return 1 / x, share

    c.fill_between(turns, (sz - ps) / 2, (sz + ps) / 2, color=LIGHT, lw=0, zorder=0)
    c.plot(turns[short < 1.2], short[short < 1.2], color=GUIDE, lw=LW_THIN, ls='--', zorder=1)
    c.axhline(0.5, color=GUIDE, lw=LW_THIN, ls=':', zorder=1)
    c.fill_between(turns, np.maximum((sz + ps) / 2, 0.5), 1.0, fc=(*to_rgb(SOFT), 0.25), ec=GUIDE, lw=0, hatch='///', zorder=0)
    c.set_xscale('log'), c.set_xlim(1e-3, 1e3), c.set_ylim(-0.02, 1.0)
    c.set_ylabel(r'$\Phi^f$')
    c.set_xlabel(r'turns over the context, $(T - 1)\theta_f / 2\pi$')
    cs = s['cases'][model]
    tracks = [point(cs, k) for k in ('niah_4000', 'niah_16000', 'niah_31000') if k in cs]
    for f in range(tracks[0][0].size if tracks else 0):
        c.plot([tr[0][f] for tr in tracks], [tr[1][f] for tr in tracks], color=MID, lw=LW_HAIR, zorder=4)
    for m in models:
        for case in [k for k in s['cases'][m] if s['tasks'][k]['category'] == 'long']:
            x, y = point(s['cases'][m], case)
            c.plot(x, y, 'o', ms=MS_SMALL, mew=0, color=model_color(m), alpha=0.6, zorder=3)
    c.set_title('(b) every model and long task', fontsize=FS_SMALL)
    handles = [Line2D([], [], ls='', marker='o', ms=MS, mew=0, color=model_color(m)) for m in models]
    c.legend(handles, [model_label(m) for m in models], fontsize=FS_TINY, frameon=False, loc='upper left', ncols=2, handletextpad=0.2, columnspacing=0.6)
    save_figure(fig, out / 'budget_share.pdf')


def _phase_covariance(fig: plt.Figure, axs: list, s: dict, model: str, cases: list[str], labels: list[str]) -> None:
    '''`prop:phase-statistics` and `prop:head-as-a-sum-of-planes` with the real frequencies of one model, into the given axes: the modulus of the covariance of the phases of two planes, |E_j[zr_j^f conj(zr_j^g)]|, computed exactly from the model's theta_f at the mean context length of each case (T - 1 = length_mean - 2 keys, rounded), planes from fast to slow; one panel per case. The diagonal is sigma_zeta,f^2; dashed, the boundaries of the fast (sigma_zeta^2 > 0.9) and slow (sigma_zeta^2 < 0.1) planes.'''
    for k, (ax, c, lab) in enumerate(zip(axs, cases, labels)):
        cs = s['cases'][model][c]
        n = int(round(cs['length_mean'] - 2))
        G = np.abs(phase_covariance(np.asarray(s['planes_extra']['theta'][model], float), n))
        F = G.shape[0]
        sz = np.diag(G).real
        im = ax.pcolormesh(np.arange(F + 1) - 0.5, np.arange(F + 1) - 0.5, G, cmap=SEQ_CMAP, vmin=0, vmax=1, edgecolors='face', linewidth=MESH_LW, antialiased=False, rasterized=False)
        ax.set_xlim(-0.5, F - 0.5), ax.set_ylim(F - 0.5, -0.5), ax.set_aspect('equal'), ax.grid(False)
        ax.spines[:].set_visible(True)
        nf, ns = int((sz > FAST).sum()), int((sz >= SLOW).sum())
        for b in (nf, ns):
            ax.axvline(b - 0.5, color=BACKGROUND, lw=LW_HAIR, ls=(0, (3, 2)))
            ax.axhline(b - 0.5, color=BACKGROUND, lw=LW_HAIR, ls=(0, (3, 2)))
        for x0, x1, name in ((-0.5, nf - 0.5, 'fast'), (ns - 0.5, F - 0.5, 'slow')):
            if x1 - x0 >= 8:
                ax.text((x0 + x1) / 2, F - 2, name, fontsize=FS_TINY, color=BACKGROUND, ha='center', va='bottom')
        ax.set_xticks([0, F - 1], ['1', f'{F}']), ax.set_yticks([0, F - 1], ['1', f'{F}'] if k == 0 else [])
        ax.set_title(f'({"ab"[k]}) {lab}', fontsize=FS_SMALL)
        ax.set_xlabel('plane $g$')
        if k == 0:
            ax.set_ylabel('plane $f$')
    cb = fig.colorbar(im, ax=list(axs), shrink=0.8, aspect=20, pad=0.03)
    cb.solids.set_edgecolor('face')
    cb.set_label('$|\\mathbb{E}_j[\\tilde\\zeta_j^f\\, \\overline{\\tilde\\zeta_j^g}]|$', fontsize=FS_SMALL, labelpad=3), cb.ax.tick_params(labelsize=FS_SMALL)


def _aggregation(c: plt.Axes, s: dict, models: list[str]) -> None:
    '''`prop:head-as-a-sum-of-planes` on every non-sink head of every model at the last query, into the given axes: the distributions of kappa_0 = (Var(p) + Var(eps)) / d_h^-1 sum_f (Var(p^f) + Var(eps^f)) (left halves) and kappa_1 = Var(eps) / d_h^-1 sum_f Var(eps^f) (right halves) by context length: the nine short tasks, the three texts and the five long tasks at each length; median (dot) and quartiles (bar).'''
    pos, inter = PART_COLORS['positional'], PART_COLORS['interaction']
    ib = lambda cs: cs['interaction_bound']
    grid = np.linspace(-3, 3, 600)
    lengths = {g: s['heads_extra']['pooled'][g]['length'] for g in ('short', 'text')}
    for k, g in enumerate(GROUPS):
        sel = lambda cc, g=g: _group(s['tasks'][cc]) == g
        rp = np.log2(_pooled(s, models, sel, lambda cs, e: e['head_position'] / e['plane_position']))
        re = np.log2(_pooled(s, models, sel, lambda cs, e: ib(cs)['measured'] / ib(cs)['plane_sum']))
        for r, side, color in ((rp, -1, pos), (re, 1, inter)):
            if r.size < 2:
                continue
            lo_r, hi_r = np.quantile(r, [0.005, 0.99])
            gg = grid[(grid >= lo_r) & (grid <= hi_r)]
            if gg.size < 2:
                continue
            dens = 0.42 * _density(r, gg)
            c.fill_betweenx(gg, k, k + side * dens, color=color, alpha=0.35, lw=0, zorder=2)
            c.plot(k + side * dens, gg, color=color, lw=LW_THIN, zorder=3)
            q25, q50, q75 = np.quantile(r, [0.25, 0.5, 0.75])
            c.plot([k + side * 0.06] * 2, [q25, q75], color=color, lw=LW, solid_capstyle='butt', zorder=4)
            c.plot(k + side * 0.06, q50, 'o', ms=MS_SMALL, mfc=BACKGROUND, mec=color, mew=LW_THIN, zorder=5)
        c.plot([k, k], [grid[0], grid[-1]], color=LIGHT, lw=LW_HAIR, zorder=1)
    c.axhline(0, color=INK, lw=LW_THIN, ls='--', zorder=1)
    c.set_xticks(range(len(GROUPS)), [f'short\n{lengths["short"]:.0f}', f'texts\n{lengths["text"]:.0f}', '4000', '16000', '31000'], fontsize=FS_SMALL)
    c.tick_params(axis='x', length=0)
    c.set_xlim(-0.55, len(GROUPS) - 0.45), c.set_ylim(-1.45, 2.95)
    c.set_yticks(np.log2([0.5, 1, 2, 4]), ['1/2', '1', '2', '4'])
    c.set_ylabel('head / sum over planes')
    c.set_xlabel('mean context length (tokens)')
    c.legend([Patch(fc=pos, alpha=0.5, lw=0), Patch(fc=inter, alpha=0.5, lw=0)], [r'$\kappa_0$, position-dependent', r'$\kappa_1$, interaction'], fontsize=FS_SMALL, loc='upper right', ncols=1, handlelength=1.2)
    c.set_title('(c) heads against the sum of their planes', fontsize=FS_SMALL)


def aggregation_real(s: dict, model: str, cases: list[str], labels: list[str], models: list[str], out: Path) -> None:
    '''One figure: (a, b) the covariance of the phases of two planes of one model in two cases (`_phase_covariance`) and (c) kappa_0 and kappa_1 over the heads of every model (`_aggregation`).'''
    fig = figure(2.0)
    gs = fig.add_gridspec(1, 2, width_ratios=(1, 1))
    left = gs[0, 0].subgridspec(1, len(cases))
    _phase_covariance(fig, [fig.add_subplot(left[0, k]) for k in range(len(cases))], s, model, cases, labels)
    _aggregation(fig.add_subplot(gs[0, 1]), s, models)
    save_figure(fig, out / 'aggregation_real.pdf')


def interaction_real(s: dict, models: list[str], out: Path) -> None:
    '''The interaction of the non-sink heads, one row per model with the densities of the three kinds of input (short tasks, texts, long tasks) over the pooled (head, case) pairs, each with its median: (a) the interaction share b_h3 = Var(eps) / Var(A); (b) kappa_2 = Var(eps) / d_h^-1 sum_f sigma_w,f^2 sigma_zeta,f^2, with 1/2 for evenly spread phases and 1, the sum of the bounds.'''
    colors = dict(zip(KINDS, ordered(3)))
    fig = figure(2.5)
    axs = fig.subplots(1, 2, sharey=True)
    ib = lambda cs: cs['interaction_bound']
    panels = (
        (lambda cs, e: np.asarray(cs['heads']['behavior'], float)[..., 2], False, (0, 0.2), [0, 0.05, 0.1, 0.15, 0.2], {}, 'interaction share $b_{h,3}$', '(a) interaction share'),
        (lambda cs, e: np.asarray(ib(cs)['measured']) / np.asarray(ib(cs)['plane_bound_sum']), True, (0.3, 1.15), [0.35, 0.5, 0.7, 1], {0.5: 'even phases', 1: 'sum of bounds'}, '$\\kappa_2$ = head / sum of plane bounds', '(b) interaction over the sum of plane bounds'),
    )
    base = {m: -i for i, m in enumerate(models)}
    for ax, (value, log, (lo, hi), ticks, refs, xlabel, title) in zip(axs, panels):
        grid = np.linspace(np.log(lo), np.log(hi), 500) if log else np.linspace(lo, hi, 500)
        xs = np.exp(grid) if log else grid
        for m in models:
            y0 = base[m]
            ax.axhline(y0, color=SOFT, lw=LW_HAIR, zorder=1)
            for j, k in enumerate(KINDS):
                x = _pooled(s, [m], lambda cc, k=k: _kind(s['tasks'][cc]) == k, value)
                x = np.log(x[x > 0]) if log else x
                dens = 0.8 * _density(x, grid)
                z = 2 + 0.1 * j - 0.5 * base[m]
                ax.fill_between(xs, y0, y0 + dens, color=colors[k], alpha=0.18, lw=0, zorder=z)
                ax.plot(xs, y0 + dens, color=colors[k], lw=LW_THIN, zorder=z + 0.01)
                med = np.median(x)
                ax.plot([np.exp(med) if log else med] * 2, [y0, y0 + np.interp(med, grid, dens)], color=colors[k], lw=LW, solid_capstyle='butt', zorder=z + 0.02)
        for r, name in refs.items():
            ax.axvline(r, color=INK, lw=LW_THIN, ls='--', zorder=20)
            ax.text(r, 0.97, name, fontsize=FS_TINY, color=NEUTRAL, ha='center', va='bottom')
        if log:
            ax.set_xscale('log'), ax.minorticks_off()
        ax.set_xlim(lo, hi), ax.set_xticks(ticks, [f'{t:g}' for t in ticks])
        ax.set_xlabel(xlabel), ax.set_title(title, fontsize=FS_SMALL, pad=10)
        ax.spines['left'].set_visible(False), ax.tick_params(axis='y', length=0)
    axs[0].set_yticks([base[m] + 0.12 for m in models], [model_label(m) for m in models])
    axs[0].set_ylim(-len(models) + 0.85, 0.95)
    handles = [(Patch(fc=colors[k], alpha=0.22, lw=0), Line2D([], [], color=colors[k], lw=LW_THIN)) for k in KINDS] + [Line2D([], [], color=NEUTRAL, lw=LW)]
    fig.legend(handles, list(KINDS.values()) + ['median'], fontsize=FS_SMALL, loc='outside lower center', ncols=4, handlelength=1.4)
    save_figure(fig, out / 'interaction_real.pdf')


def simplex_main(s: dict, model: str, names: list[str], labels: list[str], out: Path) -> None:
    '''The behavior of the non-sink heads of one model on a few inputs, one simplex per input with its vertices named and the input as title, coloured by relative depth (a colorbar as tall as the triangles), with the line P_h = 1/2 (Var(c_i) = Var(p_j)).'''
    cs = s['cases'][model]
    fig = figure(2.0)
    axs = fig.subplots(1, len(names))
    for ax, c, lab in zip(axs, names, labels):
        b, keep = np.asarray(cs[c]['heads']['behavior'], float), np.asarray(cs[c]['non_sink'], bool)
        xy = b[keep] @ VERTICES
        ax.scatter(xy[:, 0], xy[:, 1], c=_depth(keep.shape)[keep], cmap=SEQ_CMAP, vmin=0, vmax=1, s=SCATTER_SMALL, lw=0)
        ax.plot(*np.vstack([VERTICES, VERTICES[:1]]).T, color=INK, lw=LW_THIN)
        ax.plot([0.5, 0.5], [0.0, np.sqrt(3) / 2], color=GUIDE, lw=LW_THIN, ls=':')
        for v, name, ha, dy in zip(VERTICES, ('symbolic', 'positional', 'interaction'), ('left', 'right', 'center'), (-0.07, -0.07, 0.04)):
            ax.text(v[0], v[1] + dy, name, ha=ha, va='top' if dy < 0 else 'bottom', fontsize=FS_SMALL)
        ax.set_xlim(-0.05, 1.05), ax.set_ylim(-0.14, 1.0), ax.set_aspect('equal'), ax.axis('off')
        ax.set_title(lab, pad=2)
    cax = axs[-1].inset_axes([1.1, 0.0, 0.05, np.sqrt(3) / 2], transform=axs[-1].transData)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=plt.Normalize(0, 1)), cax=cax)
    cb.set_label('relative depth', fontsize=FS_SMALL)
    cb.ax.tick_params(labelsize=FS_SMALL)
    cb.solids.set_edgecolor('face')
    save_figure(fig, out / 'simplex_main.pdf')


def stability(s: dict, models: list[str], model: str, out: Path) -> None:
    '''The stability of P_h over the heads non-sink in every task. (a) Split-half reliability of every task, one marker per model, tasks grouped into short, texts and long. (b) For one model, the Spearman correlation of P_h between every pair of tasks. (c) For one model, every head's P_h over the nine short tasks (range as a line, mean as a dot), heads sorted by their mean, with the class thresholds 0.2 and 0.8.'''
    tasks = s['tasks']
    fig = figure(2.2)
    a, b, c = fig.subplots(1, 3, width_ratios=(1, 1.1, 1))
    ag = s['agreement']
    names = ag['cases']
    kind = lambda n: 0 if tasks[n]['category'] in SHORT else (1 if tasks[n]['category'] == 'text' else 2)
    bounds = [sum(kind(n) < k for n in names) for k in (1, 2)]
    for m in models:
        cs = s['cases'][m]
        a.plot(range(len(names)), [cs[n]['split_half'] for n in names], 'o', ms=MS_SMALL, mew=0, color=model_color(m), label=model_label(m))
    for x in bounds:
        a.axvline(x - 0.5, color=GUIDE, lw=LW_HAIR)
    a.set_xticks([bounds[0] / 2 - 0.5, (bounds[0] + bounds[1]) / 2 - 0.5, (bounds[1] + len(names)) / 2 - 0.5], ['short', 'texts', 'long'])
    a.tick_params(axis='x', length=0)
    a.set_ylabel(r'$\rho_S(P(D^\prime), P(D^{\prime\prime}))$')
    a.set_title('(a) reliability of each task', fontsize=FS_SMALL)
    a.legend(fontsize=FS_TINY, frameon=False, ncols=1, loc='lower right', handletextpad=0.2, borderaxespad=0.0)
    cs = s['cases'][model]
    keep = np.logical_and.reduce([np.asarray(cs[n]['non_sink'], bool) for n in names])
    R = np.stack([np.asarray(cs[n]['heads']['ph'], float)[keep] for n in names])
    mesh = b.pcolormesh(np.arange(len(names) + 1), np.arange(len(names) + 1), np.asarray(ag['spearman'], float), cmap=SEQ_CMAP, vmin=0.5, vmax=1, edgecolors='face', linewidth=MESH_LW, rasterized=False)
    for x in bounds:
        b.axvline(x, color=BACKGROUND, lw=LW_THIN), b.axhline(x, color=BACKGROUND, lw=LW_THIN)
    ticks = [bounds[0] / 2, (bounds[0] + bounds[1]) / 2, (bounds[1] + len(names)) / 2]
    b.set_xticks(ticks, ['short', 'texts', 'long']), b.set_yticks(ticks, ['short', 'texts', 'long'])
    b.tick_params(length=0)
    b.set_xlim(0, len(names)), b.set_ylim(len(names), 0), b.set_aspect('equal')
    cb = fig.colorbar(mesh, ax=b, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=FS_SMALL)
    cb.set_label(r'$\rho_S(P(D), P(D^\prime))$', fontsize=FS_SMALL)
    cb.solids.set_edgecolor('face')
    b.set_title(f'(b) agreement, {model_label(model)}', fontsize=FS_SMALL)
    short = [i for i, n in enumerate(names) if kind(n) == 0]
    Rs = R[short]
    order = np.argsort(Rs.mean(0))
    x = np.arange(order.size)
    c.vlines(x, Rs.min(0)[order], Rs.max(0)[order], color=SOFT, lw=LW_HAIR)
    c.plot(x, Rs.mean(0)[order], '.', ms=MS_SMALL, color=INK)
    for y in (0.2, 0.8):
        c.axhline(y, color=GUIDE, lw=LW_THIN, ls='--')
    c.set_xlim(-1, order.size), c.set_ylim(0, 1)
    c.set_xlabel(r'heads of $\mathcal{H}^*$, by their mean $\overline{P}_h$')
    c.set_ylabel(r'$P_h(D)$, $D \in \mathcal{D}_c$')
    c.set_title(f'(c) short tasks, {model_label(model)}', fontsize=FS_SMALL)
    save_figure(fig, out / 'stability.pdf')


def long_contexts(s: dict, models: list[str], lengths: list[int], out: Path) -> None:
    '''Long contexts. (a) `cor:ratio-in-long-contexts` on measured planes (head means over non-sink heads) of the five long tasks: |P^f - limit| of every plane, with limit (|w~^f|^2 + sigma_w,f^2) / (|w~^f|^2 + 2 sigma_w,f^2), against its turns at each length, one colour per model, with the median over bins of turns at each length (line style). (b) The mean rise of P_h over the heads non-sink at every length, from the shortest to the longest length, against its prediction from the variance of the phases alone (length_prediction), one point per model and long task, with y = x.'''
    lim = s['planes_extra']['limit']
    fig = figure(2.3)
    a, b = fig.subplots(1, 2)
    edges = np.logspace(-3, 4, 22)
    for L, ls in zip(lengths, ('-', '--', ':')):
        rows = [(m, d) for m in models if m in lim for c, d in lim[m].items() if c.endswith(f'_{L}')]
        if not rows:
            continue
        for m, d in rows:
            a.scatter(d['turns'], abs(d['pf'] - d['limit']), s=SCATTER_SMALL, color=model_color(m), lw=0, alpha=0.3, zorder=2)
        turns, err = np.concatenate([d['turns'] for _, d in rows]), np.concatenate([abs(d['pf'] - d['limit']) for _, d in rows])
        which = np.digitize(turns, edges)
        mids = [(np.sqrt(edges[i - 1] * edges[i]), np.median(err[which == i])) for i in range(1, edges.size) if (which == i).sum() >= 10]
        if mids:
            a.plot(*np.array(mids).T, color=INK, lw=LW, ls=ls, zorder=3, label=f'median, {L // 1000}k tokens')
    a.set_xscale('log'), a.set_yscale('log'), a.set_xlim(1e-3, 1e4), a.set_ylim(1e-7, 1)
    a.axvline(1, color=GUIDE, lw=LW_THIN, ls='--', zorder=1)
    a.set_xlabel(r'turns, $(T - 1)\theta_f / 2\pi$'), a.set_ylabel(r'$|P^f - $limit$|$')
    a.set_title('(a) planes: distance to the limit', fontsize=FS_SMALL)
    a.legend(fontsize=FS_SMALL, frameon=False, loc='lower left')
    tasks = sorted({t for m in models for t in s['questions'][m]['length_prediction']})
    markers = dict(zip(tasks, 'osD^v'))
    for m in models:
        for task, v in s['questions'][m]['length_prediction'].items():
            b.plot(v['delta_hat'], v['delta'], markers[task], ms=MS, mew=0, color=model_color(m), zorder=3)
    b.plot([0, 0.22], [0, 0.22], color=INK, lw=LW_THIN, ls='--', zorder=2)
    b.set_xlim(0, 0.22), b.set_ylim(0, 0.22), b.set_aspect('equal')
    b.set_xlabel(r'predicted, mean $\hat\Delta_h$'), b.set_ylabel(r'measured, mean $\Delta_h$')
    b.set_title(f'(b) heads: rise of $P_h$, {min(lengths) // 1000}k to {max(lengths) // 1000}k tokens', fontsize=FS_SMALL)
    names = {'niah': 'needle', 'niah_multikey': 'multi-key', 'niah_multivalue': 'multi-value', 'variable_tracking': 'variables', 'key_value_long': 'key-value'}
    handles = [Line2D([], [], ls='', marker='o', ms=MS, mew=0, color=model_color(m)) for m in models] + [Line2D([], [], ls='', marker=markers[t], ms=MS, mew=0, color=NEUTRAL) for t in tasks]
    b.legend(handles, [model_label(m) for m in models] + [names.get(t, t) for t in tasks], fontsize=FS_TINY, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5), handletextpad=0.2)
    save_figure(fig, out / 'long_contexts.pdf')


def fast_planes(s: dict, models: list[str], out: Path) -> None:
    '''On WikiText-2, by role (previous-token, induction, other non-sink heads; roles from the repeated tokens), the mean over a group's heads of the cumulative share of each head's (a) budget |w~^f|^2 + sigma_w,f^2 and (b) position-dependent variance, over the planes from the fastest to the slowest, against the plane's period over the context; one line per model and role.'''
    fig = figure(2.1)
    axes = fig.subplots(1, 2, sharey=True)
    colors = {'previous_token': PART_COLORS['positional'], 'induction': PART_COLORS['content'], 'other': GUIDE}
    labels = {'previous_token': 'previous-token heads', 'induction': 'induction heads', 'other': 'other heads'}
    for m in models:
        wiki, period = s['roles'].get(m, {}).get('wiki'), s['cases'].get(m, {}).get('wiki', {}).get('plane', {}).get('period_over_context')
        if not wiki or period is None or 'norm_profile' not in wiki.get('other', {}):
            continue
        order = np.argsort(period)
        for ax, key in zip(axes, ('norm_profile', 'position_profile')):
            for role, color in colors.items():
                prof = np.asarray(wiki[role].get(key, []), float)
                if prof.size and np.isfinite(prof).all():
                    ax.plot(np.asarray(period)[order], np.cumsum(prof[order]), color=color, lw=LW_THIN, alpha=0.8)
    for ax, title in zip(axes, ('(a) budget $|\\tilde w^f|^2 + \\sigma_{w,f}^2$', '(b) position-dependent variance')):
        ax.axvline(1, color=GUIDE, lw=LW_HAIR, ls='--')
        ax.set_xscale('log'), ax.set_ylim(0, 1), ax.set_xlabel('period of the plane / context length'), ax.set_title(title)
    axes[0].set_ylabel('cumulative share, fast to slow')
    handles = [Line2D([], [], color=c, lw=LW, label=labels[r]) for r, c in colors.items()]
    fig.legend(handles=handles, fontsize=FS_SMALL, ncol=3, loc='outside upper center', frameon=False)
    save_figure(fig, out / 'fast_planes.pdf')


def measure(s: dict, model: str, names: list[str], out: Path) -> None:
    '''Urrutia et al. in our convention (first token excluded), per non-sink head of one model on a few cases, one colour per case: (a) their normalized measure Dsym / (Dpos + Dsym) against P_h, with y = x; (b) the density of the variance V of the observed logits over their bound Dpos + Dsym, with 1/2.'''
    fig = figure(2.2)
    a, b = fig.subplots(1, 2)
    grid = np.linspace(0, 1, 400)
    for c, color in zip(names, ordered(len(names))):
        d = s['urrutia'][model][c]
        keep = np.asarray(d['non_sink'], bool)
        a.scatter(np.asarray(d['ph'])[keep], np.asarray(d['measure'])[keep], s=SCATTER_SMALL, color=color, lw=0, alpha=0.6, zorder=3, label=CASE_LABELS.get(c, c))
        b.plot(grid, _density(np.asarray(d['bound_theirs'], float)[keep], grid), color=color, lw=LW, zorder=3)
    a.plot([0, 1], [0, 1], color=INK, lw=LW_THIN, ls='--', zorder=2)
    a.set_xlim(0, 1), a.set_ylim(0, 1), a.set_aspect('equal'), a.set_xlabel('$P_h$'), a.set_ylabel(r'$D_{sym} / (D_{pos} + D_{sym})$')
    a.set_title('(a) normalized measure', fontsize=FS_SMALL)
    a.legend(fontsize=FS_SMALL, frameon=False, loc='upper left', markerscale=3, handletextpad=0.2)
    b.axvline(0.5, color=INK, lw=LW_THIN, ls='--', zorder=2)
    b.set_xlim(0, 1), b.set_ylim(0, 1.05), b.set_yticks([]), b.spines['left'].set_visible(False)
    b.set_xlabel(r'$V / (D_{pos} + D_{sym})$'), b.set_title('(b) observed variance over the bound', fontsize=FS_SMALL)
    save_figure(fig, out / 'measure.pdf')


def chen_angle(s: dict, models: list[str], out: Path) -> None:
    '''Chen and Yan's angle on WikiText-2, by deciles, one line per model: (a) iota = |s_w^f| / sigma_w,f^2 (median and quartiles) against the absolute cosine between the two key weight rows of each (non-sink head, plane); (b) the per-head mean absolute cosine (query and key sides) against the positional ratio.'''
    fig = figure(2.1)
    a, b = fig.subplots(1, 2)
    for m in models:
        color = model_color(m)
        c = s['chen'].get(m)
        if not c or 'cos_iota' not in c:
            continue
        for ax, key in ((a, 'cos_iota'), (b, 'ratio_cos')):
            d = {k: np.asarray(v, float) for k, v in c[key].items()}
            ax.plot(d['x'], d['median'], color=color, lw=LW_THIN, marker='o', ms=MS_SMALL, label=model_label(m))
            if key == 'cos_iota':
                ax.fill_between(d['x'], d['q25'], d['q75'], color=color, alpha=0.07, lw=0)
    a.set_xscale('log'), a.set_ylim(0, 1), a.set_xlabel(r'$|\cos|$ of the key weight rows of a plane'), a.set_ylabel(r'$\iota_{hf} = |s_w^f| / \sigma_{w,f}^2$'), a.set_title('(a) weight angle vs keys on a line')
    b.set_xlim(0, 1), b.set_xlabel('positional ratio $P_h$'), b.set_ylabel(r'mean $|\cos|$ of the head'), b.set_title('(b) weight angle vs behavior')
    fig.legend(*a.get_legend_handles_labels(), fontsize=FS_SMALL, ncol=4, loc='outside upper center', frameon=False)
    save_figure(fig, out / 'chen_angle.pdf')


def draw(cfg: dict, s: dict, out: Path) -> None:
    '''Every figure of 01 that the document uses, in their order.'''
    out.mkdir(parents=True, exist_ok=True)
    f, e = cfg['figures'], s.get('planes_extra', {})
    models = [m for m in cfg['models'] if m in s['cases']]
    main = f['main_model']
    plane_models = [m for m in models if m in e.get('zones', {})]
    head_models = [m for m in models if m in s.get('heads_extra', {}).get('cases', {})]
    if 'bounds' in e:
        plane_bounds_real(e, out)
    if plane_models:
        phase_moments_real(e, plane_models, out)
        plane_fractions_ci(e, plane_models, out)
    if main in s['cases']:
        budget_share(s, models, main, out)
        cases = [(c, lab) for c, lab in zip(f['interaction_cases'], f['interaction_labels']) if c in s['cases'][main]]
    if head_models and main in s['cases']:
        aggregation_real(s, main, [c for c, _ in cases], [lab for _, lab in cases], head_models, out)
    if head_models:
        interaction_real(s, head_models, out)
    if main in s['cases']:
        pairs = [(c, lab) for c, lab in zip(f['main_cases'], f['main_labels']) if c in s['cases'][main]]
        simplex_main(s, main, [c for c, _ in pairs], [lab for _, lab in pairs], out)
        stability(s, models, main, out)
    if plane_models:
        long_contexts(s, models, sorted(cfg['lengths']), out)
    if models:
        fast_planes(s, models, out)
    if main in s['cases']:
        measure(s, main, [c for c in f['measure_cases'] if c in s['urrutia'][main]], out)
    if models:
        chen_angle(s, models, out)
