'''The synthetic figures of the document (docs/notes.typ): python docs/figures/figures.py draws them into this folder.'''
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.plots import ARROW_SCALE, BACKGROUND, FS, FS_SMALL, FS_TINY, GUIDE, INK, LIGHT, LW, LW_BOLD, LW_HAIR, LW_THIN, MS, MS_LARGE, MS_SMALL, NEUTRAL, OPERATOR, OURS, PART_COLORS, SOFT, TEXT_WIDTH, arrow, diagonal, figure, matrix, phasors, save_figure

HERE = Path(__file__).resolve().parent


W, H = 5.0, 1.62


def axes(fig: plt.Figure, x: float, y: float, size: float) -> plt.Axes:
    return fig.add_axes((x / W, y / H, size / W, size / H))


def overview() -> None:
    rng = np.random.default_rng(3)
    n = 10
    c0 = rng.normal(0, 1.0, n)
    c = (c0 - c0.mean())[:, None] * np.ones((1, n))
    p = np.cos(2 * np.pi * np.arange(n) / 5)[None, :] * np.ones((n, 1)) * 1.2
    e = rng.normal(0, 0.45, (n, n))
    e = e - e.mean(0, keepdims=True) - e.mean(1, keepdims=True) + e.mean()
    A = c + p + e
    fig = plt.figure(figsize=(W, H))
    big, small, y0 = 1.18, 0.82, 0.2
    ax = axes(fig, 0.2, y0, big)
    matrix(ax, A, 3)
    diagonal(ax, n)
    ax.set_xlabel('position $j$')
    ax.set_ylabel('content $i$')
    ax.set_title('(a) matrix $A_{ij}$')
    mid = y0 + big / 2
    fig.text(1.62 / W, mid / H, '$=$', fontsize=OPERATOR, ha='center', va='center')
    x0, step = 1.85, small + 0.27
    for k, (M, name) in enumerate(((c, 'content $c_i$'), (p, 'position $p_j$'), (e, r'interaction $\epsilon_{ij}$'))):
        a = axes(fig, x0 + k * step, mid - small / 2, small)
        matrix(a, M, 3)
        a.set_title(name)
        if k < 2:
            fig.text((x0 + k * step + small + 0.135) / W, mid / H, '$+$', fontsize=OPERATOR, ha='center', va='center')
    fig.text((x0 + (3 * small + 2 * 0.27) / 2) / W, (y0 + big + 0.12) / H, '(b) its three parts', ha='center', va='bottom')
    save_figure(fig, HERE / 'overview.pdf')


def three_parts(n: int = 12) -> None:
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    cases = (('positional:\nconstant columns', np.cos(2 * np.pi * j / n) + 0.5 * np.sin(4 * np.pi * j / n)), ('symbolic:\nconstant rows', np.cos(2 * np.pi * i / n) - 0.5 * np.sin(6 * np.pi * i / n)), ('pure interaction:\n' + r'$A_{ij} = \cos(2\pi(i-j)/(T-1))$', np.cos(2 * np.pi * (i - j) / n)))
    fig = figure(1.48, 0.67)
    for ax, (title, A) in zip(fig.subplots(1, 3), cases):
        matrix(ax, A, 1.5)
        diagonal(ax, n)
        ax.set_title(title)
        ax.set_xlabel(f'position $j$; $\\mathbb{{V}}(\\lambda_t) = {np.diag(A).var():.2f}$')
        ax.set_ylabel('content $i$')
    save_figure(fig, HERE / 'three_parts.pdf')


def statistic_panel(ax: plt.Axes, x: np.ndarray, color: str, lim: float, circle: bool, title: str, names: tuple[str, str, str]) -> None:
    '''Points x, their center of mass (arrow from the origin), their deviations (segments from the center), the circle of radius sigma around the center (variance) and the covariance ellipse of the real and imaginary parts of the deviations, of semi-axes sqrt((sigma^2 +- |s|) / 2) with the major one at angle arg(s) / 2 (pseudo-variance), with the values of the three statistics.'''
    phasors(ax, lim, circle)
    m = x.mean()
    d = x - m
    var, pseudo = np.mean(abs(d) ** 2), np.mean(d ** 2)
    for p in x:
        ax.plot([m.real, p.real], [m.imag, p.imag], color=LIGHT, lw=LW_HAIR, zorder=1)
    ax.plot(x.real, x.imag, 'o', ms=MS_SMALL, mew=0, color=color, zorder=2)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(m.real + np.sqrt(var) * np.cos(t), m.imag + np.sqrt(var) * np.sin(t), color=GUIDE, lw=LW_THIN, ls='--', zorder=3)
    a, b, phi = np.sqrt((var + abs(pseudo)) / 2), np.sqrt(max(var - abs(pseudo), 0) / 2), np.angle(pseudo) / 2
    e = m + (a * np.cos(t) + 1j * b * np.sin(t)) * np.exp(1j * phi)
    ax.plot(e.real, e.imag, color=OURS, lw=LW, zorder=4)
    ax.annotate('', xy=(m.real, m.imag), xytext=(0, 0), arrowprops={'arrowstyle': '-|>', 'color': INK, 'lw': LW, 'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': ARROW_SCALE}, zorder=5)
    ax.set_title(title, fontsize=FS)
    ax.text(0.5, -0.04, f'${names[0]} = {abs(m) ** 2:.2f}$, ${names[1]} = {var:.2f}$, ${names[2]} = {abs(pseudo):.2f}$', transform=ax.transAxes, ha='center', va='top', fontsize=FS_SMALL)


def statistics(n: int = 30, N: int = 32) -> None:
    '''`tab:statistics` with one graphic language: points, center of mass (arrow), deviations (grey segments), circle of radius sigma (dashed) and covariance ellipse (blue, its shape set by the pseudo-variance). Top: content factors of equal variance, on a line, elongated and isotropic. Bottom: the phases of a slow, an intermediate and a fast plane over a context of N positions.'''
    rng = np.random.default_rng(5)
    unit = lambda z: (z - z.mean()) / np.sqrt(np.mean(abs(z - z.mean()) ** 2))
    g = rng.normal(0, 1, n // 2) + 1j * rng.normal(0, 1, n // 2)
    g = unit(np.concatenate([g - g.mean(), 1j * (g - g.mean())]))
    line = unit(rng.normal(0, 1, n) + 0j)
    center = 0.8 * np.exp(0.6j)
    elongated = unit(g.real + 0.45j * g.imag)
    content = ((center + 0.42 * line * np.exp(1.0j), r'(a) on a line: $|s_w^f| = \sigma_{w,f}^2$'), (center + 0.42 * elongated * np.exp(1.0j), '(b) elongated'), (center + 0.42 * g, '(c) isotropic: $s_w^f = 0$'))
    j = np.arange(1, N + 1)
    phases = ((0.12, '(d) slow plane'), (0.6, '(e) intermediate plane'), (2.3, '(f) fast plane'))
    fig = figure(4.0, 0.74)
    axs = fig.subplots(2, 3)
    for ax, (x, title) in zip(axs[0], content):
        statistic_panel(ax, x, PART_COLORS['content'], 1.75, False, title, (r'|\tilde w^f|^2', r'\sigma_{w,f}^2', '|s_w^f|'))
        ax.set_xlim(-0.3, 1.6), ax.set_ylim(-0.3, 1.6)
    for ax, (turns, title) in zip(axs[1], phases):
        statistic_panel(ax, np.exp(2j * np.pi * turns * j / N), PART_COLORS['positional'], 1.25, True, f'{title}, {turns} turns', (r'|\tilde\zeta^f|^2', r'\sigma_{\zeta,f}^2', r'|s_\zeta^f|'))
    save_figure(fig, HERE / 'statistics.pdf')


def phasor_panel(ax: plt.Axes, points: np.ndarray, color: str, centered: bool, lim: float, title: str, circle: bool) -> None:
    phasors(ax, lim, circle)
    mean = points.mean()
    if centered:
        ax.plot(points.real, points.imag, 'o', ms=MS, mew=0, color=GUIDE, alpha=0.5)
        ax.plot((points - mean).real, (points - mean).imag, 'o', ms=MS, mew=0, color=color)
        arrow(ax, 0, mean, dashed=True)
    else:
        ax.plot(points.real, points.imag, 'o', ms=MS, mew=0, color=color)
        arrow(ax, mean)
    ax.set_title(title)


def plane_parts(n: int = 12, theta: float = 0.2) -> None:
    rng = np.random.default_rng(5)
    keys = 1.3 * np.exp(-0.5j) + 0.55 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    w = np.conj(keys)
    w_dev = w - w.mean()
    zeta = np.exp(1j * (n - np.arange(n)) * theta)
    zeta_dev = zeta - zeta.mean()
    cells = {(0, 0): (np.real(w.mean() * zeta.mean()) * np.ones((n, n)), 'constant\n' + r'$\mu = \mathrm{Re}[\tilde w\,\tilde\zeta]$'), (0, 1): (np.real(w.mean() * zeta_dev)[None, :] * np.ones((n, 1)), 'positional\n' + r'$p_j = \mathrm{Re}[\tilde w\,\mathring\zeta_j]$'), (1, 0): (np.real(w_dev * zeta.mean())[:, None] * np.ones((1, n)), 'content\n' + r'$c_i = \mathrm{Re}[\mathring w_i\,\tilde\zeta]$'), (1, 1): (np.real(w_dev[:, None] * zeta_dev[None, :]), 'interaction\n' + r'$\epsilon_{ij} = \mathrm{Re}[\mathring w_i\,\mathring\zeta_j]$')}
    A = np.real(w[:, None] * zeta[None, :])
    assert np.allclose(sum(M for M, _ in cells.values()), A)
    vmax = np.abs(A).max()
    fig = figure(4.53, 0.72)
    grid = fig.subplots(3, 3)
    matrix(grid[0, 0], A, vmax)
    grid[0, 0].set_title(r'$A^f_{ij} = \mathrm{Re}[w_i\,\zeta_j]$')
    phasor_panel(grid[0, 1], zeta, PART_COLORS['positional'], False, 1.3, 'common position\n' + r'center of mass $\tilde\zeta$', True)
    phasor_panel(grid[0, 2], zeta, PART_COLORS['positional'], True, 1.3, 'specific position\n' + r'$\mathring\zeta_j = \zeta_j - \tilde\zeta$', True)
    lim = np.abs(w).max() * 1.1
    phasor_panel(grid[1, 0], w, PART_COLORS['content'], False, lim, 'common content\n' + r'center of mass $\tilde w$', False)
    phasor_panel(grid[2, 0], w, PART_COLORS['content'], True, lim, 'specific content\n' + r'$\mathring w_i = w_i - \tilde w$', False)
    for (r, c), (M, title) in cells.items():
        matrix(grid[r + 1, c + 1], M, vmax)
        grid[r + 1, c + 1].set_title(title)
    save_figure(fig, HERE / 'plane_parts.pdf')


def draw_distances(ax: plt.Axes) -> None:
    '''`prop:the-positional-ratio-as-a-relative-distance`, schematically: modulo constants, the content, positional and interaction parts are orthogonal axes, the symbolic and positional matrices span the additive plane, and A's nearest positional, symbolic and additive matrices are its orthogonal projections, at squared distances that are sums of variances of the parts.'''
    c0, p0, e0 = 1.5, 1.0, 1.3
    pr = lambda c, p, e: np.array([c + 0.55 * p, e + 0.35 * p])
    arrowstyle = lambda color: {'arrowstyle': '-|>', 'color': color, 'lw': LW, 'mutation_scale': ARROW_SCALE, 'shrinkA': 0, 'shrinkB': 0}
    plane = np.array([pr(-0.1, -0.1, 0), pr(2.1, -0.1, 0), pr(2.1, 1.55, 0), pr(-0.1, 1.55, 0), pr(-0.1, -0.1, 0)])
    ax.fill(plane[:, 0], plane[:, 1], color=LIGHT, alpha=0.45, lw=0)
    ax.plot(*plane.T, color=GUIDE, lw=LW_HAIR)
    ax.text(*(pr(2.1, 1.55, 0) + [0.06, 0]), 'additive matrices', fontsize=FS_SMALL, ha='left', va='center', color=NEUTRAL)
    ax.annotate('', xy=pr(2.4, 0, 0), xytext=pr(0, 0, 0), arrowprops=arrowstyle(PART_COLORS['content']))
    ax.text(*(pr(2.4, 0, 0) + [0.05, 0]), r'symbolic $\mathcal{S}$', color=PART_COLORS['content'], fontsize=FS_SMALL, ha='left', va='center')
    ax.annotate('', xy=pr(0, 1.85, 0), xytext=pr(0, 0, 0), arrowprops=arrowstyle(PART_COLORS['positional']))
    ax.text(*(pr(0, 1.95, 0) + [0.04, 0.05]), r'positional $\mathcal{P}$', color=PART_COLORS['positional'], fontsize=FS_SMALL, ha='left', va='bottom')
    ax.annotate('', xy=pr(0, 0, 1.45), xytext=pr(0, 0, 0), arrowprops=arrowstyle(PART_COLORS['interaction']))
    ax.text(*(pr(0, 0, 1.45) + [0, 0.04]), 'interaction', color=PART_COLORS['interaction'], fontsize=FS_SMALL, ha='center', va='bottom')
    A, add, cS, pP = pr(c0, p0, e0), pr(c0, p0, 0), pr(c0, 0, 0), pr(0, p0, 0)
    ax.plot(*np.array([cS, add, pP]).T, color=GUIDE, lw=LW_HAIR, ls=':')
    ax.plot(*np.array([A, add]).T, color=PART_COLORS['interaction'], lw=LW, ls='--')
    ax.plot(*np.array([A, cS]).T, color=PART_COLORS['content'], lw=LW, ls='--')
    ax.plot(*np.array([A, pP]).T, color=PART_COLORS['positional'], lw=LW, ls='--')
    ax.text(A[0] + 0.06, (A[1] + add[1]) / 2 - 0.25, r'$\mathbb{V}(\epsilon_{ij})$', fontsize=FS_SMALL, color=PART_COLORS['interaction'], ha='left', va='center')
    ax.text(A[0] + 0.06, A[1] - 0.32, r'$d(A,\mathcal{S})^2 = \mathbb{V}(p_j) + \mathbb{V}(\epsilon_{ij})$', fontsize=FS_SMALL, color=PART_COLORS['content'], ha='left', va='center')
    m = (A + pP) / 2
    ax.text(m[0] - 0.08, m[1] + 0.1, r'$d(A,\mathcal{P})^2 = \mathbb{V}(c_i) + \mathbb{V}(\epsilon_{ij})$', fontsize=FS_SMALL, color=PART_COLORS['positional'], ha='right', va='bottom')
    for pt, name, off, ha, va in ((A, '$A$', (0.05, 0.03), 'left', 'bottom'), (add, r'$\mu + c_i + p_j$', (0.06, -0.02), 'left', 'top'), (cS, r'$\mu + c_i$', (0.0, -0.08), 'center', 'top'), (pP, r'$\mu + p_j$', (0.02, -0.07), 'left', 'top')):
        ax.plot(*pt, 'o', color=INK, ms=MS)
        ax.text(pt[0] + off[0], pt[1] + off[1], name, fontsize=FS_SMALL, ha=ha, va=va)
    ax.set_aspect('equal'), ax.axis('off')
    ax.set_xlim(-1.0, 4.1), ax.set_ylim(-0.25, 1.75)


def distances() -> None:
    fig = figure(2.53, 0.65)
    draw_distances(fig.add_subplot())
    save_figure(fig, HERE / 'distances.pdf')


def gap(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float], label: tuple[float, float], offset: tuple[float, float], align: tuple[str, str]) -> None:
    '''Double-headed arrow from a to b in the interaction colour, labelled epsilon_ij at label shifted by offset points.'''
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '<->', 'color': PART_COLORS['interaction'], 'lw': LW, 'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': ARROW_SCALE})
    ax.annotate(r'$\epsilon_{ij}$', xy=label, xytext=offset, textcoords='offset points', color=PART_COLORS['interaction'], fontsize=FS, ha=align[0], va=align[1])


def interaction_example(n: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Synthetic head whose contents prefer shifted copies of one positional profile: content offsets, amplitudes, shifts and positions.'''
    rng = np.random.default_rng(4)
    return rng.normal(0, 0.5, n), rng.uniform(0.8, 1.2, n), rng.uniform(-1.6, 1.6, n), np.arange(n)


def interaction_matrices(n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    '''The additive and the interacting matrix of the synthetic head.'''
    c, amp, shift, j = interaction_example(n)
    return c[:, None] + np.cos(2 * np.pi * j / n)[None, :], c[:, None] + amp[:, None] * np.cos(2 * np.pi * (j[None, :] - shift[:, None]) / n)


def interaction_profiles(n: int = 6) -> None:
    '''`prop:readings-of-the-interaction`: (a, b) an additive matrix and one with interaction, each with its positional profiles p^(i)_j (centered rows, above, aligned with the columns) and its content profiles c^(j)_i (centered columns, to the right, aligned with the rows), p and c their means, and epsilon_ij the deviation of either profile from its mean at the marked cell; (c) the four-cell contrast xi = (A_ij - A_ij') - (A_i'j - A_i'j') in the matrix with interaction (it vanishes in the additive one).'''
    additive, with_e = interaction_matrices(n)
    e = np.abs(with_e - with_e.mean(1, keepdims=True) - with_e.mean(0) + with_e.mean())[1:-1, 1:-1]
    i0, k = np.array(np.unravel_index(np.argmax(e), e.shape)) + 1
    x = np.arange(n) + 0.5
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.79))
    gs = fig.add_gridspec(2, 8, width_ratios=(1, 0.5, 0.22, 1, 0.5, 0.3, 1, 0.05), height_ratios=(0.5, 1), left=0.05, right=0.99, top=0.92, bottom=0.25, wspace=0.05, hspace=0.05)
    for col, (title, A) in enumerate((('additive', additive), ('with interaction', with_e))):
        rows, cols = A - A.mean(1, keepdims=True), A - A.mean(0)
        mat = fig.add_subplot(gs[1, 3 * col])
        top = fig.add_subplot(gs[0, 3 * col], sharex=mat)
        side = fig.add_subplot(gs[1, 3 * col + 1], sharey=mat)
        matrix(mat, A, 2.5)
        mat.set_aspect('auto')
        mat.add_patch(plt.Rectangle((0, i0), n, 1, fill=False, ec=INK, lw=LW_THIN))
        mat.add_patch(plt.Rectangle((k, 0), 1, n, fill=False, ec=INK, lw=LW_THIN))
        mat.add_patch(plt.Rectangle((k, i0), 1, 1, fill=False, ec=PART_COLORS['interaction'], lw=LW_BOLD))
        mat.set_xticks([k + 0.5], ['$j$']), mat.set_yticks([i0 + 0.5], ['$i$'])
        mat.set_xlabel('position'), mat.set_ylabel('content')
        for line in rows:
            top.plot(x, line, color=SOFT, lw=LW_THIN)
        top.plot(x, rows[i0], color=INK, lw=LW_THIN, marker='o', ms=MS_SMALL)
        top.plot(x, rows.mean(0), color=PART_COLORS['positional'], lw=LW_BOLD)
        for line in cols.T:
            side.plot(line, x, color=SOFT, lw=LW_THIN)
        side.plot(cols[:, k], x, color=INK, lw=LW_THIN, marker='o', ms=MS_SMALL)
        side.plot(cols.mean(1), x, color=PART_COLORS['content'], lw=LW_BOLD)
        if col == 1:
            lo, hi = sorted((rows.mean(0)[k], rows[i0, k]))
            gap(top, (k + 0.5, rows.mean(0)[k]), (k + 0.5, rows[i0, k]), (k + 0.5, lo), (0, -4), ('center', 'top'))
            gap(side, (cols.mean(1)[i0], i0 + 0.5), (cols[i0, k], i0 + 0.5), (min(cols.mean(1)[i0], cols[i0, k]), i0 + 0.5), (-4, 0), ('right', 'center'))
        top.set_ylim(-1.7, 1.7), side.set_xlim(-1.7, 1.7)
        top.axhline(0, color=LIGHT, lw=LW_HAIR, zorder=0), side.axvline(0, color=LIGHT, lw=LW_HAIR, zorder=0)
        for ax in (top, side):
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.spines[:].set_visible(False)
        top.set_title(f'({"ab"[col]}) {title}')
    pairs = [(r, r2, q, q2) for r in range(1, n - 1) for r2 in range(r + 2, n - 1) for q in range(1, n - 1) for q2 in range(q + 2, n - 1)]
    r, r2, q, q2 = max(pairs, key=lambda c: abs(with_e[c[0], c[2]] - with_e[c[0], c[3]] - with_e[c[1], c[2]] + with_e[c[1], c[3]]))
    ax = fig.add_subplot(gs[1, 6])
    matrix(ax, with_e, 2.5)
    ax.set_aspect('auto')
    for (y, xx), sgn in zip(((r, q), (r, q2), (r2, q), (r2, q2)), ('$+$', '$-$', '$-$', '$+$')):
        ax.add_patch(plt.Rectangle((xx, y), 1, 1, fill=False, ec=INK, lw=LW))
        ax.text(xx + 0.5, y + 0.5, sgn, ha='center', va='center', fontsize=FS, color=BACKGROUND if abs(with_e[y, xx]) > 1.6 else INK)
    xi = with_e[r, q] - with_e[r, q2] - with_e[r2, q] + with_e[r2, q2]
    ax.set_title(f'(c) contrast, $\\xi = {xi:.2f}$'.replace('-', '−'))
    ax.set_xticks([q + 0.5, q2 + 0.5], ['$j$', "$j'$"]), ax.set_yticks([r + 0.5, r2 + 0.5], ['$i$', "$i'$"])
    ax.set_xlabel('position'), ax.set_ylabel('content')
    handles = [plt.Line2D([], [], color=SOFT, lw=LW_THIN), plt.Line2D([], [], color=INK, lw=LW_THIN, marker='o', ms=MS_SMALL), plt.Line2D([], [], color=PART_COLORS['positional'], lw=LW_BOLD), plt.Line2D([], [], color=PART_COLORS['content'], lw=LW_BOLD)]
    fig.legend(handles, ['centered rows (above), columns (right)', 'row $i$, column $j$', 'mean row $p_j$', 'mean column $c_i$'], loc='lower center', bbox_to_anchor=(0.36, 0.0), ncols=2, columnspacing=1.0, handlelength=1.6, fontsize=FS_SMALL, frameon=False)
    save_figure(fig, HERE / 'interaction_profiles.pdf')


def single_plane_heads(N: int = 10, n: int = 3) -> None:
    '''`prop:three-pure-heads` with r = 1: for each single-plane head, its query and the rotated keys k(x_j) e^{-i(T-j) theta} of a context of N positions (labelled by j; the logit is the real part of q times the conjugate of the rotated key, the cosine of their angle), its counterfactual matrix and the observed logits along the diagonal, the largest ringed.'''
    T, j = N + 1, np.arange(1, N + 1)
    phi = 2 * np.pi * np.array([2, 0, 4, 1, 3, 2, 4, 0, 1, 3]) / 5
    labels = np.array([4, 6, 1, 5, 9, 3, 7, 10, 8, 2])
    assert sorted(labels) == list(j) and (labels == j).sum() == 1
    theta_p, theta_e, star = 0.75 * 2 * np.pi / (N - 1), 2 * np.pi / N, 5
    assert N == 10, 'the titles state theta = pi/6'
    cases = (
        ('symbolic', '(a) symbolic: retrieval\n' + r'$\theta = 0$, $k = e^{-i\phi(x)}$, $q = e^{-i\phi(x_*)}$', np.exp(-1j * phi[star]), np.exp(-1j * phi), 0.0, 'constant rows: $(1, 0, 0)$'),
        ('positional', f'(b) positional: {n} places back\n' + rf'$\theta = \pi/6$, $k = 1$, $q = e^{{-i{n}\theta}}$', np.exp(-1j * n * theta_p), np.ones(N, complex), theta_p, 'constant columns: $(0, 1, 0)$'),
        ('interaction', '(c) interaction: fixed point\n' + r'$\theta = 2\pi/(T - 1)$, $k = e^{-i\ell(x)\theta}$, $q = e^{-i\theta}$', np.exp(-1j * theta_e), np.exp(-1j * labels * theta_e), theta_e, '$c = p = 0$: $(0, 0, 1)$'),
    )
    fig = figure(4.75, 0.92)
    axs = fig.subplots(3, 3, gridspec_kw={'height_ratios': (1.15, 1, 0.62)})
    for col, (part, title, q, k, theta, behavior) in enumerate(cases):
        color = PART_COLORS['content' if part == 'symbolic' else part]
        rotated = k * np.exp(-1j * (T - j) * theta)
        A = np.real(q * np.conj(k)[:, None] * np.exp(1j * (T - j)[None, :] * theta))
        logits = np.diag(A)
        best = np.isclose(logits, logits.max())
        ax = axs[0, col]
        phasors(ax, 1.42, True)
        groups: dict[int, list[int]] = {}
        for t, z in zip(j, rotated):
            groups.setdefault(round(np.angle(z) * 1e4) % round(2e4 * np.pi), []).append(t)
        for ts in groups.values():
            z = rotated[ts[0] - 1]
            ax.plot(z.real, z.imag, 'o', ms=MS, mew=0, color=color, zorder=3)
            ax.text(1.24 * z.real, 1.24 * z.imag, ','.join(map(str, ts)), ha='center', va='center', fontsize=FS_TINY, color=INK if best[ts[0] - 1] else NEUTRAL)
        arrow(ax, q)
        z = rotated[np.argmax(best)]
        ax.plot(z.real, z.imag, 'o', ms=MS_LARGE, mfc='none', mec=INK, mew=LW, zorder=4)
        ax.set_title(title, fontsize=FS_SMALL)
        ax = axs[1, col]
        matrix(ax, A, 1.2)
        diagonal(ax, N)
        for t in j[best]:
            ax.add_patch(plt.Rectangle((t - 1, t - 1), 1, 1, fill=False, ec=INK, lw=LW_BOLD))
        ax.set_title(behavior, fontsize=FS_SMALL)
        ax.set_xlabel('position $j$'), ax.set_ylabel('content $i$')
        ax = axs[2, col]
        ax.axhline(0, color=LIGHT, lw=LW_HAIR, zorder=0)
        ax.axhline(1, color=GUIDE, lw=LW_HAIR, ls='--', zorder=0)
        ax.vlines(j, 0, logits, color=color, lw=LW)
        ax.plot(j[~best], logits[~best], 'o', ms=MS, mew=0, color=color)
        ax.plot(j[best], logits[best], 'o', ms=MS_LARGE, mfc=color, mec=INK, mew=LW)
        ax.set_xticks(j[best], [str(t) for t in j[best]], fontsize=FS_SMALL), ax.set_xticks(j, minor=True)
        ax.set_xlim(0.3, N + 0.7), ax.set_ylim(-1.2, 1.35), ax.set_yticks([-1, 0, 1], ['$-r$', '0', '$r$'], fontsize=FS_SMALL)
        ax.set_xlabel('position $t$'), ax.set_ylabel(r'logit $\sqrt{d_h}\,\lambda_t$')
    handles = [plt.Line2D([], [], color=INK, lw=LW, marker=r'$\rightarrow$', ms=0), plt.Line2D([], [], ls='none', marker='o', ms=MS, mew=0, color=GUIDE), plt.Line2D([], [], ls='none', marker='o', ms=MS_LARGE, mfc='none', mec=INK, mew=LW)]
    fig.legend(handles, ['query $q$', r'rotated key $k(x_j)\,e^{-i(T-j)\theta}$, labelled by $j$', 'largest logit'], loc='outside lower center', ncols=3, fontsize=FS_SMALL)
    save_figure(fig, HERE / 'single_plane_heads.pdf')


if __name__ == '__main__':
    for draw in (overview, three_parts, distances, statistics, plane_parts, interaction_profiles, single_plane_heads):
        draw()
