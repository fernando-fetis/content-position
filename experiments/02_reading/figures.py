'''The figure of 02_reading, drawn from results/summary.json only.'''
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plots import BACKGROUND, FS_SMALL, FS_TINY, GUIDE, INK, LIGHT, LW, LW_HAIR, LW_THIN, MESH_LW, MS_LARGE, NEUTRAL, PART_COLORS, SCATTER, SCATTER_LARGE, SEQ_CMAP, figure, save_figure

from toy import EDITS

VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
TASK_COLORS = {'positional': PART_COLORS['positional'], 'symbolic': PART_COLORS['content'], 'interaction': PART_COLORS['interaction']}
EDIT_LABELS = {'full': 'all', 'drop_e': '$c + p$', 'only_c': '$c$', 'only_p': '$p$', 'only_e': '$\\epsilon$', 'drop_c': '$p + \\epsilon$', 'drop_p': '$c + \\epsilon$'}
OVERVIEW_EDITS = ('full', 'only_c', 'only_p', 'only_e', 'drop_e', 'drop_p', 'drop_c')
VERTEX_PART = {'symbolic': 'c', 'positional': 'p', 'interaction': 'e'}
PART_OF = {'c': 'content', 'p': 'positional', 'e': 'interaction'}


def _xy(b: np.ndarray) -> np.ndarray:
    return np.asarray(b, float) @ VERTICES


def _overview_simplex(ax: plt.Axes, t: dict) -> None:
    for k in (0.25, 0.5, 0.75):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            u, v = np.zeros(3), np.zeros(3)
            u[a], u[3 - a - b], v[b], v[3 - a - b] = 1 - k, k, 1 - k, k
            ax.plot(*np.vstack([_xy(u), _xy(v)]).T, color=LIGHT, lw=LW_HAIR, zorder=0)
    ax.plot(*np.vstack([VERTICES, VERTICES[:1]]).T, color=NEUTRAL, lw=LW_THIN, zorder=1)
    for (x, y), task, ha, va, (dx, dy), part in zip(VERTICES, ('symbolic', 'positional', 'interaction'), ('center', 'center', 'left'), ('top', 'top', 'center'), ((0, -0.05), (0, -0.05), (0.07, 0)), ('$c$', '$p$', '$\\epsilon$')):
        ax.text(x + dx, y + dy, f'{task}\nonly {part}', fontsize=FS_SMALL, ha=ha, va=va, color=TASK_COLORS[task], linespacing=1.1)
    for task, col in TASK_COLORS.items():
        if task in t['trained']:
            b, ok = np.asarray(t['trained'][task]['behavior']), np.asarray(t['trained'][task]['solved'], bool)
            ax.scatter(*_xy(b[ok]).T, s=SCATTER, marker='o', color=col, lw=0, alpha=0.85, zorder=3)
            ax.scatter(*_xy(b[~ok]).T, s=SCATTER, marker='o', facecolor=BACKGROUND, edgecolor=col, lw=LW_THIN, zorder=2)
        if task in t['multi']:
            ax.scatter(*_xy(t['multi'][task]['reader_behavior']).T, s=SCATTER, marker='s', facecolor='none', edgecolor=col, lw=LW, zorder=4)
        if task in t['hand']:
            ax.scatter(*_xy(np.asarray(t['hand'][task]['behavior'])[None]).T, s=SCATTER_LARGE, marker='*', color=col, edgecolor=INK, lw=LW_HAIR, zorder=5)
    handles = [plt.Line2D([], [], ls='none', marker='*', ms=MS_LARGE, mfc=GUIDE, mec=INK, mew=LW_HAIR), plt.Line2D([], [], ls='none', marker='o', ms=MS_LARGE, mfc=GUIDE, mew=0), plt.Line2D([], [], ls='none', marker='o', ms=MS_LARGE, mfc=BACKGROUND, mec=GUIDE, mew=LW_THIN), plt.Line2D([], [], ls='none', marker='s', ms=MS_LARGE, mfc='none', mec=GUIDE, mew=LW)]
    ax.legend(handles, ['built by hand', 'one head, solves', 'one head, fails', 'reader of three heads'], fontsize=FS_SMALL, loc='upper left', bbox_to_anchor=(-0.16, 1.03), handletextpad=0.2, labelspacing=0.3, borderaxespad=0)
    ax.set_xlim(-0.12, 1.12), ax.set_ylim(-0.17, np.sqrt(3) / 2 + 0.15), ax.set_aspect('equal'), ax.axis('off')


def _overview_edits(ax: plt.Axes, t: dict, chance: dict) -> None:
    tasks = [task for task in TASK_COLORS if task in t['multi']]
    M = np.array([[t['multi'][task]['edits_mean'][k] for k in OVERVIEW_EDITS] for task in tasks])
    ax.pcolormesh(np.arange(M.shape[1] + 1), np.arange(M.shape[0] + 1), M, vmin=0, vmax=1, cmap=SEQ_CMAP, edgecolors='face', linewidth=MESH_LW, antialiased=False, rasterized=False)
    for r, task in enumerate(tasks):
        for k, edit in enumerate(OVERVIEW_EDITS):
            a = M[r, k]
            ax.text(k + 0.5, r + 0.5, f'{a:.2f}', ha='center', va='center', fontsize=FS_SMALL, color=BACKGROUND if a < 0.6 else INK)
            if VERTEX_PART[task] in EDITS[edit]:
                ax.add_patch(plt.Rectangle((k + 0.06, r + 0.06), 0.88, 0.88, fill=False, ec=INK, lw=LW_THIN, zorder=3))
    for x in np.arange(1, M.shape[1]):
        ax.axvline(x, color=BACKGROUND, lw=LW_THIN)
    for y in np.arange(1, M.shape[0]):
        ax.axhline(y, color=BACKGROUND, lw=LW_THIN)
    ax.axvline(1, color=BACKGROUND, lw=LW), ax.axvline(4, color=BACKGROUND, lw=LW)
    ax.set_xlim(0, M.shape[1]), ax.set_ylim(M.shape[0], -0.6)
    for k, edit in enumerate(OVERVIEW_EDITS):
        for m, part in enumerate('cpe'):
            kept = part in EDITS[edit]
            ax.plot(k + 0.5 + (m - 1) * 0.2, -0.3, 'o', ms=MS_LARGE, mfc=PART_COLORS[PART_OF[part]] if kept else BACKGROUND, mec=PART_COLORS[PART_OF[part]], mew=LW_THIN, clip_on=False)
    ax.set_xticks(np.arange(M.shape[1]) + 0.5, [EDIT_LABELS[k] for k in OVERVIEW_EDITS], fontsize=FS_SMALL)
    ax.set_yticks(np.arange(len(tasks)) + 0.5, tasks)
    for r, task in enumerate(tasks):
        ax.get_yticklabels()[r].set_color(TASK_COLORS[task])
        ax.text(-0.12, r + 0.82, f'chance {chance[task]}', ha='right', va='center', fontsize=FS_TINY, color=NEUTRAL)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_xlabel('parts kept in the logits of every head (framed: keeps the part of the task)')
    ax.text(-0.12, -0.3, 'kept', ha='right', va='center', fontsize=FS_SMALL, color=NEUTRAL)
    ax.set_title('accuracy of the three-head models', loc='center')


def toy_overview(s: dict, out: Path, chance: dict) -> None:
    '''Left: the simplex, each head coloured by its task (the colour of the vertex of the part the task needs): hand-built heads (stars), single trained heads that solve their task (dots) or not (rings), readers of the three-head models (squares). Right: mean accuracy over seeds of the three-head models when every head's logits keep only some parts (dots above: content, position, interaction, filled if kept); framed cells keep the part of the task's vertex.'''
    t = s['toy']
    fig = figure(2.45)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.75])
    _overview_simplex(fig.add_subplot(gs[0]), t)
    _overview_edits(fig.add_subplot(gs[1]), t, chance)
    save_figure(fig, out / 'toy_overview.pdf')


def draw(cfg: dict, s: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if s['toy']['trained'] or s['toy']['hand']:
        n = cfg['toy_model']['length'] - 2
        toy_overview(s, out, {'positional': f"1/{cfg['toy_model']['symbols']}", 'symbolic': f"1/{cfg['toy_model']['symbols']}", 'interaction': f'1/{n}'})
