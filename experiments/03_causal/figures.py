'''Figure of 03_causal, drawn from results/summary.json only: causal_overview, the effect of each edit in every group it is applied to (top) and each edited group against its control (bottom).'''
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.plots import BACKGROUND, FS_SMALL, FS_TINY, GUIDE, LIGHT, LW, LW_BOLD, LW_HAIR, LW_THIN, MID, MS, MS_LARGE, MS_SMALL, NEUTRAL, PART_COLORS, figure, model_label, save_figure


OVERVIEW = [('only_c', 'positional', 'position removed, 25% of the heads (keep $\\mu + c$)', {'positional': 'positional', 'symbolic': 'symbolic'}, ('positional', 'matched_positional'), 'position removed: most positional vs matched'),
            ('drop_c', 'content', 'content removed, 10% (keep $\\mu + p + \\epsilon$)', {'positional': 'positional', 'symbolic': 'symbolic'}, ('symbolic', 'matched_symbolic'), 'content removed: most symbolic vs matched'),
            ('drop_e', 'interaction', 'interaction removed, 25% (keep $\\mu + c + p$)', {'interacting': 'interaction'}, ('interaction', 'random_0'), 'interaction removed: most interacting vs random')]
SLOTS = {'positional': 0, 'symbolic': 1, 'interacting': 2.2}


def causal_overview(s: dict, out: Path) -> None:
    '''Top, absolute effects of each edit by head group, one thin line per model and the mean across models in bold (geometric mean for the NLL): drop of the standardized margin on the solved short (a) and long tasks (b), in standard deviations of each task's intact margins, and change of the text NLL (c, log scale); removing position (purple) from the 25% most positional and the 25% most symbolic heads, removing content (green) from the 10% most positional and most symbolic heads, removing interaction (orange) from the 25% most interacting heads. Bottom, the edited group against its control, per model: extra margin drop (control minus edited, positive when the edited group costs more) with 95% BCa intervals, paired over prompts, on the short (d) and long tasks (e), and the ratio of the text NLL changes (f, log scale); controls are the random groups of closest removed variance for position and content and random heads for interaction; open diamonds, the same comparisons against the matched control's effects divided by its variance ratio (the proportional reference).'''
    import matplotlib.pyplot as plt
    names = list(s['models'])
    fig = figure(4.2)
    axs = fig.subplots(2, 3, height_ratios=(1, 1.45))
    measures = [('short', '(a) short tasks', 'margin drop (SD of intact margins)'), ('long', '(b) long tasks', None), ('nll', '(c) text', 'NLL change (nats / token)')]
    value = lambda m, key, kind: m[key]['nll_change'] if kind == 'nll' else -m[key][kind]['margin_change_std']
    for col, (kind, title, ylabel) in enumerate(measures):
        ax = axs[0, col]
        for edit, part, _, groups, _, _ in OVERVIEW:
            color = PART_COLORS[part]
            x = [SLOTS[g] for g in groups]
            ys = np.array([[value(s['models'][n], f'{edit}__{key}', kind) for key in groups.values()] for n in names])
            for y in ys:
                ax.plot(x, y, color=color, lw=LW_THIN, alpha=0.45, marker='o', ms=MS_SMALL, mew=0)
            center = np.exp(np.log(ys).mean(0)) if kind == 'nll' else ys.mean(0)
            ax.plot(x, center, color=color, lw=LW_BOLD, marker='o', ms=MS_LARGE, mec=BACKGROUND, mew=LW_THIN, solid_capstyle='round')
        if kind == 'nll':
            ax.set_yscale('log')
        else:
            ax.axhline(0, color=GUIDE, lw=LW_HAIR, zorder=0)
        ax.set_xticks(list(SLOTS.values()), ['most\npositional', 'most\nsymbolic', 'most\ninteracting'], fontsize=FS_SMALL)
        ax.set_xlim(-0.35, 2.55)
        ax.axvline(1.6, color=LIGHT, lw=LW_HAIR, zorder=0)
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
    top = [-s['models'][n][f'{e}__{g}'][k]['margin_change_std'] for n in names for e, _, _, gs, _, _ in OVERVIEW for g in gs.values() for k in ('short', 'long')]
    for ax in axs[0, :2]:
        ax.set_ylim(min(top) - 0.3, max(top) + 0.3)
    axs[0, 1].tick_params(labelleft=False)
    rows, y = [], 0
    for edit, part, _, _, (group, control), header in OVERVIEW:
        rows.append((None, part, header, y))
        y += 1
        for n in names:
            rows.append((n, part, (edit, group, control), y))
            y += 1
        y += 0.5
    for col, (kind, title, xlabel) in enumerate([('short', '(d) short tasks', 'extra margin drop vs control (SD)'), ('long', '(e) long tasks', 'extra margin drop vs control (SD)'), ('nll', '(f) text', 'NLL change, edited / control')]):
        ax = axs[1, col]
        for n, part, spec, yy in rows:
            color = PART_COLORS[part]
            if n is None:
                if col == 0:
                    ax.text(0.0, yy + 0.1, spec, transform=ax.get_yaxis_transform(), color=color, fontsize=FS_SMALL, ha='left', va='center')
                continue
            edit, group, control = spec
            m = s['models'][n]
            g = m[f'{edit}__{group}']
            proportional = g.get('control')
            if kind == 'nll':
                ratio = g[f'nll_over_{control}']
                ax.plot(ratio, yy, 'o', ms=MS, color=color)
                if proportional:
                    ax.plot(proportional['nll_over_scaled'], yy, 'D', ms=MS, mfc=BACKGROUND, mec=color, mew=LW, zorder=3)
            else:
                d = g[kind][f'minus_{control}']
                lo, hi = -d['ci'][1], -d['ci'][0]
                ax.plot([lo, hi], [yy, yy], color=color, lw=LW, solid_capstyle='butt')
                ax.plot(-d['mean'], yy, 'o', ms=MS, color=color, mfc=BACKGROUND if lo < 0 < hi else color, mec=color, mew=LW_THIN)
                if proportional:
                    ax.plot(-proportional['minus_scaled'][kind], yy, 'D', ms=MS, mfc=BACKGROUND, mec=color, mew=LW, zorder=3)
        for _, block, *_ in OVERVIEW:
            ys = [r[3] for r in rows if r[0] and r[1] == block]
            ax.vlines(1 if kind == 'nll' else 0, min(ys) - 0.5, max(ys) + 0.5, color=GUIDE, lw=LW_HAIR, zorder=0)
        if kind == 'nll':
            ax.set_xscale('log')
        ax.set_ylim(y + 0.4, -0.6)
        ax.set_yticks([r[3] for r in rows if r[0]], [model_label(r[0]) for r in rows if r[0]] if col == 0 else [], fontsize=FS_SMALL)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.set_title(title), ax.set_xlabel(xlabel)
        ax.text(0.98, 0.0, 'edited costs more $\\rightarrow$', transform=ax.transAxes, fontsize=FS_TINY, color=MID, ha='right', va='bottom')
    bounds = [-v for n in names for e, _, _, _, (g, c), _ in OVERVIEW for k in ('short', 'long') for v in s['models'][n][f'{e}__{g}'][k][f'minus_{c}']['ci'] + ([s['models'][n][f'{e}__{g}']['control']['minus_scaled'][k]] if 'control' in s['models'][n][f'{e}__{g}'] else [])]
    for ax in axs[1, :2]:
        ax.set_xlim(min(bounds) - 0.3, max(bounds) + 0.3)
    handles = [plt.Line2D([], [], color=PART_COLORS[part], lw=LW_BOLD, marker='o', ms=MS_LARGE, mec=BACKGROUND, mew=LW_THIN) for _, part, *_ in OVERVIEW]
    handles += [plt.Line2D([], [], color=NEUTRAL, lw=LW_THIN, alpha=0.45, marker='o', ms=MS_SMALL, mew=0), plt.Line2D([], [], color=NEUTRAL, lw=LW_BOLD, marker='o', ms=MS_LARGE, mec=BACKGROUND, mew=LW_THIN), plt.Line2D([], [], color=NEUTRAL, lw=LW, marker='o', ms=MS, mfc=BACKGROUND, mec=NEUTRAL, mew=LW_THIN), plt.Line2D([], [], ls='none', marker='D', ms=MS, mfc=BACKGROUND, mec=NEUTRAL, mew=LW)]
    labels = [o[2] for o in OVERVIEW] + ['one model', 'mean of the models', '95% interval includes 0', 'against the proportional reference']
    order = [0, 3, 1, 4, 2, 5, 6]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='outside upper center', ncols=4, fontsize=FS_SMALL, frameon=False)
    save_figure(fig, out / 'causal_overview.pdf')


def draw(cfg: dict, s: dict, out: Path) -> None:
    causal_overview(s, out)
