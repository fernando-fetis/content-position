'''Figures of 04_planes, drawn from results/summary.json only: the planes that keep their rotation (models, tasks, sigma_zeta2_planes).'''
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from src.utils.plots import BACKGROUND, FS_SMALL, FS_TINY, GUIDE, INK, LW, LW_HAIR, LW_THIN, MESH_LW, METHOD_COLORS, METHOD_LABELS, MS, SEQ_CMAP, figure, model_color, model_label, save_figure

CALIBRATED = ('variance_4096', 'variance_16000')  # drawn hollow


def _title(key: str) -> str:
    corpus, n = key.rsplit('_', 1)
    return f"{'WikiText-2' if corpus == 'wiki' else corpus}, {n} tokens"


def models(s: dict, keys: list[str], fraction: str, rules: list[str], out: Path) -> None:
    '''Increase of the loss of each rule at one fraction (nats per token, log scale), one column of points per model, one panel per corpus and length; the calibrated variants hollow.'''
    names = list(s['models'])
    fig = figure(2.3)
    axes = fig.subplots(1, len(keys), sharey=True)
    w = 0.8 / len(rules)
    for i, (ax, key) in enumerate(zip(axes, keys)):
        for j, r in enumerate(rules):
            y = [s['models'][n]['loss_increase'][key][fraction][r] for n in names]
            c = METHOD_COLORS[r]
            ax.plot(np.arange(len(names)) + (j - (len(rules) - 1) / 2) * w, y, 'o', ms=MS, color=c, mfc=BACKGROUND if r in CALIBRATED else c, mew=LW_THIN, label=METHOD_LABELS[r])
        for x in np.arange(len(names) - 1) + 0.5:
            ax.axvline(x, color=GUIDE, lw=LW_HAIR)
        ax.set_yscale('log'), ax.set_xlim(-0.5, len(names) - 0.5), ax.set_title(f'({"ab"[i]}) {_title(key)}')
        ax.set_xticks(np.arange(len(names)), [model_label(n) for n in names], fontsize=FS_SMALL, rotation=35, ha='right', rotation_mode='anchor')
    axes[0].set_ylabel('loss increase (nats / token)')
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=FS_SMALL, ncol=3, loc='outside upper center')
    save_figure(fig, out / 'models.pdf')


def tasks(s: dict, rules: list[str], out: Path) -> None:
    '''Accuracy (percent of prompts with a positive margin) of the intact model and of every rule at the fraction of the tasks, one heatmap per task, rules by rows and models by columns.'''
    names = list(s['models'])
    rows = ['intact', *rules]
    tasks = {t: f"needle, {t.split('_')[1]} tokens" if t.startswith('niah_') else t for t in s['models'][names[0]]['tasks']}
    fig = figure(2.45)
    axes = np.atleast_1d(fig.subplots(1, len(tasks), sharey=True))
    for k, (ax, (t, title)) in enumerate(zip(axes, tasks.items())):
        M = np.array([[s['models'][n]['tasks'][t][r] for n in names] for r in rows])
        ax.pcolormesh(np.arange(len(names) + 1), np.arange(len(rows) + 1), M, vmin=0, vmax=1, cmap=SEQ_CMAP, edgecolors='face', linewidth=MESH_LW, antialiased=False)
        for a in range(len(rows)):
            for b in range(len(names)):
                ax.text(b + 0.5, a + 0.5, f'{round(100 * M[a, b])}', ha='center', va='center', fontsize=FS_TINY, color=BACKGROUND if M[a, b] < 0.7 else INK)
        ax.axhline(1, color=BACKGROUND, lw=LW)
        ax.set_ylim(len(rows), 0), ax.set_title(f'({"abcd"[k]}) {title}', fontsize=FS_SMALL)
        ax.set_xticks(np.arange(len(names)) + 0.5, [model_label(n) for n in names], fontsize=FS_SMALL, rotation=90)
        ax.tick_params(length=0)
    axes[0].set_yticks(np.arange(len(rows)) + 0.5, [METHOD_LABELS[r] for r in rows], fontsize=FS_SMALL)
    save_figure(fig, out / 'tasks.pdf')


def sigma_zeta2_planes(s: dict, fraction: str, rules: list[str], lengths: list[str], threshold: float, out: Path) -> None:
    '''(a) Per rule, the share of the planes it keeps at fraction whose sigma_zeta^2 at the shortest length is below threshold, one point per model. (b) The increase of the loss of the norm rule over that of the position-dependent variance at the shortest and longest length, one line per model and corpus.'''
    names = list(s['models'])
    short, long = lengths[0], lengths[-1]
    fig = figure(2.2)
    a, b = fig.subplots(1, 2, width_ratios=[1.2, 1])
    for k, r in enumerate(rules):
        for n in names:
            a.plot(s['models'][n]['sigma_zeta2'][short][r]['share_below'], k, 'o', ms=MS, color=model_color(n), alpha=0.9)
    a.set_yticks(range(len(rules)), [METHOD_LABELS[r] for r in rules]), a.set_ylim(len(rules) - 0.5, -0.5), a.set_xlim(-0.03, 1)
    a.set_xlabel(rf'share of the kept planes with $\sigma_{{\zeta,f}}^2 < {Fraction(threshold)}$ ({short} tokens)'), a.set_title(rf'(a) planes with $\sigma_{{\zeta,f}}^2 < {Fraction(threshold)}$ ({float(fraction):g} of the planes kept)')
    for corpus, mk, ls in (('wiki', 'o', '-'), ('code', 's', '--')):
        for n, r in s['norm_to_variance'][corpus]['ratio'].items():
            b.plot([0, 1], [r[short], r[long]], ls, marker=mk, ms=MS, lw=LW_THIN, color=model_color(n), mfc=model_color(n) if corpus == 'wiki' else BACKGROUND)
    b.set_yscale('log'), b.set_yticks([2, 5, 10, 20, 50, 100], ['2', '5', '10', '20', '50', '100']), b.set_ylim(1.5, 120), b.minorticks_off(), b.set_xticks([0, 1], [f'{short} tokens', f'{long} tokens']), b.set_xlim(-0.2, 1.2), b.set_ylabel('loss increase, norm / ours')
    b.set_title('(b) the norm rule over ours')
    handles = [b.plot([], [], 'o', ms=MS, color=model_color(n), label=model_label(n))[0] for n in names]
    handles += [b.plot([], [], '-o', ms=MS, lw=LW_THIN, color=INK, label='WikiText-2')[0], b.plot([], [], '--s', ms=MS, lw=LW_THIN, color=INK, mfc=BACKGROUND, label='code')[0]]
    fig.legend(handles=handles, fontsize=FS_SMALL, ncol=5, loc='outside lower center')
    save_figure(fig, out / 'sigma_zeta2_planes.pdf')


def draw(cfg: dict, s: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    f, fraction = cfg['figures'], str(cfg['fraction'])
    models(s, f['keys'], fraction, f['rules'], out)
    tasks(s, f['rules'], out)
    sigma_zeta2_planes(s, fraction, cfg['norm_slow']['rules'], [str(t) for t in cfg['text_tokens']], cfg['norm_slow']['threshold'], out)
