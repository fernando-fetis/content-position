'''Figures of 05_finetune, drawn from results/summary.json only.'''
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.plots import FS_SMALL, LW, METHOD_COLORS, METHOD_LABELS, MS_SMALL, TEXT_WIDTH, figure, model_label, save_figure

LABELS = METHOD_LABELS | {'intact': 'full RoPE, fine-tuned'}
CORPORA = {'wiki': 'WikiText-2', 'code': 'code'}


def draw(cfg: dict, s: dict, out: Path) -> None:
    '''One row per model: per corpus and length of figures.keys, the gap of each rule to the intact model fine-tuned on the same tokens (nats, log scale) against the fine-tuning tokens; last panel, the accuracy of every run on the needle in a haystack at the longest length.'''
    models, keys = list(s['models']), cfg['figures']['keys']
    cols = len(keys) + 1
    fig = figure(1.7 * len(models) + 0.3, width=min(1.0, 1.6 * cols / TEXT_WIDTH))
    handles: dict[str, object] = {}
    for i, n in enumerate(models):
        d = s['models'][n]
        x = np.asarray(d['tokens'], float) / 1e6
        for j, key in enumerate(keys):
            ax = fig.add_subplot(len(models), cols, cols * i + j + 1)
            for r, y in d['gap'].get(key, {}).items():
                if r != 'intact':
                    handles[r], = ax.plot(x[:len(y)], y, color=METHOD_COLORS[r], marker='o', ms=MS_SMALL, lw=LW)
            ax.set_yscale('log'), ax.set_xlabel('fine-tuning tokens (M)'), ax.set_title(f'({"abcdefgh"[j]}) {CORPORA.get(key.split("_")[0], key.split("_")[0])}, {key.split("_")[1]} tokens', fontsize=FS_SMALL)
            if j == 0:
                ax.set_ylabel(f'{model_label(n)}\ngap to fine-tuned intact (nats)')
        ax = fig.add_subplot(len(models), cols, cols * (i + 1))
        task = max((t for t in d['tasks'] if t.startswith('niah_')), key=lambda t: int(t.split('_')[-1]), default=None)
        if task:
            for r, rows in d['tasks'][task].items():
                handles[r], = ax.plot(x[:len(rows)], rows, color=METHOD_COLORS[r], marker='o', ms=MS_SMALL, lw=LW)
            ax.set_ylim(0, 1.05), ax.set_xlabel('fine-tuning tokens (M)'), ax.set_ylabel('accuracy'), ax.set_title(f'({"abcdefgh"[len(keys)]}) needle, {task.split("_")[-1]} tokens', fontsize=FS_SMALL)
    fig.legend(handles.values(), [LABELS[r] for r in handles], loc='outside lower center', ncol=3, fontsize=FS_SMALL)
    save_figure(fig, out / 'finetune.pdf')
