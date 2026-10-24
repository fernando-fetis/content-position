'''Figures of NN_name, drawn from results/summary.json only, in the order the document uses them.'''
from __future__ import annotations

from pathlib import Path

from src.utils.plots import FS_SMALL, LW, MS, figure, model_color, model_label, save_figure


def quantity(s: dict, models: list[str], out: Path) -> None:
    '''<The comparison the figure makes readable>: <panels, axes, references>.'''
    fig = figure(2.4)
    ax = fig.subplots()
    for m in models:
        ...
    save_figure(fig, out / 'quantity.pdf')


def draw(cfg: dict, s: dict, out: Path) -> None:
    '''Every figure the document uses, in their order.'''
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in cfg['models'] if m in s['models']]
    if models:
        quantity(s, models, out)
