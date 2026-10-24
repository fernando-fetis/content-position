'''The style of every figure (palette, fonts, sizes) and drawing helpers; figures are drawn at their printed size.'''
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEXT_WIDTH = 6.77  # text width of the document (US letter, 2.2 cm margins), in inches
# Text: figures are drawn at their printed size and included at it, so these are the sizes on the page.
FS = 8  # axis labels, titles, tick labels, legends
FS_SMALL = 7  # annotations, legends and tick labels of crowded panels
FS_TINY = 6  # in-panel numbers
OPERATOR = 11  # the = and + between panels
# Lines and markers: four line weights and three marker sizes, nothing in between.
LW_BOLD = 1.6  # emphasis: the shared profile, the curve a panel is about
LW = 1.0  # data
LW_THIN = 0.6  # secondary data, many overlaid lines, axes
LW_HAIR = 0.4  # guides, leader lines, deviations
MS_LARGE = 4.5  # highlighted points
MS = 3.0
MS_SMALL = 1.6
SCATTER_LARGE = 40  # marker area (points squared) of a few highlighted points
SCATTER = 10  # of scatter plots with few points
SCATTER_SMALL = 2  # of dense scatter plots
ARROW_SCALE = 7  # size of arrow heads
MESH_LW = 0.3  # edges of matrix cells, drawn in the cell colour so that no seams show
# Greys, from ink to the lightest fill.
INK = '#1a1a1a'
NEUTRAL = '#4a4a4a'
MID = '#7a7a7a'
GUIDE = '#9a9a9a'
SOFT = '#bdbdbd'
LIGHT = '#d9d9d9'
BACKGROUND = 'white'  # fill of hollow markers
# The three parts of the decomposition, and the rules of the experiments (ours in blue, methods estimating the same quantity in teal, published methods warm, trivial baselines grey, part-defined controls in the colour of their part).
PART_COLORS = {'content': '#2a8a5a', 'positional': '#8e5bb5', 'interaction': '#d28500'}
OURS = '#1b4f8a'  # the main rule of the decomposition (position-dependent variance, positional ratio); also the principal-axes ellipse of the document
OURS_LIGHT = '#6f9fd6'  # the second rule of the decomposition (its bound)
METHOD_COLORS = {
    'variance': OURS, 'variance_4096': '#5b3a8c', 'variance_16000': '#8c6bb1', 'bound': OURS_LIGHT,
    'diagonal': '#3f9a96', 'elitekv': '#1a5e5a',
    'norm': '#b5574a', 'mkl': '#d9893b',
    'high': GUIDE,
    'intact': INK,
}
METHOD_LABELS = {'variance': 'position-dependent variance (ours)', 'variance_4096': 'position-dependent variance at 4096 tokens (ours)', 'variance_16000': 'position-dependent variance at 16000 tokens (ours)', 'bound': 'bound (ours)', 'diagonal': 'direct effect', 'elitekv': 'EliteKV (RoPElite)', 'norm': 'norm (MHA2MLA)', 'mkl': 'KL (MHA2MLA-VLM)', 'high': 'fastest planes', 'intact': 'intact'}
# Models: one name and one colour per model, the same in every figure (the seven base models of the benchmark ordered by family and size along the sequential map).
MODEL_LABELS = {'qwen2.5-0.5b': 'Qwen2.5-0.5B', 'qwen2.5-1.5b': 'Qwen2.5-1.5B', 'qwen2.5-3b': 'Qwen2.5-3B', 'qwen2.5-7b': 'Qwen2.5-7B', 'llama-3.2-1b-base': 'Llama-3.2-1B', 'llama-3.2-3b-base': 'Llama-3.2-3B', 'llama-3.1-8b-base': 'Llama-3.1-8B'}
CMAP = 'RdBu_r'  # diverging: signed matrices and correlations, centered at zero
SEQ_CMAP = 'viridis'  # sequential: models by size, depth, fractions (0 to 1)
matplotlib.colorbar.Colorbar.n_rasterize = 10**9  # colorbar solids stay vector
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['cmr10'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'font.size': FS,
    'axes.titlesize': FS,
    'axes.labelsize': FS,
    'xtick.labelsize': FS,
    'ytick.labelsize': FS,
    'legend.fontsize': FS,
    'figure.titlesize': FS,
    'text.color': INK,
    'axes.labelcolor': INK,
    'axes.edgecolor': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'axes.linewidth': LW_THIN,
    'xtick.major.width': LW_THIN,
    'ytick.major.width': LW_THIN,
    'xtick.minor.width': LW_HAIR,
    'ytick.minor.width': LW_HAIR,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
    'axes.titlepad': 4,
    'axes.labelpad': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': LW,
    'lines.markersize': MS,
    'patch.linewidth': LW_THIN,
    'hatch.linewidth': LW_HAIR,
    'grid.linewidth': LW_HAIR,
    'grid.color': LIGHT,
    'scatter.edgecolors': 'none',
    'legend.frameon': False,
    'legend.handlelength': 1.8,
    'legend.handletextpad': 0.5,
    'legend.columnspacing': 1.5,
    'legend.borderaxespad': 0.2,
    'errorbar.capsize': 1.5,
    'figure.constrained_layout.h_pad': 0.02,
    'figure.constrained_layout.w_pad': 0.02,
    'pdf.fonttype': 42,
})


def matrix(ax: plt.Axes, M: np.ndarray, vmax: float) -> None:
    ax.pcolormesh(np.arange(M.shape[1] + 1), np.arange(M.shape[0] + 1), M, vmin=-vmax, vmax=vmax, cmap=CMAP, edgecolors='face', linewidth=MESH_LW, antialiased=False, rasterized=False)
    ax.set_xlim(0, M.shape[1])
    ax.set_ylim(M.shape[0], 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(True)


def diagonal(ax: plt.Axes, n: int) -> None:
    ax.plot(np.arange(n) + 0.5, np.arange(n) + 0.5, 's', ms=MS, mfc='none', mec=INK, mew=LW_THIN)


def phasors(ax: plt.Axes, lim: float, circle: bool) -> None:
    if circle:
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(t), np.sin(t), color=GUIDE, lw=LW_THIN)
    else:
        ax.axhline(0, color=GUIDE, lw=LW_THIN)
        ax.axvline(0, color=GUIDE, lw=LW_THIN)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')


def arrow(ax: plt.Axes, tip: complex, tail: complex = 0, dashed: bool = False) -> None:
    ax.annotate('', xy=(tip.real, tip.imag), xytext=(tail.real, tail.imag), arrowprops={'arrowstyle': '-|>', 'color': INK, 'lw': LW, 'ls': '--' if dashed else '-', 'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': ARROW_SCALE})


def ordered(n: int) -> list:
    '''n colours for an ordered series (models by size), from the sequential map without its lightest end.'''
    return [plt.get_cmap(SEQ_CMAP)(x) for x in np.linspace(0.0, 0.8, n)]


def model_label(name: str, short: bool = False) -> str:
    '''The printed name of a model; short splits it before its size, for crowded tick labels.'''
    label = MODEL_LABELS.get(name, name)
    return label.rsplit('-', 1)[0] + '\n' + label.rsplit('-', 1)[1] if short and '-' in label else label


def model_color(name: str) -> tuple:
    '''The colour of a model: the seven base models along the sequential map; any other model in grey.'''
    base = list(MODEL_LABELS)
    return ordered(len(base))[base.index(name)] if name in base else NEUTRAL


def figure(height: float, width: float = 1.0, **kw: object) -> plt.Figure:
    '''A figure of the given height (inches) and width, a fraction of the text width; include it in Typst at its natural size, without `width`.'''
    return plt.figure(figsize=(width * TEXT_WIDTH, height), layout='constrained', **kw)


def save_figure(fig: plt.Figure, path: Path) -> None:
    '''Saves fig as a vector PDF cropped to its content with a minimal margin; refuses a figure with any raster image (imshow, rasterized artists).'''
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate': None}, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    if re.search(rb'/Subtype\s*/Image', path.read_bytes()):
        raise ValueError(f'{path}: the figure contains a raster image; every figure must be vector')
