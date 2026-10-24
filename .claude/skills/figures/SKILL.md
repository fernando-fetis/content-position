---
name: figures
description: How every figure of the project is made, synthetic (docs/figures/figures.py) or from experiments (each experiment's figures.py): where it lives, how to choose the representation, printed size, fonts, line weights, markers, the shared palette, model names and colours, vector output, and inclusion in the Typst documents. Use whenever drawing, changing, reviewing or including a figure.
---

# Figures

All style lives in `src/utils/plots.py` (sizes, fonts, line weights, markers, the palette and the colormaps, `figure`, `save_figure`, `model_label`, `model_color`, `ordered`); figure code imports its constants and never writes a size, a weight or a colour of its own. Every figure of the project looks like it belongs to one family.

## Where a figure lives
- From real results: in the experiment that produces them, drawn by its `figures.py` from its summary only (`figures/<name>.pdf`), never by a script outside it.
- Synthetic (a construction, a schematic, an example with made-up numbers): in `docs/figures/figures.py` (`make theory-figures`), drawn to `docs/figures/<name>.pdf`.
- Only the figures the document includes are drawn; a figure the document drops is deleted, PDF and code (helpers left without a caller included).

## Choosing the representation
Before drawing, state the one comparison the figure must make readable (a ratio against its bound, a trend with length, two groups against each other) and pick the form that makes it a short visual judgement: positions on a shared axis before lengths, lengths before areas, areas before colours; distributions as densities, quantile bands or boxes rather than overlapping points; the same quantity on the same axis and scale across panels; a log scale when the values span orders of magnitude. Put the reference the reader compares with in the figure (a bound, $y = x$, a chance level, a limit) and mark it in the legend. One message per figure: split a figure that tries to say two things, and remove panels, labels and decorations that do not serve the comparison. When a figure is requested, propose the representation that answers the question best, even if it differs from the one asked for, and say why in one line.

## Size and text
- A figure is drawn at its printed size and included at it: `figure(height, width)` takes the height in inches and the width as a fraction of the text width (`TEXT_WIDTH = 6.77` in, the same in every document), and the document includes it with `image("...")`, never with `width:`, so the text of every figure has the same size on the page.
- Text sizes: `FS` (8 pt) for titles, axis labels, tick labels and legends; `FS_SMALL` (7 pt) for annotations and crowded panels; `FS_TINY` (6 pt) only for numbers inside panels; `OPERATOR` for the = and + between panels. Computer Modern, like the documents.
- Text inside figures never cites a statement by number or names the document (the caption says which statement a curve or a band is); it is in English with the notation of the documents (e.g. $\mathbb{V}$ for variances, $\sigma_{\zeta,f}^2$); captions are written in the documents, in their language. Panels are labelled (a), (b), … in their titles, and the caption describes each.
- Panels in one row have the same height (`fig.subplots` or a gridspec with equal rows); either every panel carries a label the others lack, or none does. Tick labels never overlap: shorten them (`model_label(name, short=True)`), rotate full names, or move the legend.

## Lines and markers
- Four line weights: `LW_BOLD` (1.6, the curve a panel is about), `LW` (1.0, data and arrows), `LW_THIN` (0.6, secondary data, many overlaid lines, bar edges, axes), `LW_HAIR` (0.4, guides, leader lines). Nothing in between.
- Markers: `MS_LARGE`, `MS`, `MS_SMALL` for plotted points; `SCATTER_LARGE`, `SCATTER`, `SCATTER_SMALL` for scatter areas (few points, normal, dense); `ARROW_SCALE` for arrow heads.

## Colour
- Greys, darkest to lightest: `INK`, `NEUTRAL`, `MID`, `GUIDE`, `SOFT`, `LIGHT`.
- The three parts always in `PART_COLORS` (content green, positional purple, interaction orange); the rules of the experiments in `METHOD_COLORS` (ours blue, estimators of the same quantity teal, published methods warm, baselines grey). A new rule gets its colour there, in the family it belongs to, never in a figure file.
- Models: `model_label(name)` and `model_color(name)`, so a model has the same name and colour in every figure; ordered series that are not models (lengths, cases) use `ordered(n)`.
- Matrices and correlations: `CMAP` (diverging, centered at zero); sequential quantities: `SEQ_CMAP`.

## Vector only
Every figure is a vector PDF: `save_figure` crops to the content with a 0.02 in margin and refuses raster images. Matrices with `matrix()` or `pcolormesh(edgecolors='face', linewidth=MESH_LW)`, never `imshow`; no `rasterized=True`, colorbars included.

## Checking a figure
After drawing, render it with `.claude/skills/figures/scripts/review.sh <pdf or folder>` (PNGs in `.temp/figures/`; it also fails on raster images) and look at it: no overlapping text, legends off the data, nothing cut, the comparison readable at printed size. Regenerating an unchanged figure must give a byte-identical PDF.

## In the document
Figures float as in LaTeX (top or bottom of a page, or the next one); the text cites each figure where it is discussed (`@fig:name`), so its position does not matter. `#place.flush()` before the bibliography keeps floats out of it. No figure may land inside a proof: check the compiled document after any change, and where one does, move its source after the proof or give it `placement: bottom`.
