# 03_causal

The causal test of the parts (see Sources): in five models (Qwen2.5-1.5B, 3B, 7B, Llama-3.2-3B, Llama-3.1-8B), the logits of a group of heads are replaced by the diagonal of the matrix that keeps only some parts of `def:content-positional-and-interaction-parts`, and the effect is measured on solved tasks and on text, against controls: the random group of heads whose removed variance is closest to the group's, and random heads.

Importance: high.

## Sources
- Definitions: `docs/notes.typ` (`def:content-positional-and-interaction-parts`, `def:behavior-and-positional-ratio`, `prop:the-positional-ratio-as-a-relative-distance`, `prop:the-parts-of-a-plane`; the margin and accuracy in `sec:tasks`, sink heads in `sec:measurements`).
- Methodology: `docs/notes.typ` (`sec:stability`, the paragraphs *Edición de las partes*, *Efecto* and *Alcance del control*).
- Results: `docs/notes.typ` (`sec:stability`).

## Files
- `run.py`: the stages `compute` (per model: groups, edits, margins and text loss), `summarize`, `figures`, `check` (the part edit against the explicit matrix and the identity edit against the intact model).
- `figures.py`: the figure of the document listed under Figures.
- `config.yaml`: every parameter; `smoke:` for a quick CPU run.
- `parts.py`: the part edit of the logits at every query (the parts in the closed forms of `prop:the-parts-of-a-plane`, from running means) and its two checks.

## Running
On a SLURM cluster, one job per model (`psym-03-<model>`, `src/remote/launch.py`), after 01's; it reads the per-head table of 01's calibration (`calibration_experiment`, `calibration_tokens`: 16 windows of 4096 tokens of WikiText-2, 16 queries each). `make summarize|figures|check E=03_causal`.

## Main config keys
- `edits`, by the parts kept: `only_c` keeps $\mu + c$, the nearest symbolic matrix of `prop:the-positional-ratio-as-a-relative-distance` (position and interaction removed); `drop_c` keeps $\mu + p + \epsilon$ (content removed); `drop_e` keeps $\mu + c + p$ (interaction removed).
- Groups (`groups` and `fraction` of each edit): the 25% of non-sink heads with the largest $P_h$, the 10% with the smallest, the 25% with the largest interaction share; the opposite extremes; random heads (one uniform draw among the non-sink heads, `random_0`; the same heads for `only_c` and `drop_e`, which share their fraction); and a closest-variance control: among `matched_draws` (4000) uniform random groups of non-sink heads of the same size (they may share heads with the group), the one whose removed variance (the calibration variances of the removed parts, summed over its heads) is closest to the group's. Group sizes are `round(fraction * non-sink heads)`, sinks by 01's calibration (mean attention on the first token above 0.8). The first group of each edit is compared with its control (the closest-variance one for `only_c` and `drop_c`, random for `drop_e`).
- Tasks (`tasks`, `n`, `target`): retrieval, binding, latest and successor (64 prompts each) and the needle and key–value at 4000 tokens (10 each), with new prompts (seed 1); a task enters the statistics of a model only if its intact accuracy (fraction of prompts with a positive answer-minus-distractor margin) is at least `solved` (0.9). Effect: change of the margin over the (population) standard deviation of the task's intact margins, pooled over the prompts of the solved tasks of each class (`short`, `long`); the group minus its control is paired by prompt, with a 95% BCa interval resampling those pooled prompts; text (`text`, `text_n`, `text_tokens`): change of the mean loss per token on 8 windows of 1024 tokens of WikiText-2, disjoint from the calibration.

## Summary (`results/summary.json`, key `models.<model>`)
`intact` (accuracy per task), `solved`; per edit and group, `<edit>__<group>` with `nll_change` and, under `short` and `long`, `margin_change_std` and `accuracy_change`; for the first group of each edit, also `minus_matched_<group>` or `minus_random_0` (mean and `ci`) under `short` and `long`, `nll_over_<control>` (the ratio of the text NLL changes), and with a matched control `control`: `variance_ratio` (the variance of the parts it removes over the group's, each summed over the heads), `shared_heads` out of `size` and `shared_fraction`, `minus_scaled` (the group's standardized margin change minus the control's divided by `variance_ratio`, the proportional reference, per kind) and `nll_over_scaled` (the group's text NLL change over the control's divided by `variance_ratio`).

## Figures (labels in `docs/notes.typ`)
- `causal_overview.pdf`: `fig:dissociation-exp`.

## Notes
- Names stored in the raw results, with their symbols in `sec:stability`: the edits `only_c` (quitar la posición, $K = \{c\}$), `drop_c` (quitar el contenido, $K = \{p, \epsilon\}$) and `drop_e` (quitar la interacción, $K = \{c, p\}$), `keep` for $K$; the groups `positional` ($G_\text{pos}$ for `only_c`, the opposite extreme for `drop_c`), `symbolic` ($G_\text{sim}$ for `drop_c`, the opposite extreme for `only_c`), `interaction` ($G_\text{int}$), `random_0` (the random heads) and `matched_<group>` (the control).
- Sign convention: `margin_change_std` is $\mathbb{E}_x[(m'(x) - m(x)) / \sigma_{D(x)}]$, the negative of the margin drop of the text, and `minus_<control>` is the group's value minus the control's, the negative of the difference of their drops.
- `groups` makes two unused draws before the candidates of the control so that every group is the one its stored result was computed with.
