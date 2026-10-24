# 04_planes

Removing the rotation without training, at three granularities (see Sources): in the seven base models (Qwen2.5-0.5B, 1.5B, 3B, 7B, Llama-3.2-1B, 3B, Llama-3.1-8B), each rule chooses, in every KV head, the planes that keep their rotation (partial RoPE), or the whole KV heads or whole layers that lose it; the exact edit is applied and the edited model is scored by the increase of the next-token loss on held-out text and code at two lengths and by the accuracy on retrieval, binding and the needle in a haystack. The plane rules compare the cost of `prop:the-cost-of-removing-a-rotation` (the per-plane position-dependent variance, also calibrated on longer windows, and its bound of `cor:where-each-part-can-live`) with the published ones (MHA2MLA's norms, MHA2MLA-VLM's KL criterion, EliteKV's greedy order), the direct effect and the fastest planes, and the phase variance $sigma_(zeta,f)^2$ of `prop:phase-statistics` gives the share of the planes each rule keeps with $sigma_(zeta,f)^2 < 1\/2$; the head and layer rules compare the summed position-dependent variance of `cor:heads-without-rotation` with the positional ratio of `def:behavior-and-positional-ratio`, the opposite choice, random choices, the interleaved pattern of hybrid models and the first or last layers. Every score comes from the calibration of `01_measure`; the masks are read by `05_finetune`. It is used by the applications (`sec:removing-rotation`, `sec:nope`, `sec:planes`).

Importance: high.

## Sources
- Definitions: `docs/notes.typ` (`prop:the-cost-of-removing-a-rotation`, `cor:heads-without-rotation`, `cor:where-each-part-can-live`, `prop:phase-statistics`, `cor:slow-planes-and-long-contexts`, `prop:head-as-a-sum-of-planes`, `def:behavior-and-positional-ratio`, `prop:the-positional-ratio-as-a-relative-distance`).
- Methodology: `docs/notes.typ` (`sec:removing-rotation`: the paragraph *Evaluación*; `sec:planes`: `tab:plane-rules` and the paragraphs around it; `sec:nope`: *Reglas para cabezales y capas*).
- Results: `docs/notes.typ` (`sec:planes-training-free`: `fig:planes-models`, `fig:planes-tasks`, `fig:planes-sigma-zeta2`; `sec:nope`, from its *Predicciones* on: `tab:nope-heads`, `tab:nope-layers`). Heads and layers on Llama-3.1-8B are pending.

## Files
- `run.py`: the stages `compute` (per model: the plane masks of every rule and fraction, the KV heads and layers of every head and layer rule, the intact and edited evaluations, each layer alone), `summarize`, `figures`, `check` (partial RoPE with every plane kept against the intact model; masks with the right number of planes, nested across fractions; head and layer plans that remove exactly their heads or layers).
- `figures.py`: the figures listed below.
- `config.yaml`: every parameter; `smoke:` runs tiny-llama on the CPU.
- Shared code: `src/analysis/plane_scores.py` (the masks and EliteKV's order as a score), `src/analysis/calibration.py` (the per-head table and KV-head aggregation of 01's calibration, the text windows), `src/analysis/counterfactual.py` (the sink threshold), `src/edits/partial_rope.py` (the edit and its check), `src/eval/text_tasks.py` (the text and task evaluation), `src/utils/stats.py` (BCa intervals, Spearman).

## Running
On a SLURM cluster (`src/remote/launch.py`): `psym-04-<model>`, one job per model after its `psym-01-<model>`, computes the planes and then the KV heads and layers, skipping the cases already done; it reads 01's calibration at `calibration_tokens` and `long_calibration_tokens`. Locally: `make summarize|figures|check E=04_planes`.

## Main config keys
- `models`: the seven base models; `calibration_experiment: 01_measure`, `calibration_tokens: 1024`, `long_calibration_tokens: [4096, 16000]`.
- Planes: `rules: [high, norm, mkl, diagonal, elitekv, bound, variance, variance_4096, variance_16000]` (fastest planes, MHA2MLA's norms, MHA2MLA-VLM's KL criterion, the direct effect, EliteKV's RoPElite order; the bound, the position-dependent variance and the same variance calibrated at 4096 and 16000 tokens), each keeping in every KV head the max(1, round(f F)) planes of largest score, `fractions: [0.0625, 0.125, 0.25]`; `fraction: 0.25` for the tasks and the planes with $sigma_(zeta,f)^2 < 1\/2$.
- KV heads: `head_rules: [variance, ph, high_variance, random_0, random_1, random_2]` (lowest summed position-dependent variance, lowest positional ratio taken as the largest over the non-sink query heads, the opposite control, three seeded draws), `head_fractions: [0.25, 0.5]`; the head plans at `fraction` are also scored on the tasks.
- Layers: `layer_rules: [variance, interleaved, first, last, random_0, random_1, random_2]` (lowest summed variance, every round(1 / f)-th layer, the first or last layers, three seeded draws), `layer_fraction: 0.25`; `single_layers: true`, each layer alone on the text at the shortest length.
- Evaluation: `text` (the second half of the WikiText-2 test set and Python standard-library source), `text_n: 16` windows per corpus and length, `text_tokens: [1024, 4096]`; `tasks: [retrieval, binding, niah]`, `task_n: {short: 32, long: 50}`, `lengths: [4000, 16000]`.
- Comparisons: `compared` (variance against norm, mkl, diagonal, elitekv, variance_4096 and variance_16000; paired 95% BCa intervals over windows of the difference of the increases of the loss, `n_boot: 10000`); `norm_slow: {rules: [norm, mkl, elitekv, variance, variance_4096, variance_16000], threshold: 0.5}` (the rules whose kept planes are counted, and the threshold on $sigma_(zeta,f)^2$).
- `check: {tokens: 96, tolerance: 1.0e-5}`; `chunk` (queries per chunk of the edited attention, per model).

## Summary (`results/summary.json`)
- `models.<model>`: `loss_increase` (the increase of the loss, edited minus intact NLL, nats per token, per `<corpus>_<tokens>`, fraction and rule); `intervals` (paired intervals of the pairs of `compared`); `tasks` (accuracy per task of the intact model and of every rule at `fraction`); `score_correlation.<rule>` (Spearman correlation between the plane scores of `variance` and of each other rule); `sigma_zeta2.<tokens>.<rule>.share_below` (at the shortest length per rule of `norm_slow`, at the other lengths for `norm` only, the share of the planes kept at `fraction` whose $sigma_(zeta,f)^2$ is below the threshold); `overlap` (the share of the planes `variance` keeps that `variance_4096` and `variance_16000` also keep); `seconds` (wall-clock cost of the closed-form parts and of EliteKV's order in the calibration pass at `calibration_tokens`, and their ratio); `rotation` (KV heads and layers: `heads` and `layers` with `loss_increase` and `intervals` of `variance` against each other rule, `layers.chosen.variance`, the layers `variance` chose; `tasks` per head and layer plan; `single_layers`, from the increase of the loss of each layer alone, its Spearman correlation with the layer's summed variance (`spearman_variance`) and, for `variance` and `last`, the sum of these increases over the layers the rule chose (`summed`); `score_correlation` over KV heads of the variance and $Ph$; `all_sink`).
- `norm_to_variance.<corpus>`: `ratio` (the increase of the loss of `norm` over that of `variance`, per model and length), `spearman_share_below_vs_log_ratio` (per length, across models, between `norm`'s `share_below` and the log of `ratio`), `shrinks` (models where the ratio shrinks from the shortest to the longest length), `models` (models counted).
- `meta`: the run metadata.

## Figures (labels in `docs/notes.typ`)
- `models.pdf`: `fig:planes-models`.
- `tasks.pdf`: `fig:planes-tasks`.
- `sigma_zeta2_planes.pdf`: `fig:planes-sigma-zeta2`.

## Notes
Config keys and keys of the raw results whose names differ from the text (existing results store them):
- `norm_slow` (config): the rules whose kept planes are counted and the threshold $1\/2$ on $sigma_(zeta,f)^2$; it does not mean slow planes, which in the text are those with $sigma_(zeta,f)^2 approx 0$.
- `kept` (raw, `<model>__selection` and the masks of `<model>__groups`): the planes that keep their rotation, the complement of $R$ in each KV head.
- `n_freqs` (raw, `<model>__groups`): $F$, the number of planes; `n_layers`: $L$; `head_ph`: the largest $Ph$ over the non-sink query heads of each KV head.
- `text` (raw, evaluations): per corpus and length, the loss of each window (mean next-token NLL, nats per token); the increase of the loss is computed from it.
