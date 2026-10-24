# 01_measure

The shared measurement: for every head of the seven base models, the parts of the counterfactual matrix (`def:counterfactual-logit-matrix` and `def:content-positional-and-interaction-parts`) at the last query of every prompt of the 27 tasks, with the per-plane statistics of `prop:the-parts-of-a-plane` and the deviations of `def:deviations`; on top of them, the analyses that use measured heads (the planes and their aggregation, stability, readers, depth, roles, long contexts, and the comparisons with previous measures; see Sources). It also stores what 03 and 04 read: calibration passes on WikiText-2.

Importance: high.

## Sources
- Definitions: `docs/notes.typ` (`def:counterfactual-logit-matrix`, `def:content-positional-and-interaction-parts`, `def:behavior-and-positional-ratio`, `def:deviations`, `prop:the-parts-of-a-plane`, `cor:where-each-part-can-live`, `prop:phase-statistics`, `prop:pseudo-variance-and-pca`, `cor:slow-planes-and-long-contexts`, `prop:head-as-a-sum-of-planes`, `prop:the-positional-ratio-as-a-relative-distance`, `cor:ratio-in-long-contexts`, `prop:deviations-in-terms-of-the-parts`); the tasks, models and measurements in `sec:setup`.
- Methodology: `docs/notes.typ` (`sec:measurements`, and each analysis where it is reported: `sec:slow-fast`, `sec:plane-measures`, `sec:length`, `sec:head-from-planes`, `sec:ratio`, `sec:metric`, `sec:stability`, `sec:vertices`, `sec:long-contexts`, `sec:barbero`, `sec:urrutia`, `sec:chen`).
- Results: `docs/notes.typ`, in the same sections.

## Files
- `run.py`: the stages. `compute` (one result per model and case: the benchmark cases, the repeated random tokens of the induction-circuit roles, Chen and Yan's quantities, the calibration passes), `summarize`, `figures`, `check`.
- `analyses.py`: the questions per model answered from the raw results and case summaries (stability, interaction, depth, plane budgets, readers, KV groups, bounds, aggregation of the interaction, $\eta$, the length prediction) and the whole-head quantities pooled over models (`head_extras`).
- `planes.py`: the per-plane summaries (`plane_extras`): the phase statistics of `prop:phase-statistics`, the three-plane mosaic of `cor:where-each-part-can-live`, the plane fractions and ratios to the bounds, the budget of `cor:slow-planes-and-long-contexts`, the long-context limit of `cor:ratio-in-long-contexts`.
- `measure.py`: the measurement of one input (one forward pass): per query, layer and head the parts of the counterfactual matrix, the deviations of Urrutia et al., the attention on the first token, the answer and the distractor; per plane the statistics of `prop:the-parts-of-a-plane`; the head table and the split-half reliability.
- `roles.py`: the induction-circuit roles on repeated random tokens (previous-token and induction scores) and their summary.
- `chen.py`: Chen and Yan's quantities (pseudo-variance of the content factors, cosine between the weight rows of a plane).
- `checks.py`: identity checks (closed forms against explicit matrices, phases against `prop:phase-statistics`, the permutation means of `prop:deviations-in-terms-of-the-parts` over all 8! orders of a short context, EliteKV's removal matrices and greedy order).
- `figures.py`: the figures of the document named under Sources, in their order.
- `config.yaml`: every parameter; the `smoke:` block runs tiny-llama on the CPU.

## Running
On a SLURM cluster, one job per model (`psym-01-<model>`, `src/remote/launch.py`). Locally: `make summarize E=01_measure` (a few minutes), `make figures E=01_measure`, `make check E=01_measure`. Raw results (`results/<model>__<case>.*`) are not tracked.

## Main config keys
`models`, `tasks`, `n` (prompts per category: 128 short, 32 texts, 50 long), `max_tokens` (1024, texts), `lengths` (4000, 16000, 31000), `ratio_threshold` (0.8), `reader_fraction` (0.05), `read` (accuracy 0.9, margin 2), `n_split` (50), `repeated` (the induction circuit: 16 sequences of 64 random tokens, 8 queries, thresholds 0.5 and 0.3), `calibration` (the passes for 03 and 04), `figures` (main model Qwen2.5-7B and the cases drawn). Sink heads: mean attention on the first token above 0.8 (`src/analysis/counterfactual.py`).

## Summary (`results/summary.json` + `summary.npz`)
- `tasks.<case>`: task, category and target length of every case.
- `cases.<model>.<case>`: per head $P_h$ (`ph`) and the behavior (`heads`), `non_sink`, `length_mean`, `split_half` (over the heads non-sink in every case of the model), `interaction_bound` (per head $Var(\epsilon_{ij})$ and its two plane sums), and per plane the head means of the position-dependent variances, $|\tilde w^f|^2$, $\sigma_{w,f}^2$ and the period over the context (`plane`).
- `questions.<model>`: `stability`, `b3` (per case, the median interaction share $b_{h,3}$ and the share above 0.2), `depth` (WikiText-2), `planes` (per case, the content ratio to its bound in slow planes and the largest $\Phi^f$ and its largest excess over the interval of `cor:slow-planes-and-long-contexts`, `phi_max` and `phi_excess`), `reading` (read cases and the readers' gap), `groups` (KV-group $\eta^2$), `bounds` (largest violations of `cor:where-each-part-can-live`), `interaction_bound` (the median $\kappa_2$ by kind of input, `kappa2`), `decomposition` ($\eta$ over the short tasks, `eta`), `length_prediction` (the means of $\Delta_h$ and $\hat\Delta_h$, `delta` and `delta_hat`, their Spearman correlation, the error of the prediction at the shortest length).
- `urrutia.<model>.<case>`: per head $P_h$ (`ph`), the normalized measure and $V$ over $D_{pos} + D_{sym}$ (`bound_theirs`); the agreement of the first two (Spearman, share on opposite sides of 1/2) and the median of the third.
- `roles.<model>`: the number of previous-token heads; per input (repeated tokens, WikiText-2) and role, the median $P_h$; on WikiText-2 also the median $\nu_h$ and $\pi_h$ (`nu`, `pi`) and the profiles over planes.
- `chen.<model>`: the share of (non-sink head, plane) pairs with $\iota_{hf} > 0.9$ (`iota_share`), the median weight-row cosines, the correlations of $\iota_{hf}$ with them and of $P_h$ with the per-head mean cosine, and the deciles drawn.
- `agreement`: for the main model, its cases ordered as short tasks, texts and long tasks, and the Spearman correlation of $P_h$ between every pair of them over $\mathcal{H}^*$, the heads non-sink in every case.
- `planes_extra`: `theta` (per model, the frequencies $\theta_f$), `zones` (per model on WikiText-2 and the longest single needle: the phase moments, the number of fast planes and the median phase covariance between distinct fast planes), `zone_context` (per case of `zones`, the mean number of positions of the context over models and whether they all have the same), `bounds` (the three planes of the main model drawn: $\sigma_{\zeta,f}^2$, $|\tilde w^f|^2$ over the budget and each variance over its bound), `fractions` (per model `ratio_q`, the quartiles of $R_K^f$ in the fast, transition (period 2 to 13 times the context) and slow planes; pooled over models in bins of the period, the quartiles of $\psi_K^f$ and $R_K^f$), `budget` (per length, the shares of the budget and $\Phi^f$, `phi`), `limit` (per model and long case, per plane $P^f$, `pf`, and its limit), `limit_stats` (per length, the median of $|P^f - \text{limit}|$ over fast planes).
- `heads_extra`: per head the position-dependent variance and its plane sum (`cases`), $\kappa_0$ and $\kappa_1$ pooled by length group (`pooled.<group>.aggregation.kappa0`, `kappa1`), and per case the fraction of non-sink heads with $P_h$ above `ratio_threshold` (`geometry`).

## Figures (labels in `docs/notes.typ`)
- `plane_bounds_real.pdf`: `fig:part-bounds`.
- `phase_moments_real.pdf`: `fig:dispersion`.
- `plane_fractions_ci.pdf`: `fig:planes-exp`.
- `budget_share.pdf`: `fig:budget-exp`.
- `aggregation_real.pdf`: `fig:aggregation-exp` (the covariance of the phases of two planes, and the aggregation over planes).
- `interaction_real.pdf`: `fig:interaction-exp`.
- `simplex_main.pdf`: `fig:simplex-exp`.
- `stability.pdf`: `fig:stability-exp`.
- `long_contexts.pdf`: `fig:long-exp`.
- `fast_planes.pdf`: `fig:fast-planes-exp`.
- `measure.pdf`: `fig:measure-exp`.
- `chen_angle.pdf`: `fig:chen-exp`.

## Notes
- $\mu^f$ is not stored per plane, so the bound figure shows only the bound of the constant.
- The raw results keep names fixed before the text: `metric` is the margin $m(x)$ of `sec:tasks`, `plane_w2` is $|\tilde w^f|^2$ and `plane_sigma_w2` is $\sigma_{w,f}^2$.
