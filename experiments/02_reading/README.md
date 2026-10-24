# 02_reading

Heads at the three vertices of the simplex in one-layer RoPE transformers (`prop:three-pure-heads`): three synthetic tasks whose answer is fixed by a position, a content or their interaction; heads built by hand as in `prop:three-pure-heads`, single heads trained on each task, and three-head models trained on the mixture, measured with the parts of `def:content-positional-and-interaction-parts` at the query and tested with edits that keep only some parts in every head, as in the causal test of `sec:stability`.

Importance: medium.

## Sources
- Definitions: `docs/notes.typ` (`def:content-positional-and-interaction-parts`, `def:behavior-and-positional-ratio`, `prop:three-pure-heads`).
- Methodology: `docs/notes.typ` (`sec:vertices`; the part edits as in `sec:stability`).
- Results: `docs/notes.typ` (`sec:vertices`).

## Files
- `run.py`: the stages `compute` (hand-built heads, five seeds of single and three-head models, their measurements and edits), `summarize`, `figures`, `check` (the edit that keeps every part against the intact logits, and the hand-built heads solving their tasks).
- `toy.py`: the tasks, the model (embedding, one attention layer with RoPE, unembedding; no normalization or MLP), the hand-built heads, training, the measurement of the parts at the query, and the part edits (`EDITS`).
- `figures.py`: the figure of the document named under Sources.
- `config.yaml`: every parameter; `smoke:` for a quick CPU run.

## Running
One cluster job (`psym-02-toy`), or any machine with a GPU. `make summarize|figures|check E=02_reading`.

## Main config keys
`toy_model` (32 tokens: first token, 30 context positions, query; 16 symbols, 40 keys; width 128, head dimension 32, 16 planes, base $10^4$), `hand` (logit and output scales), `train` (Adam 0.003 with cosine decay, 30000 steps of 512), `seeds` (five), `multi_heads` (3), `solved` (0.9), `n_eval` (1000 held-out examples per task), `check_tolerance` ($10^{-4}$).

## Summary (`results/summary.json`, key `toy`)
- `hand.<task>`: behavior and accuracy of the hand-built head.
- `trained.<task>`: per seed, the final held-out accuracy, `solved` and the behavior of the single head.
- `multi.<task>`: per seed, the behavior of the reader (the head with the most attention on the answer; `reader_behavior`); the mean attention on the answer of the reader and of the head with the second most (`reader_attention`, `second_attention`); the accuracy under each edit, averaged over seeds (`edits_mean`).

## Figures (labels in `docs/notes.typ`)
- `toy_overview.pdf`: `fig:toy-exp`.

## Notes
- `att_answer` (under `measure` in every raw result, per task in `toy__multi_<seed>`): the attention of the query (the last token) on the answer's position, per example, layer and head (examples × layers × heads); the mean attention on the answer of the summary is its mean over examples.
- The positional and interaction hand-built heads repeat the single-plane designs of `prop:three-pure-heads` over the 16 RoPE planes, since this RoPE has no plane of period exactly 30; the symbolic one, with no frozen plane available, gives each key a random unit vector on the slowest half of the planes, shared by its query and its pair tokens. In the interaction task the other numbers of a list are drawn uniformly among those different from their position, so a list is not a permutation; hence the symbolic and interaction heads are near their vertices but not on them (the positional one is on its vertex). The hand-built heads use one-hot embeddings (width the vocabulary size), not the width of the trained models.
- The accuracy of a trained single head is the last one of its training log, on held-out examples of seed 10000 + seed; its parts and the hand-built and three-head measurements and edits use one fixed held-out set per task (seed 20000 + task index).
