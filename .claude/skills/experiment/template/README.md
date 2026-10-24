# NN_name

<One paragraph: what the experiment measures (citing by label the definitions and statements it computes), on which models and data, and which sections of docs/ use it. No results.>

Importance: <high | medium | low>.

## Sources
- Definitions: <the .typ file(s), with the labels of the definitions and statements the experiment computes, e.g. `def:...`, `prop:...`>.
- Methodology: <the .typ file and section label where the design of the experiment is written>.
- Results: <the .typ file and section label where its results are written (a note in .temp/ while they are evaluated, a document of `docs/` once kept)>.

## Files
- `run.py`: the stages `compute`, `summarize`, `figures`, `check`.
- `figures.py`: the figures the document uses.
- `config.yaml`: every parameter; the `smoke:` block runs tiny-llama on the CPU.

## Running
<The cluster jobs (names in src/remote/launch.py) and the local stages: `make summarize|figures|check E=NN_name`.>

## Main config keys
<The keys that set the design, with their values.>

## Summary (`results/summary.json`)
<What each top-level key holds, without values.>

## Figures
- `<name>.pdf`: `<fig:label>` (the label of the figure that includes it in `docs/`).
