---
name: experiment
description: How an experiment of this project is organized, implemented, documented and verified (experiments/NN_name/): folder, stages, config, results and summary, figures, README. Use when creating, changing, cleaning, running locally or documenting an experiment.
---

# Experiments

An experiment measures something the document in `docs/` needs. Its results are written into the texts named in its README's Sources (a note in `.temp/` while they are evaluated, a document of `docs/` once kept; `writing` skill); the experiment itself holds code, config, summary, figures and a short README, never a report.

## Naming and order
`experiments/NN_name/`, two digits in natural order (from the shared measurement to the uses); dependencies point to lower numbers, and a config key names the source (e.g. `calibration_experiment: <NN_name>`). Renumbering moves every reference: config keys, `src/remote/launch.py`, READMEs, `README.md`.

## A new experiment
Copy `template/` (next to this file: `run.py`, `figures.py`, `config.yaml`, `README.md`) to `experiments/NN_name/`, fill it in, add its jobs to `src/remote/launch.py`, and run its smoke stages before any cluster job.

## Folder (final version only)
```
experiments/NN_name/
  run.py        the stages
  config.yaml   every parameter, and a smoke: block
  README.md     what the experiment measures and how (below)
  figures.py    every figure the document uses, and only those
  results/      raw results per case (*.json + *.npz, untracked) and summary.json + summary.npz (tracked)
  figures/      the PDFs of figures.py (tracked)
  *.py          helpers used only by this experiment, few and named by what they hold (e.g. planes.py)
```
No one-off scripts, tests, inspection or merge scripts, backups or stale files: scratch work goes to `.temp/`. Shared code goes to `src/` (`README.md` lists every module); code used by one experiment stays in its folder. A module moves to `src/` when a second experiment needs it, and back into the experiment's folder when only one still imports it.

## Stages (`python src/run.py NN_name <stage> [--force] [--smoke] [--key value ...]`, or `make <stage> E=NN_name`)
- `compute(cfg, exp)`: loads models one at a time (`src.core.models.for_each_model`) and writes one raw result per case with `save(result_path(exp, model, case), lambda: ...)`. A finished case is skipped, so a requeued job resumes, and a new variant (a new rule, task or arm) is a new case that runs alone without recomputing the others: design every experiment so. It reads another experiment's raw results only through the config key that names it. The parts come from `src/analysis/counterfactual.py` (closed forms) inside `src.core.models.stream_qk`; forward passes that need no gradients run under `@torch.no_grad()`.
- `summarize(cfg, exp)`: every derived number into `results/summary.json` (`write_summary`): variances averaged over prompts before any ratio, 95% BCa intervals (`src.utils.stats.bootstrap_mean`). It stores only what is used: a number or figure of `docs/`, or a quantity another experiment reads; an analysis nobody uses is removed with its code. No number of `docs/` exists outside a summary.
- `figures(cfg, exp)`: reads only the summary and draws, in `figures.py` (one `draw(cfg, s, out)`), the figures `docs/` includes, in the order they appear; a figure the document drops is deleted with its code (`figures` skill).
- `check(cfg, exp)`: identity checks (edits against the intact model, closed forms against explicit computations) that fail above their tolerance.
- `make smoke` runs every stage of every experiment with `--smoke` (tiny-llama on the CPU, outputs under `.temp/smoke/`); every experiment must pass it, including contexts too short for some analyses (guard empty selections).

## Configuration
Keys shared across experiments: `models` (names of `src/models.yaml`), `tasks` (the benchmark of `src/core/suite.py`) with `n` per category, `seed`, `device` (`auto`), `dtype` (`float32`), `gpus` (1), `n_boot`; per-model values as `{default: ..., <model>: ...}` (`per_model`). Keys that split a run into jobs (`SELECTION` in `src/utils/results.py`) may differ between result files; any other difference makes `summarize` refuse unless `--force`. Every key is read by the code; the `smoke:` block overrides only existing keys.

## Code
Python 3.12, modern type annotations (`X | None`, `list[int]`), single quotes, few comments, docstrings that say what is computed and in which units, no hard-wrapped lines, no abstractions the experiment does not need. Names follow the notation of `docs/` (e.g. `var_c`, `var_p`, `var_e`, `sigma_zeta2`, `w2` for $|tilde(w)^f|^2$), so that the code reads as the text; when the text renames a symbol, the code follows (config keys and keys of raw results, which existing results store, keep their name and are listed with their symbol in the README's Notes), and docstrings, configs and READMEs cite definitions, statements and sections by their Typst label in backticks (`prop:the-parts-of-a-plane`, `sec:stability`), never by number (numbers change whenever a statement is added) and never by naming the document; `.claude/skills/audit-repo/scripts/labels.py` checks that every cited label exists.

## README.md
A description only, in English, short and precise: never expected results, obtained results, numbers that are results, or analysis. Technical content is referenced, not restated: cite the definitions and statements the experiment computes by label instead of writing their formulas. Sections:
- one paragraph: what the experiment measures, on which models and data;
- `Importance: high | medium | low`: how much the claims of the document rest on it (it orders the status reports and the queue);
- `## Sources`: Definitions (the `.typ` files and the labels of what it computes), Methodology (the `.typ` file and section where its design is written) and Results (where its results are written: a note in `.temp/` while they are evaluated, a document of `docs/` once kept). The texts may be anywhere; this section is what the audits compare the code against, so keep it current when a text moves;
- `## Files`: one line per file;
- `## Running`: the cluster jobs and the local stages;
- `## Main config keys`: the keys that set the design, with their values;
- `## Summary`: what each top-level key holds, without values;
- `## Figures`: each file with the label of the figure that includes it (`fig:...`; never its number, which changes), or when it will be included;
- `## Notes` (optional): descriptive only (what is not stored, how something is built).

## Removing an experiment
An experiment that did not work, or that the document no longer uses, is removed with everything only it used, so nothing is left orphaned:
- first, its record in the notes of `.temp/others/` (the idea, what ran, the results with numbers, why it was removed and what would revive it, and the removed text of the document if it had any);
- its folder (code, config, README, summary, figures; raw results are kept outside the repository, never deleted without a copy);
- its jobs in `src/remote/launch.py` (and the pilot list), and its rows in any dependency of another experiment (an experiment that reads it must be changed or removed first);
- the shared code in `src/` that only it imported (find the importers of each module), the models of `src/models.yaml` only it uses, the files of `data/` only it reads, and the entries of `docs/references.bib` cited only for it;
- what the document says about it, on the user's instructions (sections, figures, numbers, statements developed only for it), and the labels other files cite from those parts;
- then run `.claude/skills/audit-repo/scripts/labels.py`, `make smoke` and `make docs`, and update `todo.md`. The other experiments keep their numbers: gaps are fine.

## Verifying a change to an experiment
Check that every kept key of its summary has the same value as before and that its figures are byte-identical unless the change was meant to alter them (`.claude/skills/audit-repo/scripts/reproduce.py <NN_name>`), then rebuild them (`make summarize E=<NN_name>` and `.venv/bin/python src/run.py <NN_name> figures`); run its smoke stages; compile the documents (`make docs`). Before its results move into a main document, run the `audit-experiment` skill on it.
