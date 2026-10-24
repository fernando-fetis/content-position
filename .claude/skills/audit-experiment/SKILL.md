---
name: audit-experiment
description: An exhaustive audit of one experiment against the texts it follows: every quantity it computes, analyzes or draws must be exactly what its definitions and methodology say, every choice of the code that changes how a result is read must be written in the methodology (implementation details stay in the code), and every reported number and figure must match the summary; the experiment must also follow the experiment, figures and writing skills. Use when an experiment is created or changed, before its results move into a main document, and from audit-repo for each experiment.
---

# Experiment audit

What an experiment computes must be exactly what its texts define and describe: no factor, normalization, set, threshold or aggregation that the texts do not state. The only accepted differences are numerical guards (an epsilon in a division, a clip against overflow) and they are listed. The texts are named in the `## Sources` section of the experiment's README: where its definitions are (labels in a `.typ` file), where its methodology is written, and where its results are reported (a document of `docs/` or a note in `.temp/`); the audit follows them wherever they are.

## 1. Inventory (one subagent per experiment, Opus, or the main session for a single one)
- Completeness first: which cases (model, set, task, arm) the config asks for, which have raw results here and which are still running or queued on the cluster (`remote` skill, status script). Results may be partial while a campaign runs: a missing case is not an error. It is listed in the report, and the audit checks that the results text says plainly what is pending, claims only what the finished cases support, and that no summary number or figure mixes in a case that is only partly done (a model with some of its arms).
- Read the experiment's README (Sources first), `config.yaml`, every `.py` of the experiment, the shared code it calls (`src/`), and the sources: the definitions and statements by label, the methodology, and the results section.
- List every quantity the results text reports or the figures draw (each number, each curve, each table cell), and every quantity the summary stores.

## 2. Text to code, one quantity at a time
For each quantity, write one row: its definition in the sources (label and formula), its code path (the raw computation in `compute` and `src/`, the aggregation in `summarize`, the drawing in `figures.py`), and a verdict. Check that they agree exactly in:
- the formula: constants and factors (1/2, 1/d_h, the 1/sqrt(d_h) of the logits), signs, real parts and moduli, centering, which variance or covariance;
- the data: tasks, prompts and their number, the query (last token or several), the context positions (first token and the query excluded or not), windows and their lengths, seeds;
- the sets: which heads (non-sink, the threshold on the first token, the sets of the methodology), which planes, which layers, which models; thresholds (on $P_h$, on $sigma_(zeta,f)^2$, on accuracy);
- index conventions (0-based in code against 1-based in the text) wherever an index is printed;
- the aggregation: means over prompts before any ratio, medians or means over which set, pooling over models, ranges over which models, intervals (bootstrap over what, BCa, level);
- the reported value: the rounding shown, the units, the sign convention of differences.

## 3. Code to text
List every choice of the code that can change a reported number (a filter, a threshold, a default, a resampling, a special case for a model, a skipped case). The methodology states those that change how a result is read (what is measured, on which data, how it is aggregated, a metric's limits); implementation details (tokenizer quirks, seeds, per-model exceptions that do not change a conclusion, exact positions) stay in the code and the README's Notes, never in the text. Report a missing choice only if a reader would misread a result without it.

## 4. Summary against use
Every quantity `summarize` computes and stores is used: by a number or claim of the results text, or by a figure (metadata such as counts and the run configuration excepted). List each top-level and nested key of `results/summary.json` with where it is used; a key nothing uses is removed from `summarize` (and the README's Summary), not only from the text, and a computation whose result is discarded (an interval computed and dropped, a quantity computed for a figure that no longer exists) is removed too. Conversely, a number the text or a figure needs is computed in `summarize`, never in `figures.py`, a scratch script or by hand. Before calling a key unused, search the results texts for the numbers it holds and the figures for what they draw.

## 5. Numbers and figures
Every reported number against the summary (model by model, with the rounding shown), every conclusion against the numbers and figures it cites (a claim about all models against one that deviates, a monotone claim against a curve that is not), every figure's caption against what the figure draws.

## 6. Conformity with the skills
The experiment follows the rules of its skills as in their final version; check each against the skill itself, not from memory:
- `experiment`: the folder holds only what that skill lists (no report, no one-off scripts, no stale raw results, no figures nothing includes); the stages, the configuration and the code follow it; its README has exactly the sections it lists, describes the experiment only (no expected or obtained results, no analysis), names its Sources and gives its importance; code used only by this experiment lives in its folder, and what it reads from `src/` is shared.
- Names: the code is the text written another way, so every quantity, index and set of the sources keeps its symbol's name in the code (the permutation $tau$ is `tau`, $Var(p_j)$ is `var_p`, $Ph$ is `ph`, a position of the context is `j` or `t` as in the text), in variables, functions, summary keys and figure labels; a name the code chose before the text fixed its symbol is renamed. Names that cannot change because results already store them (config keys and the keys of raw results) are listed with their symbol in the README's Notes.
- Hygiene: docstrings and comments precise and concise, citing statements by label; no deprecated or unused functions or config keys nothing reads (summary keys: section 4); nothing missing (everything a stage imports or reads exists).
- `figures`: every figure vector, drawn from the summary only, at its printed size, with the shared palette, sizes and model names (no literal colours, sizes or line weights), and every figure included in a document.
- `writing`: the experiment's text in its Sources (methodology and results) is written as that skill says: rerunnable from the text, every number from the summary, conclusions no stronger than the numbers, no file, folder, config key or experiment number named, figures with precise captions.

## 7. Act and report
- A discrepancy is fixed on the side that is wrong: the code when the text states what was meant (then rebuild the summary and figures and report every value that changes), the text when the code computes what was meant (the description in its methodology, the results in its results text). The documents of `docs/` are edited only on the user's instructions: propose such fixes instead of applying them, unless the user asked for the audit to fix them.
- Never change what an experiment computes while its jobs are queued or running on the cluster; note the fix instead.
- Report: the completeness table (case: done, partial, pending), the table of quantities (definition, code path, verdict), the gaps of the description, the mismatched numbers, every fix made and every value that changed, and the open points. When run from `audit-repo`, return the report to the main session; alone, write it to `.temp/audit-<NN_name>-<YYYY-MM-DD>.md`.
