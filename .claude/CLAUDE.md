# Project guide

Research project on content, position and interaction in RoPE attention: the exact decomposition of a head's counterfactual logit matrix, its theory under RoPE, the experiments around it and, once they have run, the paper (target ICML 2027). The prior work it builds on (positional and symbolic heads, the uses of RoPE's planes) is its lineage, not a competitor.

## Layout
- `docs/`: the main folder of the project, where everything we want to keep is written: the document (one Spanish Typst file, versioned with its PDF; each experiment's README names the sections that use it), `lib.typ` (layout and notation macros), `references.bib`, and `figures/` with the synthetic figures and `figures.py`, which draws them.
- `experiments/NN_name/`: one experiment per folder, on the benchmark of `src/core/suite.py`: code, config, summary, figures and a README that describes the experiment. Results live in `docs/`, never in the experiment's folder.
- `data/`: every input text of the experiments. `src/`: shared code (modules, the model registry `models.yaml`, the entry point `run.py`, `remote/` for the cluster and Modal). `Makefile`: every common command (`make help`). `README.md` (root): how the project works, the setup, how to run things, and one line per folder; it never details the contents of `docs/`, `experiments/` or `.temp/` (no list of documents, experiments or scratch files): each experiment is described by its own README, and the documents describe themselves.
- `todo.md` (untracked): the pending work; read it at the start of a session and summarize what matters. Add new pending work concisely; delete finished items, never mark them done.
- `.temp/` (untracked): scratch space (one-off scripts, backups before a refactor, transcriptions of handwritten notes, audit reports), safe to delete at any time and never a dependency of the repository, except `.temp/others/`, which the user keeps: notes on theory beyond the document, untried ideas, experiments that were removed with their results and lessons, and a draft of the paper.

## Skills
- `experiment`: how an experiment is organized, implemented, documented and verified.
- `figures`: how every figure is made and included.
- `writing`: how the document in `docs/` are written (rigor, notation, proofs, Typst style).
- `remote`: running experiments on a GPU machine, a SLURM cluster (CENIA so far) or Modal, reporting their status, bringing results back.
- `new-theory`: turning handwritten math dropped in `.temp/` into a transcription and iterating on it.
- `audit-experiment`: an exhaustive audit of one experiment against the texts named in its README's Sources (every quantity, both directions, every reported number).
- `audit-repo`: a full audit of the repository, coordinated by the main session with parallel subagents (it runs `audit-experiment` for each experiment), with fixes and a report in `.temp/`.

## Rules of this project
- Nothing heavy runs on the local machine: experiments run on a GPU machine or a cluster (Modal only for small pilots, with the user's approval); summaries, figures, documents and smoke tests run locally.
- Every number in `docs/` comes from an experiment's `results/summary.json`; report each result as it arrives (`remote` skill), and change the document only when the user asks (`writing` skill).
- Every figure is vector; only the figures the document uses exist (`figures` skill).
- Code: Python 3.12, modern type annotations, single quotes, few and precise comments, no hard-wrapped lines in any file (code, Markdown, Typst); names follow the notation of the documents, and definitions and statements are cited by their Typst label, never by number. What an experiment computes is exactly what its sources define and describe (skill `audit-experiment`).
- The code is not tied to one machine or cluster: experiments run with `src/run.py` anywhere, and what is specific to a cluster lives in its profile and in the `remote` skill. Code changes that would alter running or pending cluster jobs wait until those jobs start or end; the cluster checkout is updated with `git pull` only when no queued job depends on files the pull would change.
- Git: commit messages in one line in Spanish; commit and push after each finished block of work; never add AI attribution (no `Co-Authored-By`, no "Generated with" lines) to commits, PRs or files.
- When an instruction can be read in meaningfully different ways, ask before acting; when the intent is clear, proceed. Briefly justify decisions of approach, architecture or trade-off; for long tasks, give a one- or two-line update at natural milestones.
- Parallelize independent work (independent tool calls in one batch, subagents for self-contained subtasks); keep tightly coupled work in the main context. Subagent models: Sonnet for easy, mechanical tasks (searches, renames, checks of numbers against a summary, formatting); Opus for every other task, audits included; Fable only when an instruction asks for it explicitly.
