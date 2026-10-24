# content-position

Content, position and interaction in RoPE attention: an exact decomposition of a head's counterfactual logit matrix into content, positional and interaction parts, its closed forms under rotary position embeddings, and what follows for pretrained models.

## How the project works

- **`docs/` is the main folder**: everything we want to keep is written there, in one Spanish Typst document, versioned with its PDF (`make docs` rebuilds it). Every number in it comes from an experiment's summary, and every figure from an experiment's `figures/` folder (real results) or from `docs/figures/` (synthetic ones).
- **New theory** starts on paper: photos or scans of handwritten statements, proof sketches or diagrams go in `.temp/`; Claude transcribes them into a Typst note and we iterate on it (correctness, literature, variants, simpler proofs, numerical checks) until it is ready to move into `docs/` or to become an experiment (skill `new-theory`).
- **Experiments** measure what the document needs; each one is a folder of `experiments/` with its own `README.md` (what it measures, how, its importance, and its sources: where its definitions, methodology and results are written). They run on a GPU machine or a SLURM cluster (Modal only for small pilots); summaries, figures and the document are built locally (skills `experiment` and `remote`).
- **Figures** share one style and palette and are always vector PDFs; only the figures the document uses exist (skill `figures`).
- **Writing** follows the skill `writing`: rigor, one meaning per symbol, complete proofs, experiments described so that they could be rerun.
- **References between files** use Typst labels: code, configs and READMEs cite a definition, statement or section by its label (`prop:the-parts-of-a-plane`), never by number, so adding or reordering statements breaks nothing.
- **Checking**: an experiment is audited against the texts its README names (skill `audit-experiment`: every quantity computed exactly as defined and described, every reported number from its summary), and the whole repository before it is shared (skill `audit-repo`). An experiment that did not work is removed with everything only it used (skill `experiment`).
- **Claude Code**: `.claude/` holds the project guide (`CLAUDE.md`) and the skills named above. `todo.md` (untracked) lists the pending work.

## Setup

- Tools: Python 3.12, git, [Typst](https://typst.app) (`typst compile`), and Poppler (`pdftoppm`, to look at figures).
- Environment: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt` (`requirements.txt` pins every package; torch from the CUDA 12.6 wheels on Linux).
- `.env` (never committed): `HF_TOKEN=hf_...`, a Hugging Face token whose account has accepted the licenses of the gated models in `src/models.yaml`.
- Where experiments run: any machine with a GPU runs `src/run.py` directly. On a SLURM cluster, copy `src/remote/slurm.example.ini` to `src/remote/slurm.ini` (partition, account, GPU and memory of that cluster; untracked) and add to `.env` the folder for environments and caches (`WORK_DIR`) and the Hugging Face cache the compute nodes read models from (`MODEL_CACHE`).
- Modal (optional, pilots only): `.venv/bin/pip install -r src/remote/requirements.txt` and `modal token new --profile <name>`.

## Folders

- `docs/`: the document and its PDF, its notation macros and layout (`lib.typ`), the bibliography and the synthetic figures.
- `experiments/`: one folder per experiment, each described by its own `README.md`.
- `src/`: code shared by the experiments.
  - `core/`: model loading, RoPE geometry, query/key capture during the forward pass (`stream_qk`) and attention override (`models.py`); the synthetic tasks, text corpora and long-context tasks (`tasks.py`); the benchmark, the inputs built from the tasks (`suite.py`).
  - `analysis/`: the counterfactual matrix in factored form and its parts per head and per plane, the positional ratio, the deviations of Urrutia et al. (`counterfactual.py`); the calibration passes on WikiText-2: the parts per query, plane scores, EliteKV's order (`calibration.py`); the plane scores of every rule (`plane_scores.py`).
  - `edits/`: exact partial RoPE and its trainable version (`partial_rope.py`).
  - `eval/`: the text and task evaluation shared by the experiments that edit a model (`text_tasks.py`).
  - `utils/`: results I/O with run metadata (`results.py`), statistics and BCa intervals (`stats.py`), the figure style (`plots.py`).
  - `__init__.py`: `ROOT`, `RUN` (run metadata), `registry()`; `models.yaml`: the registry of every model used (Hugging Face id and pinned revision; `tiny-llama`, random weights, for local tests); `run.py`: runs one stage of one experiment.
  - `remote/`: SLURM clusters (`slurm.sbatch`, one stage on one GPU; `launch.py`, the job table, submitted with a cluster profile from `slurm.example.ini`; `prefetch.py`, model downloads) and Modal (`modal_app.py`; `requirements.txt` is the Modal client to install locally, the image is built from the root `requirements.txt`).
- `data/`: the input texts: WikiText-2 (`wikitext_test.txt`, the test split; `wikitext.txt`, passages) and Shakespeare passages (`shakespeare.txt`) for haystacks, and twenty whole modules of CPython 3.12 (`stdlib.txt`, PSF License) for code. Every input read from disk is here; the prompts of every task are generated from these texts by code with fixed seeds, and the models and the fine-tuning text (a FineWeb-Edu shard) are downloaded from Hugging Face at the revisions pinned in `src/models.yaml` and the configs.
- `Makefile`: every common command (`make help`). `.claude/`: project guide and skills. `.temp/`: scratch space, untracked.

## Running

```
make compute E=<NN_name> ARGS='--models "[<model>]"'
make summarize E=<NN_name>
make figures docs
```

- `compute` skips finished cases, so interrupted runs resume; `--key value` overrides a key of `config.yaml`; `--force` recomputes. On a SLURM cluster, `python src/remote/launch.py [--only NAME,...] [--dry-run] [--watch]` on the login node submits the job table with the options of `src/remote/slurm.ini` (skill `remote`).
- `make smoke`: every stage of every experiment on `tiny-llama` on the CPU (a few minutes), with the `smoke:` block of each config and outputs under `.temp/smoke/`.
- `check`: each experiment's identity checks (edits against the intact model, closed forms against explicit computations); it fails on any deviation above its tolerance.
