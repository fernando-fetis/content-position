---
name: remote
description: Running this project's experiments off the local machine (any GPU machine, a SLURM cluster, or Modal for small pilots), monitoring them, reporting their status, and bringing their results back to summarize locally; the reference for the CENIA cluster is in cenia.md. Use when launching, checking, debugging or collecting experiments, when asked for their status, or for anything about a cluster or Modal.
---

# Remote compute

Nothing heavy runs on the local machine: only summaries, figures, documents and the smoke tests (a tiny model on the CPU). The code does not depend on where it runs: `src/run.py <exp> compute` runs an experiment on any machine with a GPU and the environment of `requirements.txt`, and the launchers below only submit that command.

## Where to run
- **A machine with a GPU** (a workstation, a cloud instance): clone, set up the environment and `.env` as in `README.md`, then `python src/run.py <exp> compute [--models "[...]"]`. Finished cases are skipped, so a run that stops resumes; bring the raw results to where the summaries are built (`results/` of the experiment).
- **A SLURM cluster**: `src/remote/slurm.sbatch` runs the compute stage on one GPU (it builds the environment under `WORK_DIR`, reads models from `MODEL_CACHE`, both from `.env`), and `src/remote/launch.py` submits the job table `JOBS` with the options of the cluster's profile `src/remote/slurm.ini` (untracked; template `slurm.example.ini`). A new cluster needs only its profile and the two paths in `.env`. CENIA, the cluster used so far, is described in `cenia.md`, with its profile in `cenia/slurm.ini`.
- **Modal** (paid): only for small or urgent pilots, with the user's approval and a cost estimate, after checking that the cluster cannot do the same job in a similar total time; never a campaign.
- When unsure, state the estimated runtime and the choice in one line before launching.

## Experiments: status, results, reporting
- **Status**, whenever asked: `.venv/bin/python .claude/skills/remote/scripts/status.py [--host <ssh alias>]` lists every experiment with its importance (the `Importance` line of its README), the jobs done, running, queued, failed and not submitted, and a rough time left; add what blocks a job and what was done about it, and ask the user for the importance of an experiment without one.
- **When a job or an experiment finishes**, bring its results back and summarize them without waiting to be asked: `.claude/skills/remote/scripts/fetch.sh <NN_name> [host]` rsyncs the raw results (never the summary), checks the file counts on both sides, and runs `summarize` and `figures` locally (move stale local raw results aside first). It refuses results of different configurations, apart from the keys that split jobs (`SELECTION` in `src/utils/results.py`); `--force` only when the difference cannot change results, and say so.
- **Reporting a result**: model by model, with its intervals; say plainly whether it is good or bad news and whether it is main or secondary; then commit the summary and figures.
- Raw results are not in git: after every campaign keep a second copy outside the local machine, and never delete raw results anywhere without a copy.

## SLURM campaigns
- `launch.py`: jobs named `psym-<NN>-<model>[-<part>]`, the setup job first, `afterok` dependencies on it and on the results each job reads (`--kill-on-invalid-dep=yes`), a time limit and a nice per job. On the login node: `setsid nohup python3 src/remote/launch.py --watch > logs/launch.log 2>&1 < /dev/null &`; `--dry-run` prints the commands and the GPU-hours, `--only a,b` submits some jobs, `--only pilot` the pilot. It skips queued jobs and jobs with a done marker (`logs/done/<job>`), waits while the submit limit is reached, fills free slots by priority (lowest nice first), resubmits timeouts with 24 h and the dependents SLURM cancelled, and reports a FAILED job without resubmitting it: fix the cause, then resubmit it by hand with `sbatch` under the same name, time and nice (and the profile's options if the checkout's `slurm.sbatch` lacks them).
- Time limits close to the expected runtime (about twice the pilot's): a scheduler that holds GPUs for other users' jobs only backfills jobs that end before those reservations, so a 24 h job can wait indefinitely while shorter jobs start. A job that times out resumes from its saved cases.
- Priority among our own jobs: `scontrol update jobid=<id> nice=<n>` (it subtracts n). Age accumulates while a job waits, so an old job can outrank a newer one with a smaller nice: to put an experiment first, raise the nice of the others well above (hundreds to thousands).
- While the submit limit is reached, nothing new enters the queue, timeouts included: to let an important job in, ask the user before cancelling a pending job of lower importance (the watcher resubmits it later).
- Never cancel, stop, hold or move a job (ours or anyone's) without asking the user; changing the nice of our own jobs or lowering their time limit is fine.
- A queued job reads the code when it starts, not when it is submitted: commit and push locally, then `git pull` on the cluster only when no queued job depends on files the pull would change, move or rename (docstring-only changes are safe); otherwise the change waits.
- Helper processes on the login node are matched precisely when killed (`pgrep -f "^bash logs/<script>"`): `pkill -f <pattern>` also matches and kills its own ssh session.
- Checking: `squeue --me`; history with `sacct -S <date> -u $USER -X -o JobName%40,State,Elapsed` (without `-S`, only today); logs in `logs/<job>_<id>.out`. A model that fails to load in the first minutes, or "couldn't find them in the cached files", means its weights are missing from `MODEL_CACHE` (jobs run offline except `setup` and jobs with `HF_ONLINE=1`). A forward pass that does not need gradients runs under `@torch.no_grad()`: without it, long windows keep every activation and run out of memory.

## Modal
- Setup (once per user): `.venv/bin/pip install -r src/remote/requirements.txt` (the client), then `modal token new --profile <name>` (credentials in `~/.modal.toml`; never print or commit token values). The image is built from the root `requirements.txt`, and `.env` is passed as a secret.
- `src/remote/modal_app.py`: `main` runs the compute stage of an experiment on a GPU and returns its results (`MODAL_PROFILE=<name> .venv/bin/modal run src/remote/modal_app.py::main --experiment <exp> --gpu A10G --args "..."`, results in `.temp/pilot/<exp>/results/`), and `download --models a,b` fills the shared model cache (the volume named in `modal_app.py`); `modal run --detach` keeps a run alive after the local session ends.
