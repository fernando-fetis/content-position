#!/bin/bash
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
# The compute stage of one experiment on one GPU of a SLURM cluster (or the environment and model downloads, for the experiment `setup`). launch.py submits it with the options of the cluster's profile (partition, account, GPU, CPUs, memory, time); paths come from .env: WORK_DIR (virtual environments and pip cache) and MODEL_CACHE (the Hugging Face cache, on storage the compute node may read models from).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f .env ] && { set -a; . ./.env; set +a; }
WS=${WORK_DIR:-$HOME/.cache/$(basename "$SLURM_SUBMIT_DIR")}
VENV=$WS/venv-$(sha256sum requirements.txt | cut -c1-12)
# A cache used for the first time is empty: setup or jobs with HF_ONLINE=1 fill it.
NODE_HF=${MODEL_CACHE:-$WS/hf}
mkdir -p "$NODE_HF" "$WS/pip"
exec 9>"$WS/venv.lock"
flock 9
if ! "$VENV/bin/python" -c "import torch, transformers" 2>/dev/null; then
  rm -rf "$VENV"
  python3 -m venv --without-pip "$VENV"
  curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python" - -q
  PIP_CACHE_DIR="$WS/pip" "$VENV/bin/python" -m pip install -q -r requirements.txt
fi
flock -u 9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME="$NODE_HF" PYTHONUNBUFFERED=1
EXPERIMENT=$1
shift
[ "$EXPERIMENT" = setup ] || "$VENV/bin/python" -c "import sys, torch; sys.exit(None if torch.cuda.is_available() else f'torch {torch.__version__} sees no CUDA device on {torch.version.cuda}; compare nvidia-smi with the wheel in requirements.txt')"
if [ "$EXPERIMENT" = setup ]; then
  df -h "$WS" .
  "$VENV/bin/python" src/remote/prefetch.py "$@"
else
  OFFLINE=1; [ -n "${HF_ONLINE:-}" ] && OFFLINE=0
  HF_HUB_OFFLINE=$OFFLINE "$VENV/bin/python" src/run.py "$EXPERIMENT" compute ${FORCE:+--force} "$@"
fi
mkdir -p logs/done && touch "logs/done/$SLURM_JOB_NAME"
