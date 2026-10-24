#!/bin/bash
# Brings an experiment's raw results back from the cluster, checks that the file counts match on both sides, and computes its summary and figures locally.
# Usage: .claude/skills/remote/scripts/fetch.sh <NN_name> [host] [remote repo path]   (defaults: cenia, ~/<repo>)
set -euo pipefail
EXP=$1
HOST=${2:-cenia}
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
REPO=${3:-$(basename "$ROOT")}
cd "$ROOT"
rsync -a --exclude summary.json --exclude summary.npz "$HOST:$REPO/experiments/$EXP/results/" "experiments/$EXP/results/"
remote=$(ssh -o ConnectTimeout=20 "$HOST" "ls $REPO/experiments/$EXP/results | grep -v '^summary\.' | wc -l" < /dev/null)
local=$(ls "experiments/$EXP/results" | grep -v '^summary\.' | wc -l)
echo "raw files: $remote on $HOST, $local here"
[ "$local" -ge "$remote" ] || { echo "transfer incomplete"; exit 1; }
.venv/bin/python src/run.py "$EXP" summarize
.venv/bin/python src/run.py "$EXP" figures
git status --short "experiments/$EXP"
