#!/bin/bash
# Runs the smoke stages (compute --force, summarize, figures, check, each with --smoke: tiny-llama on the CPU, outputs under .temp/smoke/) of the given experiments, or of all, one at a time, and prints one line per experiment. Logs in .temp/audit/smoke_<exp>.log.
# Usage: .claude/skills/audit-repo/scripts/smoke.sh [<NN_name> ...]
cd "$(dirname "$0")/../../../.." || exit 1
mkdir -p .temp/audit
exps=("$@"); [ ${#exps[@]} -eq 0 ] && exps=($(ls -d experiments/[0-9][0-9]_* | xargs -n1 basename))
status=0
for e in "${exps[@]}"; do
  log=.temp/audit/smoke_$e.log; : > "$log"
  ok=1
  for stage in compute summarize figures check; do
    grep -q "^def $stage" "experiments/$e/run.py" || continue
    extra=(); [ $stage = compute ] && extra=(--force)
    .venv/bin/python src/run.py "$e" $stage --smoke "${extra[@]}" >> "$log" 2>&1 || { echo "$e: FAIL at $stage (see $log)"; ok=0; status=1; break; }
  done
  [ $ok = 1 ] && echo "$e: ok"
done
exit $status
