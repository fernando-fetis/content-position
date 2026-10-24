'''Status of every experiment on a SLURM cluster, from the local checkout: one ssh call reads the queue, the job history and the done markers of the remote checkout; the planned jobs come from the local src/remote/launch.py (JOBS). Prints, per experiment, its importance (the Importance line of its README), the jobs done, running, queued, failed and not submitted, and a rough time left (the median elapsed time of its finished jobs times the jobs left, divided by the jobs running, at least one).

Usage: .venv/bin/python .claude/skills/remote/scripts/status.py [--host cenia] [--repo ~/<repo>] [--days 30]'''
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src' / 'remote'))
from launch import JOBS  # noqa: E402


def seconds(t: str) -> float:
    '''SLURM elapsed time ([D-]HH:MM:SS or MM:SS) in seconds.'''
    days, _, clock = t.rpartition('-')
    parts = [int(x) for x in clock.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def importance(nn: str) -> str:
    for readme in ROOT.glob(f'experiments/{nn}_*/README.md'):
        if m := re.search(r'^Importance:\s*(\w+)', readme.read_text(), re.M):
            return m.group(1)
    return '?'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--host', default='cenia')
    p.add_argument('--repo', default=f'~/{ROOT.name}')
    p.add_argument('--days', type=int, default=30)
    a = p.parse_args()
    cmd = f"squeue -h --me -o '%j|%T|%M|%r'; echo '@@'; sacct -n -X -P -S now-{a.days}days -o JobName,State,Elapsed; echo '@@'; ls {a.repo}/logs/done"
    out = subprocess.run(['ssh', '-o', 'ConnectTimeout=20', a.host, cmd], capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout
    queue_txt, history_txt, done_txt = out.split('@@')
    queue = {l.split('|')[0].removeprefix('psym-'): l.split('|')[1:] for l in queue_txt.split('\n') if '|' in l}
    done = {l.strip().removeprefix('psym-') for l in done_txt.split() if l.strip()}
    last, elapsed = {}, {}
    for line in history_txt.split('\n'):
        if line.count('|') >= 2:
            name, state, el = line.split('|')[:3]
            name = name.removeprefix('psym-')
            last[name] = state
            if state == 'COMPLETED':
                elapsed.setdefault(name, seconds(el))
    by_exp: dict[str, list[str]] = {}
    for job in JOBS:
        by_exp.setdefault(job.name.split('-')[0], []).append(job.name)
    print(f'{"experiment":<12}{"importance":<12}{"done":>6}{"running":>9}{"queued":>8}{"failed":>8}{"not sub.":>10}   time left')
    for nn, names in by_exp.items():
        n_done = sum(n in done for n in names)
        running = [n for n in names if queue.get(n, [''])[0] == 'RUNNING']
        queued = [n for n in names if queue.get(n, [''])[0] == 'PENDING']
        failed = [n for n in names if n not in done and n not in queue and last.get(n, '').startswith(('FAILED', 'OUT_OF_MEMORY'))]
        left = len(names) - n_done
        not_sub = left - len(running) - len(queued) - len(failed)
        times = [elapsed[n] for n in names if n in elapsed]
        eta = '-' if not left else (str(timedelta(seconds=round(statistics.median(times) * left / max(1, len(running))))) if times else 'unknown (no finished job)')
        print(f'{nn:<12}{importance(nn):<12}{n_done:>6}{len(running):>9}{len(queued):>8}{len(failed):>8}{not_sub:>10}   {eta}')
    reasons: dict[str, list[str]] = {}
    for n, s in queue.items():
        if s[0] == 'PENDING' and s[2] not in ('Priority', 'Dependency', 'None'):
            reasons.setdefault(s[2], []).append(n)
    for reason, names in reasons.items():
        print(f'pending ({reason}): {len(names)} jobs, e.g. {names[0]}')


if __name__ == '__main__':
    main()
