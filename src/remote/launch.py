'''Submits the experiments to a SLURM cluster through src/remote/slurm.sbatch, with the sbatch options of the cluster's profile (src/remote/slurm.ini, from slurm.example.ini): one job per entry of JOBS, after the setup job and the jobs whose results it reads (afterok). Queued or finished jobs (logs/done/psym-<name>) are skipped, so rerunning submits only what is missing; FORCE=1 resubmits the wanted jobs (every job without --only) once, and they recompute their cases; their dependencies outside --only stay as they are. It waits while the QOS submit limit is reached, so run it with nohup on the login node, from the checkout root: python src/remote/launch.py [--only pilot|NAME,...] [--dry-run] [--watch].'''
from __future__ import annotations

import argparse
import configparser
import datetime
import getpass
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'src' / 'remote' / 'slurm.ini'  # the cluster's sbatch options, untracked (template: slurm.example.ini)
START = time.time()  # under FORCE, a wanted job is finished only when its done marker is newer than this


@dataclass
class Job:
    name: str
    args: list[str]
    hours: int
    after: list[str] = field(default_factory=list)
    gpu: bool = True
    nice: int = 0  # SLURM ranks our pending jobs by how long they have been eligible, not by submission: a positive nice lets later jobs that matter more go first
    env: dict[str, str] = field(default_factory=dict)  # HF_ONLINE=1: the compute stage may download


def models(names: list[str]) -> list[str]:
    return ['--models', '[' + ', '.join(names) + ']']


MAIN = ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'qwen2.5-7b']
CROSS = ['llama-3.2-1b-base', 'llama-3.2-3b-base', 'llama-3.1-8b-base']
BASE = MAIN + CROSS  # the seven base models of 01 and 04
SIZE = {'qwen2.5-0.5b': 0, 'qwen2.5-1.5b': 0, 'llama-3.2-1b-base': 0, 'qwen2.5-3b': 1, 'llama-3.2-3b-base': 1, 'qwen2.5-7b': 2, 'llama-3.1-8b-base': 2}  # small, 3B, 7B and 8B
HOURS = {'01': (6, 12, 24), '03': (4, 6, 12), '04': (6, 12, 24)}  # time limit by model size: about twice the times of the pilot on the small models, scaled by size; a job that times out is resubmitted with 24 hours by --watch and resumes
hours = lambda key, m: HOURS[key][SIZE[m]]
CAUSAL = ['qwen2.5-1.5b', 'qwen2.5-3b', 'qwen2.5-7b', 'llama-3.2-3b-base', 'llama-3.1-8b-base']  # 03's models
FINETUNE = {'qwen2.5-0.5b': ['intact', 'variance', 'norm', 'mkl', 'high'], 'qwen2.5-1.5b': ['variance_16000', 'variance_4096', 'variance', 'norm', 'intact']}  # 05's runs
LARGE_FIRST = ['llama-3.1-8b-base', 'qwen2.5-7b', 'llama-3.2-3b-base', 'qwen2.5-3b', 'qwen2.5-1.5b', 'llama-3.2-1b-base', 'qwen2.5-0.5b']  # base models by size


# Priority: lower nice first, so free submission slots and GPUs go to the jobs the others depend on and to the most important experiments; within an experiment, larger models first, so that a campaign does not end with a long job. A job held by its dependencies does not block later ones.
JOBS = [
    Job('setup', ['setup', *BASE], 3, gpu=False),  # downloads and environment: no GPU, so it does not wait for one; the short limit lets the scheduler fit it in
    *[Job(f'01-{m}', ['01_measure', *models([m])], hours('01', m)) for m in LARGE_FIRST],
    Job('04-qwen2.5-1.5b', ['04_planes', *models(['qwen2.5-1.5b'])], hours('04', 'qwen2.5-1.5b'), ['01-qwen2.5-1.5b']),
    *[Job(f'05-qwen2.5-1.5b-{r}', ['05_finetune', *models(['qwen2.5-1.5b']), '--rule', r], 8, ['04-qwen2.5-1.5b'], env={'HF_ONLINE': '1'}) for r in FINETUNE['qwen2.5-1.5b']],  # one fine-tuning run per job; the FineWeb-Edu shard is downloaded
    *[Job(f'04-{m}', ['04_planes', *models([m])], hours('04', m), [f'01-{m}']) for m in LARGE_FIRST if m != 'qwen2.5-1.5b'],
    *[Job(f'05-qwen2.5-0.5b-{r}', ['05_finetune', *models(['qwen2.5-0.5b']), '--rule', r], 4, ['04-qwen2.5-0.5b'], env={'HF_ONLINE': '1'}) for r in FINETUNE['qwen2.5-0.5b']],
    *[Job(f'03-{m}', ['03_causal', *models([m])], hours('03', m), [f'01-{m}']) for m in LARGE_FIRST if m in CAUSAL],
    Job('02-toy', ['02_reading'], 3),  # hand-built and trained toy heads
]

# --only pilot: every experiment on Qwen2.5-0.5B (03 on Qwen2.5-1.5B, with the 01 job it reads; 05's intact and variance runs).
PILOT = ['setup', '01-qwen2.5-0.5b', '01-qwen2.5-1.5b', '02-toy', '03-qwen2.5-1.5b', '04-qwen2.5-0.5b', '05-qwen2.5-0.5b-intact', '05-qwen2.5-0.5b-variance']


CLUSTER: dict = {'gpu': [], 'cpu': [], 'max_queued': 32}  # filled by load_profile


def load_profile(path: Path) -> None:
    '''Reads the cluster profile (INI): the sbatch options of [all] and of [gpu] or [cpu] become `--key=value` flags of every job of that kind, and [launch] max_queued is the number of submitted jobs the cluster allows per user.'''
    if not path.exists():
        raise SystemExit(f'no cluster profile at {path}: copy src/remote/slurm.example.ini there and fill it in')
    ini = configparser.ConfigParser()
    ini.read(path)
    flags = lambda section: [f'--{k}={v}' for k, v in ini[section].items()] if ini.has_section(section) else []
    CLUSTER.update(gpu=flags('all') + flags('gpu'), cpu=flags('all') + flags('cpu'), max_queued=ini.getint('launch', 'max_queued', fallback=32))


def queued() -> dict[str, str]:
    out = subprocess.run(['squeue', '-h', '--me', '-o', '%j %i'], capture_output=True, text=True, check=True).stdout.split()
    return dict(zip(out[::2], out[1::2]))


def last_states() -> dict[str, str]:
    '''State of the latest run of every job of the last 30 days, by job name.'''
    start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
    out = subprocess.run(['sacct', '-n', '-X', '-P', '-u', getpass.getuser(), '-S', start, '-o', 'JobName,State'], capture_output=True, text=True, check=True).stdout.splitlines()
    return dict(line.split('|')[:2] for line in out if line)


def submit(cmd: list[str]) -> str:
    '''Runs sbatch and returns the job id, retrying every 120 s while it refuses with QOSMaxSubmitJobPerUserLimit (squeue may lag behind the cap).'''
    while True:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip().split(';')[0]
        if 'QOSMaxSubmitJobPerUserLimit' not in r.stderr + r.stdout:
            raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
        print('submit limit reached, retrying in 120 s', flush=True)
        time.sleep(120)


def submit_missing(wanted: set[str], dry_run: bool) -> int:
    '''One pass over JOBS: submits every wanted job neither queued nor finished whose dependencies are queued or finished; a job that timed out gets 24 hours (it resumes from its saved cases), one that failed is reported, not resubmitted. Returns the number of wanted jobs not finished.'''
    setup = JOBS[0].name  # every other job depends on it
    done = ROOT / 'logs' / 'done'
    def finished(n: str) -> bool:
        marker = done / f'psym-{n}'
        forced = os.environ.get('FORCE') and (not wanted or n in wanted)
        return marker.exists() and (not forced or marker.stat().st_mtime > START)

    live = lambda: {k.removeprefix('psym-'): v for k, v in queued().items()}
    ids, recent = ({} if dry_run else live()), {}  # recent: jobs submitted in this pass, with their time (squeue may not list them yet)
    states = {} if dry_run else last_states()
    left = 0
    for job in sorted(JOBS, key=lambda j: j.nice):  # by priority, so that a freed submission slot goes to the job that matters most
        if wanted and job.name not in wanted:
            continue
        name = f'psym-{job.name}'
        if finished(job.name):
            continue
        left += 1
        if job.name in ids:
            continue
        state = states.get(name, '')
        if state.startswith(('FAILED', 'OUT_OF_MEMORY')):
            print(f'{name} {state}: not resubmitted, check its log', flush=True)
            continue
        if not dry_run:
            while len(queued()) >= CLUSTER['max_queued']:
                time.sleep(60)
            ids = {**{k: v for k, (v, t) in recent.items() if time.time() - t < 300}, **live()}  # refreshed after the wait: a dependency may have finished or failed meanwhile
        deps = [d for d in ([] if job.name == setup else [setup, *job.after]) if not finished(d)]
        if waiting := [d for d in deps if d not in ids]:
            print(f'{name} waits for {", ".join(waiting)}', flush=True)
            continue
        after = [ids[d] for d in deps]
        hours = 24 if state == 'TIMEOUT' else job.hours
        cmd = ['sbatch', '--parsable', f'--job-name={name}', f'--time={hours}:00:00', *([f'--nice={job.nice}'] if job.nice else []), *CLUSTER['gpu' if job.gpu else 'cpu'], *([f'--dependency=afterok:{":".join(after)}', '--kill-on-invalid-dep=yes'] if after else []), *(['--export=ALL,' + ','.join(f'{k}={v}' for k, v in job.env.items())] if job.env else []), 'src/remote/slurm.sbatch', *job.args]
        if dry_run:
            ids[job.name] = f'<{job.name}>'
            print(shlex.join(cmd))
            continue
        (ROOT / 'logs').mkdir(exist_ok=True)  # slurm.sbatch writes its output there
        ids[job.name] = submit(cmd)
        recent[job.name] = (ids[job.name], time.time())
        print(f'{name} {ids[job.name]}' + (' (24 h after a timeout)' if hours != job.hours else ''), flush=True)
    return left


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', default='', help='comma-separated job names (without the psym- prefix), or pilot')
    parser.add_argument('--dry-run', action='store_true', help='print the sbatch commands without submitting')
    parser.add_argument('--watch', action='store_true', help='repeat every 15 minutes until every wanted job has finished: resubmits timed-out jobs and the dependents SLURM cancelled')
    parser.add_argument('--profile', type=Path, default=PROFILE, help='the cluster profile: sbatch options for every job, for GPU jobs and for CPU jobs, and the submit limit')
    args = parser.parse_args()
    load_profile(args.profile)
    names = {j.name for j in JOBS}
    if missing := {d for j in JOBS for d in j.after} - names:
        raise SystemExit(f'dependencies on unknown jobs: {sorted(missing)}')
    wanted = {n for name in filter(None, args.only.split(',')) for n in (PILOT if name == 'pilot' else [name])}
    if args.dry_run:
        submit_missing(wanted, True)
        gpu = [j for j in JOBS if j.gpu and (not wanted or j.name in wanted)]
        print(f'{len(gpu)} GPU jobs, {sum(j.hours for j in gpu)} GPU-hours of limits')
        return
    while (left := submit_missing(wanted, False)) and args.watch:
        print(f'{datetime.datetime.now():%Y-%m-%d %H:%M} {left} jobs not finished', flush=True)
        time.sleep(900)


if __name__ == '__main__':
    main()
