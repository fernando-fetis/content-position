'''Checks that an experiment's summary and figures are reproduced by its current code from the existing raw results, without changing either: keeps a copy of summary.json, summary.npz and the figure PDFs, rebuilds the summary and compares every key and array with the copy (meta.time ignored, floats to a relative 1e-12), reruns the figures stage and lists the PDFs whose bytes changed, then writes the copies back. To keep a rebuilt summary, run `make summarize E=<NN_name>` and `.venv/bin/python src/run.py <NN_name> figures` afterwards. One experiment at a time; run it in the background for long summaries.

Usage: .venv/bin/python .claude/skills/audit-repo/scripts/reproduce.py <NN_name> [<NN_name> ...]'''
from __future__ import annotations

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def diff(a, b, p: str = '') -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in (([f'missing {p}/{k}'] if k not in b else [f'new {p}/{k}']) if (k not in a) != (k not in b) else diff(a[k], b[k], f'{p}/{k}'))]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (u, v) in enumerate(zip(a, b)) for d in diff(u, v, f'{p}[{i}]')]
    if isinstance(a, float) and isinstance(b, float) and (a == b or abs(a - b) <= 1e-12 * max(1.0, abs(a)) or (a != a and b != b)):
        return []
    return [] if a == b else [f'changed {p}']


def snapshot(exp: Path) -> dict[Path, bytes]:
    files = [exp / 'results' / 'summary.json', exp / 'results' / 'summary.npz', *sorted((exp / 'figures').glob('*.pdf'))]
    return {f: f.read_bytes() for f in files if f.exists()}


def check(name: str) -> bool:
    exp = ROOT / 'experiments' / name
    res = exp / 'results'
    before = snapshot(exp)
    if res / 'summary.json' not in before:
        print(f'{name}: no summary to compare with')
        return False
    run = lambda stage: subprocess.run([str(ROOT / '.venv' / 'bin' / 'python'), 'src/run.py', name, stage], cwd=ROOT, capture_output=True, text=True)
    problems: list[str] = []
    try:
        out = run('summarize')
        if out.returncode:
            print(f'{name}: summarize failed\n{out.stderr[-2000:]}')
            return False
        a, b = json.loads(before[res / 'summary.json']), json.loads((res / 'summary.json').read_text())
        a.get('meta', {}).pop('time', None), b.get('meta', {}).pop('time', None)
        problems += diff(a, b)
        if (old := before.get(res / 'summary.npz')) is not None:
            A, B = np.load(BytesIO(old)), np.load(res / 'summary.npz')
            problems += [f'npz missing {k}' for k in set(A.files) - set(B.files)] + [f'npz new {k}' for k in set(B.files) - set(A.files)]
            problems += [f'npz changed {k}' for k in set(A.files) & set(B.files) if A[k].shape != B[k].shape or not np.allclose(A[k], B[k], rtol=1e-12, atol=0, equal_nan=True)]
        out = run('figures')
        if out.returncode:
            problems.append(f'figures failed: {out.stderr[-500:]}')
        after = snapshot(exp)
        problems += [f'figure changed: {f.relative_to(ROOT)}' for f in after if f.suffix == '.pdf' and before.get(f) != after[f]]
    finally:
        for f in snapshot(exp):
            if f not in before:
                f.unlink()
        for f, data in before.items():
            f.write_bytes(data)
    print(f'{name}: ' + ('reproduced' if not problems else f'{len(problems)} differences') + ''.join(f'\n  {q}' for q in problems))
    return not problems


if __name__ == '__main__':
    sys.exit(0 if all([check(e) for e in sys.argv[1:]]) else 1)
