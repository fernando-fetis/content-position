'''Checks the references from code, configs and READMEs to the Typst documents: every cited label (`def:...`, `prop:...`, `thm:...`, `lem:...`, `cor:...`, `rem:...`, `sec:...`, `eq:...`, `fig:...`, `tab:...`, in backticks) must exist as <label> in some .typ file of docs/ or .temp/ (where notes with methodologies and results may live), no label may be defined in two documents of docs/, and no statement may be cited by number ("Proposition 7"), which goes stale when the numbering changes. Prints every problem; exits 1 if any.

Usage: .venv/bin/python .claude/skills/audit-repo/scripts/labels.py'''
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LABEL = r'(?:def|prop|thm|lem|cor|rem|sec|eq|fig|tab):[A-Za-z0-9-]+'
NUMBERED = re.compile(r'\b(Definition|Proposition|Theorem|Lemma|Corollary|Remark)s? \d+\b')


def main() -> None:
    defined: dict[str, list[str]] = {}
    for doc in sorted((ROOT / 'docs').glob('**/*.typ')) + sorted((ROOT / '.temp').glob('**/*.typ')):
        for label in re.findall(rf'<({LABEL})>', doc.read_text()):
            defined.setdefault(label, []).append(str(doc.relative_to(ROOT)))
    bad = 0
    for label, docs in defined.items():
        if len({d for d in docs if d.startswith('docs/')}) > 1:  # scratch notes in .temp/ may repeat labels of the document
            print(f'DUPLICATE {label}: {", ".join(sorted(set(docs)))}')
            bad += 1
    files = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split('\n')
    for f in filter(None, files):
        if f.startswith(('docs/', 'data/', '.claude/')) or not f.endswith(('.py', '.yaml', '.md', '.typ', '.sbatch', '.ini')):
            continue
        for i, line in enumerate((ROOT / f).read_text().split('\n'), 1):
            for label in re.findall(rf'`({LABEL})`', line):
                if label not in defined:
                    print(f'MISSING {f}:{i}: `{label}` is defined in no document')
                    bad += 1
            for m in NUMBERED.finditer(line):
                print(f'NUMBER {f}:{i}: "{m.group(0)}": cite the statement by its label')
                bad += 1
    print(f'{bad} problems ({len(defined)} labels defined)')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
