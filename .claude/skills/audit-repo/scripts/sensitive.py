'''Scans every tracked text file for personal or private information: local paths of home folders, e-mail addresses, tokens and keys, and the identity of whoever runs it (git user name and e-mail, login name, machine name, home folder), read at run time so that no name is ever stored in the repository. Extra terms can be passed with --term. Prints file:line and the match; exits 1 if anything is found.

Usage: .venv/bin/python .claude/skills/audit-repo/scripts/sensitive.py [--term WORD ...]'''
from __future__ import annotations

import argparse
import getpass
import re
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SKIP = ('data/',)  # input corpora: public text, scanned only for tokens and keys
PATTERNS = {
    'home path': r'/(Users|home)/[A-Za-z0-9_-][A-Za-z0-9._-]*',
    'e-mail': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    'token or key': r'(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',
}
ALLOWED_EMAILS = r'noreply|example\.(com|org)'


def identity() -> list[str]:
    git = lambda key: subprocess.run(['git', 'config', key], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    terms = [git('user.name'), git('user.email'), getpass.getuser(), socket.gethostname().split('.')[0], str(Path.home())]
    terms += [w for w in git('user.name').split() if len(w) > 3]
    return sorted({t for t in terms if t and len(t) > 2})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--term', action='append', default=[])
    a = p.parse_args()
    terms = identity() + a.term
    term_re = re.compile('|'.join(re.escape(t) for t in terms), re.I) if terms else None
    found = 0
    files = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split('\n')
    for f in filter(None, files):
        path = ROOT / f
        if path.suffix in ('.pdf', '.npz', '.npy', '.png') or not path.is_file():
            continue
        try:
            lines = path.read_text().split('\n')
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(lines, 1):
            hits = []
            for label, pat in PATTERNS.items():
                if f.startswith(SKIP) and label != 'token or key':
                    continue
                hits += [(label, m.group(0)) for m in re.finditer(pat, line) if not (label == 'e-mail' and re.search(ALLOWED_EMAILS, m.group(0)))]
            if term_re and not f.startswith(SKIP):
                hits += [('identity', m.group(0)) for m in term_re.finditer(line)]
            for label, text in hits:
                print(f'{f}:{i}: {label}: {text}')
                found += 1
    print(f'{found} findings (identity terms checked: {len(terms)})')
    sys.exit(1 if found else 0)


if __name__ == '__main__':
    main()
