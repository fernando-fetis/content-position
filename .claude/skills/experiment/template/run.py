'''NN_name: <one sentence: what the experiment measures, citing by label the statements it tests>.'''
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.core.models import Model, for_each_model
from src.core.suite import prompts
from src.utils.results import figures_dir, read, result_path, save, write_summary
from src.utils.stats import bootstrap_mean


@torch.no_grad()
def measure(cfg: dict, m: Model, task: str) -> dict:
    '''<What is computed for one model and one case, and in which units>: one array per quantity, over prompts.'''
    out: dict = {}
    for pr in prompts(task, m.tokenizer, cfg['n'], cfg['seed']):
        ...
    return out


def compute(cfg: dict, exp: Path) -> None:
    '''One raw result per model and case: a finished case is skipped, so a requeued job resumes and a new variant (a new case) runs alone.'''
    def run(m: Model) -> None:
        for task in cfg['tasks']:
            save(result_path(exp, m.name, task), lambda task=task: measure(cfg, m, task))

    for_each_model(cfg, cfg['models'], run)


def summarize(cfg: dict, exp: Path) -> None:
    '''Only what docs/ or another experiment uses: per model and case, <the derived numbers>, with 95% BCa intervals; variances averaged over prompts before any ratio.'''
    summary: dict = {'models': {}}
    for name in cfg['models']:
        cases = {t: d for t in cfg['tasks'] if (d := read(exp, name, t)) is not None}
        if cases:
            summary['models'][name] = {t: {'mean': float(np.mean(d['value'])), 'ci': list(bootstrap_mean(np.asarray(d['value']), cfg['n_boot'], cfg['seed']))} for t, d in cases.items()}
    write_summary(exp, summary)


def figures(cfg: dict, exp: Path) -> None:
    from figures import draw
    draw(cfg, read(exp, 'summary'), figures_dir(exp))


def check(cfg: dict, exp: Path) -> None:
    '''Identity checks (an edit against the intact model, a closed form against the explicit computation); fails above the tolerance.'''
    def run(m: Model) -> None:
        err = 0.0
        if err > cfg['check_tolerance']:
            raise SystemExit(f'{m.name}: deviation {err:.2e} above {cfg["check_tolerance"]:.0e}')
        print(f'ok   {m.name}: <what is checked>: {err:.2e} (tolerance {cfg["check_tolerance"]:.0e})')

    for_each_model(cfg, cfg['models'][:1], run)
