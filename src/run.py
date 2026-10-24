'''Runs one stage of one experiment: python src/run.py <experiment> compute|summarize|figures|check [--force] [--smoke] [--key value ...] (or through the Makefile). The experiment is a folder of experiments/, by name (01_measure) or by path (experiments/01_measure).'''
from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import ROOT, RUN
from src.core import models
from src.utils import results
from src.utils.results import EXPERIMENTS, experiment_config


def config(experiment: str, overrides: list[str], smoke: bool = False) -> dict:
    '''The experiment's config, with its smoke block merged over it when smoke is set, then the command-line overrides.'''
    cfg = experiment_config(experiment)
    if smoke:
        block = cfg.pop('smoke', None)
        if block is None:
            raise SystemExit(f'{experiment} has no smoke: block in its config.yaml')
        if unknown := set(block) - set(cfg):
            raise SystemExit(f'unknown config keys in the smoke block of {experiment}: {sorted(unknown)}')
        cfg.update(block)
    if len(overrides) % 2 or not all(k.startswith('--') for k in overrides[::2]):
        raise SystemExit('overrides must be pairs of --key value')
    for key, value in zip(overrides[::2], overrides[1::2]):
        name = key.removeprefix('--').replace('-', '_')
        if name not in cfg:
            raise SystemExit(f'unknown config key {name!r} for {experiment}; available: {sorted(cfg)}')
        cfg[name] = yaml.safe_load(value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description='run one stage of one experiment')
    parser.add_argument('experiment')
    parser.add_argument('stage', choices=('compute', 'summarize', 'figures', 'check'))
    parser.add_argument('--force', action='store_true', help='compute: recompute cases whose results exist; summarize: accept results of different runs')
    parser.add_argument('--smoke', action='store_true', help='test run on the laptop: merges the smoke: block of the config and writes results and figures under .temp/smoke/<experiment>/')
    args, overrides = parser.parse_known_args()
    args.experiment = Path(args.experiment).name
    if args.smoke:
        results.OUTPUT_ROOT = ROOT / '.temp' / 'smoke'
    cfg = config(args.experiment, overrides, args.smoke)
    results.FORCE = args.force
    RUN.update(experiment=args.experiment, stage=args.stage, config=cfg, versions={'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__}, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    exp = EXPERIMENTS / args.experiment
    if args.stage == 'summarize':
        sources = [v for k, v in cfg.items() if k.endswith('_experiment')]
        if not sources and all(p.name.endswith('summary.json') for p in results.results_dir(exp).glob('*.json')):
            print(f'{args.experiment}: no raw results, summary kept')
            return
        results.check_results(args.experiment, *sources)
    sys.path.insert(0, str(exp))
    spec = importlib.util.spec_from_file_location(f'experiments.{args.experiment}', exp / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stage = getattr(module, args.stage, None)
    if stage is None:
        raise SystemExit(f'{args.experiment} has no {args.stage} stage')
    stage(cfg, exp)
    for name, (passes, start, end) in models.PASSES.items():
        print(f'{name}: {passes} forward passes, {(end - start) / passes:.3f} s per pass (backward included)')
    for i in range(torch.cuda.device_count()):
        print(f'cuda:{i} peak GiB: {torch.cuda.max_memory_allocated(i) / 2 ** 30:.1f} allocated, {torch.cuda.max_memory_reserved(i) / 2 ** 30:.1f} reserved, of {torch.cuda.get_device_properties(i).total_memory / 2 ** 30:.1f}')


if __name__ == '__main__':
    main()
