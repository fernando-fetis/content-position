from __future__ import annotations

import json
import math
import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml

from src import ROOT, RUN

EXPERIMENTS = ROOT / 'experiments'
FORCE = False
OUTPUT_ROOT: Path | None = None  # smoke runs: results and figures go to OUTPUT_ROOT/<experiment>/
SELECTION = ('models', 'tasks', 'gpus', 'mlp_chunk', 'chunk', 'figures', 'rule', 'head_rules', 'head_fractions', 'layer_rules', 'layer_fraction', 'single_layers', 'n_boot', 'smoke')  # config keys that may differ between the results of one experiment: job splits, drawing, lists of rules whose results do not depend on one another, n_boot (it sets only the intervals of the summary, never a raw result) and the smoke block of the local test


JsonTree = dict | list | str | int | float | bool | None


def _split(obj: object, key: str, arrays: dict[str, np.ndarray]) -> JsonTree:
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        obj = (obj.float() if obj.dtype == torch.bfloat16 else obj).numpy()
    if isinstance(obj, np.ndarray):
        arrays[key] = obj
        return {'__npz__': key}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if '/' in str(k):
                raise ValueError(f'key {k!r} contains "/", which separates nested keys in the .npz')
            out[str(k)] = _split(v, f'{key}/{k}' if key else str(k), arrays)
        return out
    if isinstance(obj, (list, tuple)):
        return [_split(v, f'{key}/{i}', arrays) for i, v in enumerate(obj)]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return {'__float__': str(obj)}
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f'cannot save {type(obj).__name__} at {key!r}')


def _join(obj: JsonTree, arrays: dict[str, np.ndarray]) -> object:
    if isinstance(obj, dict):
        if set(obj) == {'__npz__'}:
            return arrays[obj['__npz__']]
        if set(obj) == {'__float__'}:
            return float(obj['__float__'])
        return {k: _join(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_join(v, arrays) for v in obj]
    return obj


def save(path: Path, obj: dict | Callable[[], dict]) -> None:
    '''Writes obj and the run metadata (RUN, time) to path as strict JSON, arrays in the sibling .npz; the .npz is written first and both atomically, so an existing JSON marks a complete result. A callable obj is computed only when path is missing or FORCE is set.'''
    if callable(obj):
        if path.exists() and not FORCE:
            print(f'{path.name} exists, skipped', flush=True)
            return
        obj = obj()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    arrays: dict[str, np.ndarray] = {}
    tree = _split({**obj, 'meta': {**RUN, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}}, '', arrays)
    npz = path.with_suffix('.npz')
    if arrays:
        with open(npz.with_name(npz.name + '.tmp'), 'wb') as f:
            np.savez_compressed(f, **arrays)
        os.replace(npz.with_name(npz.name + '.tmp'), npz)
    elif npz.exists():
        npz.unlink()
    path.with_name(path.name + '.tmp').write_text(json.dumps(tree, allow_nan=False))
    os.replace(path.with_name(path.name + '.tmp'), path)


def load(path: Path) -> dict:
    tree = json.loads(path.read_text())
    npz = path.with_suffix('.npz')
    if not npz.exists():
        return _join(tree, {})
    with np.load(npz) as arrays:
        return _join(tree, {k: arrays[k] for k in arrays.files})


def output_dir(exp: Path) -> Path:
    '''The folder of the results and figures of exp: exp, or OUTPUT_ROOT/<experiment> when set.'''
    return exp if OUTPUT_ROOT is None else OUTPUT_ROOT / exp.name


def results_dir(exp: Path) -> Path:
    return output_dir(exp) / 'results'


def figures_dir(exp: Path) -> Path:
    '''Where the figures stage of exp writes its PDFs (exp/figures, or OUTPUT_ROOT/<experiment>/figures).'''
    return output_dir(exp) / 'figures'


def result_path(exp: Path, *parts: str) -> Path:
    return results_dir(exp) / ('__'.join(parts) + '.json')


def read(exp: Path, *parts: str) -> dict | None:
    path = result_path(exp, *parts)
    return load(path) if path.exists() else None


def write_summary(exp: Path, summary: dict) -> None:
    save(result_path(exp, 'summary'), summary)


def check_results(*experiments: str) -> None:
    '''Refuses (warns under FORCE) to summarize when the results of any of the experiments were computed with configurations that differ beyond SELECTION.'''
    runs: dict[tuple[str, str], list[str]] = {}
    for name in dict.fromkeys(experiments):
        for path in sorted(results_dir(EXPERIMENTS / name).glob('*.json')):
            if not path.name.endswith('summary.json'):
                cfg = {k: v for k, v in json.loads(path.read_text()).get('meta', {}).get('config', {}).items() if k not in SELECTION}
                runs.setdefault((name, json.dumps(cfg, sort_keys=True)), []).append(path.name)
    if len(runs) > len({n for n, _ in runs}):
        msg = 'results come from different configurations:\n' + '\n'.join(f'{n}, config {cfg}: {", ".join(files[:4])}{" ..." if len(files) > 4 else ""}' for (n, cfg), files in runs.items())
        if not FORCE:
            raise SystemExit(msg + '\nrecompute them, or pass --force to summarize anyway')
        warnings.warn(msg, stacklevel=2)


def experiment_config(experiment: str) -> dict:
    return yaml.safe_load((EXPERIMENTS / experiment / 'config.yaml').read_text())


def per_model(cfg: dict, key: str, model: str) -> object:
    v = cfg[key]
    return (v[model] if model in v else v['default']) if isinstance(v, dict) else v

