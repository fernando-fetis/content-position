'''05_finetune: partial RoPE (a quarter of the planes of each KV head, chosen by the rules of 04_planes, `tab:plane-rules`) followed by fine-tuning on FineWeb-Edu (`sec:finetune`), evaluated along the way as 04 evaluates its edits (`sec:removing-rotation`), against the intact model fine-tuned on the same tokens.'''
from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from src.analysis.calibration import windows
from src.core.models import Model, for_each_model
from src.edits.partial_rope import check_partial_rope, check_partial_rope_agreement, partial_rope, partial_rope_train
from src.eval.text_tasks import evaluate, evaluation_inputs, nll
from src.utils import results
from src.utils.results import EXPERIMENTS, figures_dir, per_model, read, result_path, results_dir, save, write_summary
from src.utils.stats import bootstrap_mean

from train import corpus, evaluation_steps, finetune


def kept_planes(cfg: dict, m: Model, rule: str) -> np.ndarray:
    '''The mask (layers, KV heads, planes) a rule of `tab:plane-rules` keeps at the config's fraction, from 04_planes' selection.'''
    sel = read(EXPERIMENTS / cfg['selection_experiment'], m.name, 'selection')
    if sel is None:
        raise SystemExit(f'{m.name}: no selection result in {cfg["selection_experiment"]}; run its compute first')
    return np.asarray(sel['kept'][f'{rule}_{cfg["fraction"]}'], bool)


def compute(cfg: dict, exp: Path) -> None:
    '''One result per model and run (rule or intact), `sec:finetune`: the evaluation of 04 (`sec:removing-rotation`) at every step of eval_tokens (step_<s>), those steps, and the training loss per step.'''
    for name in cfg['models']:
        for rule in per_model(cfg, 'runs', name):
            if cfg['rule'] not in ('all', rule) or (result_path(exp, name, rule).exists() and not results.FORCE):  # skip a finished run before loading its model
                continue

            def run(m: Model, rule: str = rule) -> None:
                def body() -> dict:
                    inputs = evaluation_inputs(m, cfg)
                    tokens = corpus(m, cfg['data'], cfg['tokens'], results_dir(exp) / f'tokens__{m.tokenizer.name_or_path.replace("/", "_")}.npy')
                    with nullcontext() if rule == 'intact' else partial_rope_train(m, kept_planes(cfg, m, rule)):
                        evals, train_loss = finetune(m, tokens, cfg['seq'], cfg['batch'], per_model(cfg, 'micro', m.name), cfg['optimizer'], cfg['train_dtype'], cfg['eval_tokens'], lambda: evaluate(m, inputs, True))
                    return {'steps': np.array(sorted(evals)), 'loss': train_loss, **{f'step_{s}': e for s, e in evals.items()}}

                save(result_path(exp, m.name, rule), body)

            for_each_model(cfg, [name], run)


def ratio(a: float, b: float) -> float:
    '''a / b, NaN when b is zero (a rule that coincides with the intact model, as in the smoke run).'''
    return a / b if b else math.nan


def summarize(cfg: dict, exp: Path) -> None:
    '''Per model, `sec:finetune`: the evaluation steps and tokens seen; per corpus and length, rule and step: gap, the mean over windows of the NLL minus that of the intact model fine-tuned on the same tokens (nats per token); versus_variance, each rule minus the variance rule with paired 95% BCa intervals over windows; per corpus and length and rule: ratio_to_variance, the final gap over that of the variance rule; tasks, the accuracy per task, run and step (`tab:finetune-tasks`, `fig:finetune`).'''
    summary: dict = {'models': {}}
    for n in cfg['models']:
        runs = {r: d for r in per_model(cfg, 'runs', n) if (d := read(exp, n, r)) is not None}
        if 'intact' not in runs:
            continue
        steps = [int(s) for s in runs['intact']['steps']]
        out: dict = {'steps': steps, 'tokens': [s * cfg['seq'] * cfg['batch'] for s in steps], 'gap': {}, 'versus_variance': {}, 'ratio_to_variance': {}, 'tasks': {}}
        for r, d in runs.items():
            for s in steps:
                if (e := d.get(f'step_{s}')) is None:
                    continue
                ref, ours = runs['intact'][f'step_{s}']['text'], runs.get('variance', {}).get(f'step_{s}')
                for key, x in (e['text'] if r != 'intact' else {}).items():
                    x = np.asarray(x, float)
                    out['gap'].setdefault(key, {}).setdefault(r, []).append(float((x - ref[key]).mean()))
                    if r != 'variance' and ours is not None:
                        diff = x - np.asarray(ours['text'][key], float)
                        out['versus_variance'].setdefault(key, {}).setdefault(r, []).append({'mean': float(diff.mean()), 'ci': list(bootstrap_mean(diff, cfg['n_boot'], cfg['seed']))})
                for t, x in e['tasks'].items():
                    out['tasks'].setdefault(t, {}).setdefault(r, []).append(float(np.mean(np.asarray(x) > 0)))
        for key, g in out['gap'].items():
            done = {r: y for r, y in g.items() if len(y) == len(steps)}
            if 'variance' in done:
                out['ratio_to_variance'][key] = {r: ratio(y[-1], done['variance'][-1]) for r, y in done.items() if r != 'variance'}
        summary['models'][n] = out
    write_summary(exp, summary)


def figures(cfg: dict, exp: Path) -> None:
    from figures import draw
    draw(cfg, read(exp, 'summary'), figures_dir(exp))


def check(cfg: dict, exp: Path) -> None:
    '''On one text window per model: both partial RoPEs with every plane kept give the intact logits (check.tolerance); the two agree with every other plane removed, and give the same NLL with 04's mask of every rule of the model (check.agreement_tolerance); and evaluation_steps clips budgets past the last step.'''
    ck, failures = cfg['check'], []
    assert evaluation_steps([0, 10 * cfg['seq'] * cfg['batch'], 10 ** 12], cfg['seq'], cfg['batch'], 7) == [0, 7], 'evaluation_steps does not clip to the number of steps'
    assert evaluation_steps([0, 2 * cfg['seq'] * cfg['batch']], cfg['seq'], cfg['batch'], 7) == [0, 2]

    def body(m: Model) -> None:
        ids = windows(m, cfg['text']['wiki'], 1, ck['tokens'], cfg['seed'] + 1)[0]
        d_id, d_ag = check_partial_rope(m, ids), check_partial_rope_agreement(m, ids)
        print(f'{m.name}: identity with every plane kept {d_id:.2e} (tolerance {ck["tolerance"]:.0e}); agreement of the two edits with alternate planes {d_ag:.2e} (tolerance {ck["agreement_tolerance"]:.0e})', flush=True)
        if d_id > ck['tolerance']:
            failures.append(f'{m.name}: identity deviation {d_id:.2e}')
        if d_ag > ck['agreement_tolerance']:
            failures.append(f'{m.name}: agreement deviation {d_ag:.2e}')
        rules = [r for r in per_model(cfg, 'runs', m.name) if r != 'intact']
        if read(EXPERIMENTS / cfg['selection_experiment'], m.name, 'selection') is None:
            print(f'{m.name}: no selection result in {cfg["selection_experiment"]}, the NLL agreement per rule is skipped', flush=True)
            return
        for rule in rules:
            kept = kept_planes(cfg, m, rule)
            with partial_rope(m, kept):
                exact = nll(m, ids)
            with partial_rope_train(m, kept):
                trainable = nll(m, ids)
            print(f'{m.name}, {rule}: NLL {exact:.6f} (exact edit of 04) against {trainable:.6f} (trainable edit), deviation {abs(exact - trainable):.2e}', flush=True)
            if abs(exact - trainable) > ck['agreement_tolerance']:
                failures.append(f'{m.name}, {rule}: NLL deviation {abs(exact - trainable):.2e}')

    for_each_model(cfg, cfg['models'], body)
    if failures:
        raise SystemExit('check failed:\n' + '\n'.join(failures))
    print('check passed', flush=True)
