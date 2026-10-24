'''The shared evaluation of the edits: next-token loss on text windows and the answer-minus-distractor margin on the benchmark prompts, with whatever attention is in force.'''
from __future__ import annotations

import numpy as np
import torch

from src.analysis.calibration import windows
from src.core.models import Model, count
from src.core.suite import TASKS, Prompt, prompts


@torch.no_grad()
def nll(m: Model, ids: torch.Tensor, chunk: int = 512) -> float:
    '''Mean next-token NLL (nats) over the positions 0, ..., T - 2 of ids; the unembedding runs on chunk positions at a time.'''
    hs = m.model.model(input_ids=ids[None].to(m.device), use_cache=False).last_hidden_state[0]
    count(m.name)
    tot, cnt = 0.0, 0
    for a in range(0, len(ids) - 1, chunk):
        b = min(len(ids) - 1, a + chunk)
        lp = torch.log_softmax(m.model.lm_head(hs[a:b]).float(), -1)
        tot += float(-lp.gather(1, ids[a + 1:b + 1, None].to(m.device)).sum())
        cnt += b - a
    return tot / cnt


@torch.no_grad()
def metric(m: Model, pr: Prompt) -> float:
    '''The answer logit minus the distractor logit at the last token of the prompt.'''
    lg = m.model(pr.ids[None].to(m.device), logits_to_keep=1, use_cache=False).logits[0, -1].float()
    return float(lg[pr.answer_id] - lg[pr.distractor_id])


def evaluation_inputs(m: Model, cfg: dict) -> dict:
    '''The evaluation inputs of one model: text windows per corpus and length under 'text' (keys (name, tokens)), and prompts under 'tasks' (keyed by the task, long tasks by <task>_<length>).'''
    text = {(name, L): windows(m, spec, cfg['text_n'], L, cfg['seed'] + 1) for name, spec in cfg['text'].items() for L in cfg['text_tokens']}
    tasks = {}
    for t in cfg['tasks']:
        if TASKS[t]['category'] == 'long':
            for L in cfg['lengths']:
                tasks[f'{t}_{L}'] = prompts(t, m.tokenizer, cfg['task_n']['long'], cfg['seed'] + 1, target=L)
        else:
            tasks[t] = prompts(t, m.tokenizer, cfg['task_n']['short'], cfg['seed'] + 1)
    return {'text': text, 'tasks': tasks}


def _key(k: object) -> str:
    return '_'.join(map(str, k)) if isinstance(k, tuple) else str(k)


def evaluate(m: Model, inputs: dict, tasks: bool) -> dict:
    '''With whatever attention is in force: the NLL of each window per corpus and length (text.<name>_<tokens>) and, when tasks, the margin of each prompt per task (tasks.<task>).'''
    out = {'text': {_key(k): np.array([nll(m, w) for w in ws]) for k, ws in inputs['text'].items()}}
    if tasks:
        out['tasks'] = {t: np.array([metric(m, p) for p in ps]) for t, ps in inputs['tasks'].items()}
    return out

