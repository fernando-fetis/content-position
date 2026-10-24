'''The training side of 05_finetune (`sec:finetune`: *Datos*, *Entrenamiento*, *Evaluación*): the training tokens (FineWeb-Edu, or a local text for tests), the learning-rate schedule and the fine-tuning loop with its evaluations.'''
from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from itertools import batched
from pathlib import Path

import numpy as np
import torch

from src import ROOT
from src.core.models import Model


def _documents(data: dict) -> Iterator[str]:
    '''The training documents in file order: the rows of a parquet file of a Hugging Face dataset (repo, file, revision), or the paragraphs of data/<source>.txt.'''
    if 'source' in data:
        yield from (p for p in (ROOT / 'data' / f'{data["source"]}.txt').read_text().split('\n\n') if p.strip())
        return
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(data['repo'], data['file'], repo_type='dataset', revision=data['revision'])
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=['text']):
        yield from batch.column('text').to_pylist()


def corpus(m: Model, data: dict, n_tokens: int, cache: Path) -> np.ndarray:
    '''The first n_tokens tokens of the training text of `sec:finetune` (each document followed by the end-of-text token), cached per tokenizer at cache; concurrent jobs each write a temporary file and rename it.'''
    if cache.exists() and (arr := np.load(cache)).size >= n_tokens:
        return arr[:n_tokens]
    out, total = [], 0
    for docs in batched(_documents(data), 1024):
        for ids in m.tokenizer(list(docs), add_special_tokens=False).input_ids:
            out.append(np.asarray(ids + [m.tokenizer.eos_token_id], np.uint32))
            total += len(ids) + 1
        if total >= n_tokens:
            break
    if total < n_tokens:
        raise ValueError(f'the training text has {total} tokens, fewer than the {n_tokens} requested')
    arr = np.concatenate(out)[:n_tokens]
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f'{cache.stem}.{os.getpid()}.tmp.npy')
    np.save(tmp, arr)
    tmp.replace(cache)
    return arr


def schedule(step: int, steps: int, opt: dict) -> float:
    '''Learning-rate multiplier at a step (`sec:finetune`, *Entrenamiento*): linear warm-up over the fraction opt['warmup'] of the steps, then cosine decay to opt['final'].'''
    warm = max(1, round(opt['warmup'] * steps))
    if step < warm:
        return (step + 1) / warm
    p = (step - warm) / max(1, steps - warm)
    return opt['final'] + (1 - opt['final']) * 0.5 * (1 + math.cos(math.pi * p))


def evaluation_steps(eval_tokens: list[int], seq: int, batch: int, steps: int) -> list[int]:
    '''The steps (updates done) nearest to the token budgets eval_tokens (`sec:finetune`, *Evaluación*), clipped to the number of steps, deduplicated.'''
    return sorted({min(round(t / (seq * batch)), steps) for t in eval_tokens})


def finetune(m: Model, tokens: np.ndarray, seq: int, batch: int, micro: int, opt: dict, dtype: str, eval_tokens: list[int], evaluate: Callable[[], dict]) -> tuple[dict[int, dict], np.ndarray]:
    '''AdamW fine-tuning of every parameter (`sec:finetune`, *Entrenamiento*) on tokens cut into sequences of seq tokens, batch sequences per step (micro per pass), with autocast to dtype and gradient checkpointing; evaluate() runs before each step of evaluation_steps (after the last one for the final step). Returns the evaluations by step and the training loss per step (nats per token).'''
    data = torch.from_numpy(tokens[: tokens.size // seq * seq].astype(np.int64)).view(-1, seq)
    steps = data.shape[0] // batch
    eval_steps = set(evaluation_steps(eval_tokens, seq, batch, steps))
    model = m.model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    decay = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    optim = torch.optim.AdamW([{'params': decay, 'weight_decay': opt['weight_decay']}, {'params': rest, 'weight_decay': 0.0}], lr=opt['lr'], betas=tuple(opt['betas']), fused=m.device.type == 'cuda')
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: schedule(s, steps, opt))
    cast = getattr(torch, dtype)
    evals, losses = {}, np.zeros(steps)
    for step in range(steps + 1):
        if step in eval_steps:
            model.eval()
            evals[step] = evaluate()
            print(f'step {step}/{steps}: evaluated', flush=True)
        if step == steps:
            break
        model.train()
        rows = data[step * batch:(step + 1) * batch]
        for i in range(0, batch, micro):
            ids = rows[i:i + micro].to(m.device)
            with torch.autocast(m.device.type, dtype=cast, enabled=cast is not torch.float32):
                loss = model(input_ids=ids, labels=ids).loss
            (loss * ids.shape[0] / batch).backward()
            losses[step] += float(loss.detach()) * ids.shape[0] / batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt['clip'])
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()
    model.eval()
    return evals, losses
