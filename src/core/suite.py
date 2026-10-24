'''The benchmark: the tasks with their category, and the prompts built from them. The experiment completes a task's spec with n and seed.'''
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.core.tasks import make, render, render_checks

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

TASKS = {
    'index': {'category': 'positional', 'spec': {'task': 'index', 'length': 16}, 'about': 'the word at a position named in the question'},
    'retrieval': {'category': 'symbolic', 'spec': {'task': 'retrieval', 'length': 16}, 'about': 'the number paired with a letter (each letter once)'},
    'binding': {'category': 'symbolic', 'spec': {'task': 'binding', 'pairs': 16}, 'about': 'the color paired with a name (each name once)'},
    'fixed_point': {'category': 'interaction', 'spec': {'task': 'fixed_point', 'length': 10}, 'about': 'the word whose number label equals its position in the list'},
    'number_hop1': {'category': 'positional', 'spec': {'task': 'number_hop', 'hops': 1, 'letters': 16}, 'about': 'number task of Urrutia et al. (arXiv 2605.31558), 1 hop: the letter n places left of the final number n'},
    'number_hop2': {'category': 'interaction', 'spec': {'task': 'number_hop', 'hops': 2, 'letters': 8, 'context': 16}, 'about': 'number task of Urrutia et al. (arXiv 2605.31558), 2 hops: the final number points to a number, which points to the answer letter'},
    'letter_hop1': {'category': 'symbolic', 'spec': {'task': 'letter_hop', 'length': 16}, 'about': 'letter task of Urrutia et al. (arXiv 2605.31558), 1 hop: the token whose first letter is the second character of the final token'},
    'latest': {'category': 'mixed', 'spec': {'task': 'mix', 'length': 16}, 'about': 'the number paired with a letter that appears twice, at its later occurrence'},
    'successor': {'category': 'mixed', 'spec': {'task': 'successor', 'length': 16}, 'about': 'the word right after a word named in the question'},
    'wiki': {'category': 'text', 'spec': {'task': 'corpus', 'source': 'wikitext_test'}, 'about': 'windows of the WikiText-2 test set'},
    'code': {'category': 'text', 'spec': {'task': 'corpus', 'source': 'stdlib'}, 'about': 'windows of Python standard-library source'},
    'random_words': {'category': 'text', 'spec': {'task': 'random_words', 'length': 1200}, 'about': 'the vocabulary of the tasks in random order (a control with no structure)'},
    'niah': {'category': 'long', 'spec': {'task': 'niah', 'keys': 1, 'values': 1}, 'about': 'RULER needle in a haystack, one needle'},
    'niah_multikey': {'category': 'long', 'spec': {'task': 'niah', 'keys': 4, 'values': 1}, 'about': 'RULER needle in a haystack with three distractor needles'},
    'niah_multivalue': {'category': 'long', 'spec': {'task': 'niah', 'keys': 1, 'values': 4}, 'about': 'RULER needle in a haystack, four values of one key'},
    'variable_tracking': {'category': 'long', 'spec': {'task': 'variable_tracking'}, 'about': 'RULER variable tracking, two chains of five assignments'},
    'key_value_long': {'category': 'long', 'spec': {'task': 'key_value_long'}, 'about': 'one key among many random key-value pairs'},
}


def spec(name: str, n: int, seed: int, **extra: object) -> dict:
    '''The generator spec of a task with at least n examples (binding yields one per pair, so ceil(n / pairs) contexts) and a seed.'''
    s = dict(TASKS[name]['spec'])
    if s['task'] == 'binding':
        s['n_contexts'] = -(-n // s['pairs'])
    else:
        s['n'] = n
    s['seed'] = seed
    return s | extra


LENGTH_KEY = {'niah': 'blocks', 'variable_tracking': 'blocks', 'key_value_long': 'pairs'}
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class Prompt:
    '''One input: token ids, query positions (the last is the last token) and, for answered tasks, the answer and distractor tokens predicted at the last token and the context positions of the answer's occurrence (the queried one when the task names it) and of the distractor's first occurrence.'''
    ids: torch.Tensor
    queries: list[int]
    answer_id: int | None = None
    distractor_id: int | None = None
    answer_pos: list[int] | None = None
    distractor_pos: list[int] | None = None


def _span(ex: object, text: str, word: str, queried: bool) -> tuple[int, int] | None:
    '''Character span in text of an occurrence of word in the context: inside the queried block (blocks) or at the queried index of the sequence (hop prompts) when queried, else its first occurrence before the question.'''
    if ex.prompt is not None:
        start = text.rindex('Input: "') + len('Input: "')
        seq = text[start:text.index('"', start)]
        if queried and ex.query_block is not None:
            offset = sum(len(t) + 1 for t in seq.split(' ')[:ex.query_block])
            return start + offset, start + offset + len(seq.split(' ')[ex.query_block])
        i = seq.find(word)
        return (start + i, start + i + len(word)) if i >= 0 else None
    context_end = text.index('\n' + ex.question) if ex.question and ('\n' + ex.question) in text else len(text)
    if queried and ex.query_block is not None and ex.blocks:
        b0 = len('Context:\n') + sum(len(b) + 1 for b in ex.blocks[:ex.query_block])
        block = ex.blocks[ex.query_block]
        i = block.rfind(word)
        return (b0 + i, b0 + i + len(word)) if i >= 0 else (b0, b0 + len(block))
    i = text.find(word, 0, context_end)
    return (i, i + len(word)) if i >= 0 else None


def _answered(ex: object, tokenizer: PreTrainedTokenizerBase) -> Prompt | None:
    ch = render_checks(ex, tokenizer)
    if ch is None:
        return None
    last = ch.positions[0]
    text = ex.prompt if ex.prompt is not None else 'Context:\n' + '\n'.join(ex.blocks) + '\n' + ex.question + '\n' + ex.prefill
    offsets = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)['offset_mapping']
    def tokens(span: tuple[int, int] | None) -> list[int]:
        return [] if span is None else [t for t, (a, b) in enumerate(offsets) if a < span[1] and b > span[0] and b > a and t < last]
    answer = (ex.answers or [ex.answer])[0].strip()
    distractor = (ex.distractors or [ex.distractor])[0].strip()
    return Prompt(ch.ids[:last + 1], [last], ch.answer_ids[0], ch.distractor_ids[0], tokens(_span(ex, text, answer, True)), tokens(_span(ex, text, distractor, False)))


def _fitted(name: str, tokenizer: PreTrainedTokenizerBase, target: int, depth: float, seed: int) -> Prompt | None:
    '''One example of a long task at the given depth, its haystack (or pairs) resized until the prompt is within 2% of target tokens: proportional resizing, then bisection; the closest prompt when no size fits.'''
    base = TASKS[name]['spec']
    key, size, tried = LENGTH_KEY[base['task']], max(2, target // 30), {}
    while size not in tried and len(tried) < 20:
        pr = tried[size] = _answered(make({**base, 'n': 1, key: size, 'depths': (depth,), 'seed': seed})[0], tokenizer)
        if pr is None or abs(len(pr.ids) - target) <= 0.02 * target:
            return pr
        below = max((s for s, p in tried.items() if len(p.ids) < target), default=None)
        above = min((s for s, p in tried.items() if len(p.ids) > target), default=None)
        size = max(2, round(size * target / len(pr.ids)))
        if (len(tried) >= 5 or size in tried) and below is not None and above is not None:
            size = (below + above) // 2
    return min(tried.values(), key=lambda p: abs(len(p.ids) - target))


def prompts(name: str, tokenizer: PreTrainedTokenizerBase, n: int, seed: int, max_tokens: int = 1024, target: int | None = None) -> list[Prompt]:
    '''n inputs of a task (fewer when an answer cannot be told from its distractor by one token). Text: windows of max_tokens. Long tasks: target tokens, depths cycling over DEPTHS. The query is the last token (at least position 3).'''
    cat = TASKS[name]['category']
    if cat == 'long':
        out = [_fitted(name, tokenizer, target, DEPTHS[u % len(DEPTHS)], seed + u) for u in range(n)]
    elif cat == 'text':
        out = []
        for ex in make(spec(name, n, seed))[:n]:
            ids = render(ex, tokenizer, max_tokens)
            out.append(Prompt(ids, [max(3, len(ids) - 1)]))
    else:
        out = [_answered(ex, tokenizer) for ex in make(spec(name, n, seed))[:n]]
    return [p for p in out if p is not None]
