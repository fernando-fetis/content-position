from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from src import ROOT

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

WORDS = ('anchor', 'breeze', 'cactus', 'dagger', 'ember', 'falcon', 'glacier', 'harbor', 'island', 'jigsaw', 'kettle', 'lantern', 'meadow', 'needle', 'orchid', 'pillow', 'quarry', 'ribbon', 'saddle', 'tunnel', 'umbrella', 'violin', 'walnut', 'zephyr')
FIRST = ('Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry', 'Irene', 'Jack', 'Karen', 'Liam', 'Mona', 'Nathan', 'Olivia', 'Peter', 'Quinn', 'Rachel', 'Simon', 'Tina')
LAST = ('Adams', 'Baker', 'Clark', 'Diaz', 'Evans', 'Ford', 'Gray', 'Hill', 'Ito', 'Jones', 'Khan', 'Lopez', 'Mills', 'Nolan', 'Ortiz', 'Park', 'Reed', 'Silva', 'Torres', 'Wong')
WORDS_B = ('basket', 'candle', 'dolphin', 'feather', 'garden', 'helmet', 'jacket', 'kitten', 'ladder', 'magnet', 'napkin', 'oyster', 'parrot', 'rocket', 'salmon', 'teapot', 'tiger', 'velvet', 'wagon', 'wizard', 'yogurt', 'button', 'cherry', 'pencil')
COLORS = ('black', 'blue', 'brown', 'cyan', 'gray', 'green', 'magenta', 'olive', 'orange', 'pink', 'purple', 'red', 'teal', 'violet', 'white', 'yellow')
FILLER = ('The grass is green.', 'The sky is blue.', 'The sun is yellow.', 'Here we go.', 'There and back again.')

TEMPLATES = {
    'index': ('{0}.', 'Question: What is the word at position {k}?', 'Answer: the word at position {k} is'),
    'retrieval': ('{0} {1}.', 'Question: What number follows {q}?', 'Answer: {q}'),
    'mix': ('{0} {1}.', 'Question: What number follows {q}?', 'Answer: {q}'),
    'binding': ('{0} likes the color {1}', 'Question: What color does {q} like the most?', 'Answer: {q} likes the color'),
    'successor': ('{0}.', 'Question: Which word comes right after {q}?', 'Answer: the word right after {q} is'),
    'fixed_point': ('{0}: {1}.', 'Question: Which word is listed at the position given by its own number?', 'Answer: the word listed at the position given by its own number is'),
}


@dataclass
class Example:
    blocks: list[str] = field(default_factory=list)
    question: str = ''
    prefill: str = ''
    answer: str = ''
    distractor: str = ''
    query_block: int | None = None
    text: str | None = None
    answers: list[str] | None = None
    distractors: list[str] | None = None
    prompt: str | None = None


def _example(task: str, items: list, query: int, answer: str, distractor: str, k: str) -> Example:
    sentence, question, prefill = TEMPLATES[task]
    return Example(blocks=[sentence.format(*(x if isinstance(x, (list, tuple)) else (x,))) for x in items], question=question.format(k=k, q=k), prefill=prefill.format(k=k, q=k), answer=' ' + answer, distractor=' ' + distractor, query_block=query)


def _other(values: list, i: int) -> str:
    return next((str(values[(i + s) % len(values)]) for s in range(1, len(values)) if values[(i + s) % len(values)] != values[i]), str(values[i]))


def index(n: int, length: int = 8, seed: int = 0) -> list[Example]:
    rng, out = random.Random(seed), []
    for u in range(n):
        items, k = rng.sample(WORDS, length), u % length
        d = k + 1 if k + 1 < length else k - 1
        out.append(_example('index', items, k, items[k], items[d], str(k + 1)))
    return out


def retrieval(n: int, length: int = 8, seed: int = 0) -> list[Example]:
    rng, out = random.Random(seed), []
    for u in range(n):
        syms, vals = rng.sample(string.ascii_uppercase, length), [rng.randint(10, 99) for _ in range(length)]
        k = u % length
        out.append(_example('retrieval', list(zip(syms, vals)), k, str(vals[k]), _other(vals, k), syms[k]))
    return out


def mix(n: int, length: int = 8, seed: int = 0) -> list[Example]:
    rng, out = random.Random(seed), []
    for _ in range(n):
        syms = rng.sample(string.ascii_uppercase, length - 1)
        q = syms[rng.randrange(length - 2)]
        first = syms.index(q)
        syms.insert(rng.randrange(min(first + 2, length - 1), length), q)
        vals = rng.sample(range(10, 100), length)
        last = len(syms) - 1 - syms[::-1].index(q)
        out.append(_example('mix', list(zip(syms, vals)), last, str(vals[last]), str(vals[first]), q))
    return out


def binding(n_contexts: int, pairs: int = 9, seed: int = 0) -> list[Example]:
    '''Name-color pairs; one example per pair of each context.'''
    rng, out = random.Random(seed), []
    names = [f'{a} {b}' for a in FIRST for b in LAST]
    for _ in range(n_contexts):
        ents, cols = rng.sample(names, pairs), [rng.choice(COLORS) for _ in range(pairs)]
        for k in range(pairs):
            out.append(_example('binding', list(zip(ents, cols)), k, cols[k], _other(cols, k), ents[k]))
    return out


def successor(n: int, length: int = 16, seed: int = 0) -> list[Example]:
    '''A list of length distinct words; the answer is the word right after the named one (a fixed offset from a content), the distractor the word right before.'''
    rng, out = random.Random(seed), []
    for u in range(n):
        items, k = rng.sample(WORDS, length), 1 + u % (length - 2)
        out.append(_example('successor', items, k + 1, items[k + 1], items[k - 1], items[k]))
    return out


HOP_PROMPT = {
    'number': 'You will be given a sequence of space-separated tokens. Your task is to find the target token by following these rules:\n1. Identify the very last token in the sequence, which will always be a number \u2018n\u2018.\n2. Move \u2018n\u2018 positions to the left from that final token.\n3. The token you land on is the target token.\nOutput just the target token and absolutely nothing else. Do not include any explanations or formatting.\n',
    'number_multi': 'You will be given a sequence of space-separated tokens. Your task is to find the target token by following these rules:\n1. Identify the very last token in the sequence, which will always be a number \u2018n\u2018.\n2. Move \u2018n\u2018 positions to the left from that token.\n3. If the token you land on is a number, repeat step 2 from it, with that number as \u2018n\u2018.\n4. The letter you land on is the target token.\nOutput just the target token and absolutely nothing else. Do not include any explanations or formatting.\n',
    'letter': 'You will be given a sequence of space-separated tokens. Your task is to find the target token by following these rules:\n1. Start with the very last token in the sequence.\n2. Identify the second character of that token.\n3. Scan backwards (to the left) to find the token that starts with that exact character; there is only one token that fulfills this condition.\n4. This is your target token.\nOutput just the target token and absolutely nothing else. Do not include any explanations or formatting.\n',
}
HOP_EXAMPLES = {
    'number': [('a z b y c x d w 5', 'y'), ('v t k g n k o h m p 4', 'o'), ('w y b c v t r i p p 10', 'w')],
    'letter': [('a4 b3 c2 d1 fc', 'c2'), ('w9 t7 l5 g1 n4 l6 u9 k7 m5 p8 gk', 'k7'), ('q5 r7 x4 t7 f4 k7 q4 u2 u3 r5 ex', 'x4')],
}


def _hop_prompt(kind: str, examples: list[tuple[str, str]], sequence: str) -> str:
    '''The prompt of Table 2 of Urrutia et al. (arXiv 2605.31558): rules, worked examples and the task, ending in "Output:".'''
    shots = ''.join(f'\nExample {i + 1}:\nInput: "{x}"\nOutput: {y}\n' if kind != 'letter' else f'\nInput: "{x}"\nOutput: {y}\n' for i, (x, y) in enumerate(examples))
    return HOP_PROMPT[kind] + ('\nExample:' if kind == 'letter' else '') + shots + f'\nTask:\nInput: "{sequence}"\nOutput:'


def _number_chain(rng: random.Random, letters: int, context: int, hops: int) -> tuple[list[str], int]:
    '''A sequence of the number task as in their SequenceMultiHopDatasetGenerator: letters at 0..letters-1, integers at letters..context (the last is the query), each the offset back to a uniform earlier position, or along a chain of hops - 1 sorted integer positions ending at a letter; returns the tokens and the answer position.'''
    target = rng.randrange(letters)
    chain = [context, *sorted(rng.sample(range(letters, context), hops - 1), reverse=True), target]
    points = {k: rng.randrange(k) for k in range(letters, context + 1)}
    points.update(zip(chain[:-1], chain[1:]))
    tokens = [rng.choice(string.ascii_lowercase) for _ in range(letters)] + [str(k - points[k]) for k in range(letters, context + 1)]
    return tokens, target


def number_hop(n: int, hops: int = 1, letters: int = 16, context: int = 16, seed: int = 0) -> list[Example]:
    '''The number task of Urrutia et al. (arXiv 2605.31558). hops = 1: their LLM format and prompt (Table 2, verbatim): letters lowercase letters drawn with replacement, then the query n in 1..letters, the answer the letter n places left of it. hops > 1: their dataset construction (_number_chain), their prompt with the rule repeated for numbers and three worked examples of the same construction. The distractor is a neighbour of the answer with another letter.'''
    rng, out = random.Random(seed), []
    def make_one(r: random.Random) -> tuple[list[str], int]:
        if hops == 1:
            toks = [r.choice(string.ascii_lowercase) for _ in range(letters)]
            k = r.randint(1, letters)
            return toks + [str(k)], letters - k
        return _number_chain(r, letters, context, hops)
    shots = HOP_EXAMPLES['number'] if hops == 1 else [(' '.join(t), t[a]) for t, a in (make_one(random.Random(seed - 1 - i)) for i in range(3))]
    kind = 'number' if hops == 1 else 'number_multi'
    for _ in range(n):
        toks, a = make_one(rng)
        last = letters - 1
        other = next((toks[b] for b in (a + 1, a - 1, a + 2, a - 2) if 0 <= b <= last and toks[b] != toks[a]), None)
        if other is None:
            continue
        out.append(Example(prompt=_hop_prompt(kind, shots, ' '.join(toks)), answer=' ' + toks[a], distractor=' ' + other, query_block=a))
    return out


def letter_hop(n: int, length: int = 16, seed: int = 0) -> list[Example]:
    '''The letter task of Urrutia et al. (arXiv 2605.31558) in their LLM format and prompt (Table 2, verbatim, 1 hop): length letter+digit tokens, then a query whose second character is the first letter of exactly one of them, the answer; the distractor is a neighbour of the answer.'''
    rng, out = random.Random(seed), []
    for _ in range(n):
        key = rng.choice(string.ascii_lowercase)
        others = [c for c in string.ascii_lowercase if c != key]
        toks = [rng.choice(others) + str(rng.randint(1, 9)) for _ in range(length)]
        a = rng.randrange(length)
        toks[a] = key + str(rng.randint(1, 9))
        query = rng.choice(others) + key
        other = next(toks[b] for b in (a + 1, a - 1, a + 2, a - 2) if 0 <= b < length and toks[b][0] != key)
        out.append(Example(prompt=_hop_prompt('letter', HOP_EXAMPLES['letter'], ' '.join([*toks, query])), answer=' ' + toks[a], distractor=' ' + other, query_block=a))
    return out


def fixed_point(n: int, length: int = 10, seed: int = 0) -> list[Example]:
    '''A list of length words labelled by a permutation of 1..length with exactly one fixed point; the answer is the word whose label equals its position, which neither labels nor positions alone determine. The distractor is the word whose label is one more than its position, else another word.'''
    rng, out = random.Random(seed), []
    for _ in range(n):
        items, fixed = rng.sample(WORDS, length), rng.randrange(length)
        while True:
            rest = [i for i in range(length) if i != fixed]
            perm = rest[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(rest, perm)):
                break
        labels = [0] * length
        labels[fixed] = fixed
        for a, b in zip(rest, perm):
            labels[a] = b
        other = next((i for i in range(length) if labels[i] == i + 1), (fixed + 1) % length)
        blocks = [(str(labels[i] + 1), items[i]) for i in range(length)]
        out.append(_example('fixed_point', blocks, fixed, items[fixed], items[other], ''))
    return out


def _passages(name: str) -> list[str]:
    return [p.strip() for p in (ROOT / 'data' / f'{name}.txt').read_text().split('=====') if p.strip()]


def corpus(n: int, source: str, chars: int = 8000, seed: int = 0, span: tuple[float, float] = (0.0, 1.0)) -> list[Example]:
    '''n windows of chars characters of data/<source>.txt, each starting at a uniform line start inside the fraction span of the file and ending inside it; disjoint spans give disjoint windows.'''
    rng, text = random.Random(seed), (ROOT / 'data' / f'{source}.txt').read_text()
    lo, hi = int(span[0] * len(text)), int(span[1] * len(text))
    text = text[lo:hi]
    starts = [0] + [i + 1 for i, ch in enumerate(text[:-chars]) if ch == '\n']
    return [Example(text=text[s:s + chars]) for s in (rng.choice(starts) for _ in range(n))]


def random_words(n: int, length: int = 40, seed: int = 0) -> list[Example]:
    '''Texts of length words drawn uniformly with replacement from the tasks' vocabularies: their tokens without their structure.'''
    rng, vocab = random.Random(seed), (*WORDS, *WORDS_B, *FIRST, *LAST, *COLORS)
    return [Example(text=' '.join(rng.choice(vocab) for _ in range(length))) for _ in range(n)]


def key_value(n: int, pairs: int = 40, depths: tuple[int, ...] = (0, 10, 20, 30, 39), seed: int = 0) -> list[Example]:
    '''Retrieval of the value of one key among pairs random six-character key-value pairs, with n examples per depth of the queried pair; the distractor is a value of another pair with another first character.'''
    rng, out = random.Random(seed), []
    code = lambda: ''.join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    for d in depths:
        for _ in range(n):
            keys, values = [code() for _ in range(pairs)], [code() for _ in range(pairs)]
            other = next((v for v in values if v[0] != values[d][0]), None) or next(c for c in iter(code, None) if c[0] != values[d][0])
            out.append(Example(blocks=[f'Key {k} has value {v}.' for k, v in zip(keys, values)], question=f'What is the value of key {keys[d]}?', prefill=f'The value of key {keys[d]} is', answer=' ' + values[d], distractor=' ' + other, query_block=d))
    return out


def _haystack(rng: random.Random, kind: str, n: int) -> list[str]:
    '''n haystack sentences: the filler sentences in turn (noise, as in RULER), or consecutive sentences of the WikiText-2 and Shakespeare passages from a random start, cycled (essay).'''
    if kind == 'noise':
        return [FILLER[i % len(FILLER)] for i in range(n)]
    sents = [x for p in (*_passages('wikitext'), *_passages('shakespeare')) for x in re.split(r'(?<=[.!?])\s+', ' '.join(p.split())) if x]
    start = rng.randrange(len(sents))
    return [sents[(start + i) % len(sents)] for i in range(n)]


def _number(rng: random.Random, digits: int, avoid: set[str] = frozenset()) -> str:
    '''A number of the given digits whose first digit is not among avoid.'''
    return rng.choice([c for c in '123456789' if c not in avoid]) + ''.join(rng.choice(string.digits) for _ in range(digits - 1))


def _sequence(values: list[str]) -> list[str]:
    return [(', ' if i else ' ') + v for i, v in enumerate(values)]


def niah(n: int, blocks: int = 100, keys: int = 1, values: int = 1, depths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0), haystack: str = 'essay', seed: int = 0) -> list[Example]:
    '''RULER needle in a haystack: needles "One of the special magic numbers for <key> is: <7 digits>." among blocks haystack sentences, n examples per depth. The queried key has values needles, the first at the depth and the others at uniform positions; keys - 1 distractor keys have one needle each at uniform positions. The answer is the queried key's values in order of appearance, each checked against a distractor value with another first digit: the value of a distractor needle when there is one, else a random number.'''
    rng, out = random.Random(seed), []
    names = [f'{a}-{b}' for a in WORDS for b in WORDS_B]
    for d in depths:
        for _ in range(n):
            hay, ks = _haystack(rng, haystack, blocks), rng.sample(names, keys)
            vals = [_number(rng, 7) for _ in range(values)]
            others = [_number(rng, 7) for _ in range(keys - 1)]
            spots = sorted(rng.randint(0, blocks) for _ in range(values - 1))
            needles = [(round(d * blocks), 0, ks[0], vals[0])] + [(s, 1 + i, ks[0], v) for i, (s, v) in enumerate(zip(spots, vals[1:]))] + [(rng.randint(0, blocks), values + i, k, v) for i, (k, v) in enumerate(zip(ks[1:], others))]
            needles.sort()
            for s, _, k, v in reversed(needles):
                hay.insert(s, f'One of the special magic numbers for {k} is: {v}.')
            answer = [v for _, _, k, v in needles if k == ks[0]]
            wrong = [next((o for o in others if o[0] != v[0]), None) or _number(rng, 7, {v[0]}) for v in answer]
            q, p = (('What is the special magic number for {k} mentioned in the provided text?', 'The special magic number for {k} mentioned in the provided text is') if values == 1 else ('What are all the special magic numbers for {k} mentioned in the provided text? List them in the order they appear.', 'The special magic numbers for {k} mentioned in the provided text are'))
            seq, dis = _sequence(answer), _sequence(wrong)
            out.append(Example(blocks=hay, question=q.format(k=ks[0]), prefill=p.format(k=ks[0]), answer=seq[0], distractor=dis[0], answers=seq, distractors=dis))
    return out


def variable_tracking(n: int, blocks: int = 100, chains: int = 2, hops: int = 4, depths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0), haystack: str = 'noise', seed: int = 0) -> list[Example]:
    '''RULER variable tracking: chains of hops + 1 assignments ("VAR X1 = 12345.", "VAR X2 = VAR X1.", ...) of five-letter variables among blocks haystack sentences, n examples per depth. The queried chain starts at the depth with its later assignments at uniform positions after it, the other chains at uniform positions; the answer is the queried chain's variables in order, each checked against the variable of another chain at the same hop (first letters differ).'''
    rng, out = random.Random(seed), []
    for d in depths:
        for _ in range(n):
            hay = _haystack(rng, haystack, blocks)
            nums = [_number(rng, 5)]
            nums += [_number(rng, 5, {x[0] for x in nums}) for _ in range(chains - 1)]
            letters = rng.sample(string.ascii_uppercase, chains)
            names = [[c + ''.join(rng.choice(string.ascii_uppercase) for _ in range(4)) for _ in range(hops + 1)] for c in letters]
            start = round(d * blocks)
            spots = [[start, *sorted(rng.randint(start, blocks) for _ in range(hops))]] + [sorted(rng.randint(0, blocks) for _ in range(hops + 1)) for _ in range(chains - 1)]
            stmts = [(s, c, j, f'VAR {names[c][j]} = {nums[c] if j == 0 else "VAR " + names[c][j - 1]}.') for c in range(chains) for j, s in enumerate(spots[c])]
            for s, _, _, text in sorted(stmts, key=lambda x: (x[0], x[1], x[2]), reverse=True):
                hay.insert(s, text)
            seq, dis = _sequence(names[0]), _sequence(names[1 % chains] if chains > 1 else [rng.choice([c for c in string.ascii_uppercase if c != x[0]]) + x[1:] for x in names[0]])
            out.append(Example(blocks=hay, question=f'Find all variables that are assigned the value {nums[0]} in the text above.', prefill=f'According to the chain of variable assignment in the text above, {hops + 1} variables are assigned the value {nums[0]}, they are:', answer=seq[0], distractor=dis[0], answers=seq, distractors=dis))
    return out


def key_value_long(n: int, pairs: int = 40, depths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0), seed: int = 0) -> list[Example]:
    '''key_value with the queried pair at fractional depths of pairs pairs.'''
    return key_value(n, pairs, tuple(round(d * (pairs - 1)) for d in depths), seed)


GENERATORS = {'corpus': corpus, 'index': index, 'retrieval': retrieval, 'mix': mix, 'binding': binding, 'successor': successor, 'number_hop': number_hop, 'letter_hop': letter_hop, 'fixed_point': fixed_point, 'random_words': random_words, 'niah': niah, 'variable_tracking': variable_tracking, 'key_value_long': key_value_long}


def make(spec: dict) -> list[Example]:
    spec = dict(spec)
    return GENERATORS[spec.pop('task')](**spec)


def render(ex: Example, tokenizer: PreTrainedTokenizerBase, max_tokens: int | None = None) -> torch.Tensor:
    '''The token ids of a text example (ex.text), with the tokenizer's special tokens, truncated to max_tokens.'''
    ids = tokenizer(ex.text, return_tensors='pt', add_special_tokens=True).input_ids[0]
    return ids[:max_tokens] if max_tokens else ids


@dataclass
class Checked:
    '''A prompt and its teacher-forced answer: the next-token logits at positions[i] decide check i, answer_ids[i] against distractor_ids[i].'''
    ids: torch.Tensor
    positions: list[int]
    answer_ids: list[int]
    distractor_ids: list[int]


def render_checks(ex: Example, tokenizer: PreTrainedTokenizerBase) -> Checked | None:
    '''The prompt followed by the answers, each tokenized on its own; check i sits at the first token where answer i and distractor i differ. None when one is a prefix of the other.'''
    text = ex.prompt if ex.prompt is not None else 'Context:\n' + '\n'.join(ex.blocks) + '\n' + ex.question + '\n' + ex.prefill
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=True).input_ids[0]
    enc = lambda s: tokenizer(s, add_special_tokens=False).input_ids
    seq, positions, a_ids, d_ids = ids.tolist(), [], [], []
    for a, d in zip(ex.answers or [ex.answer], ex.distractors or [ex.distractor]):
        a, d = enc(a), enc(d)
        p = 0
        while p < min(len(a), len(d)) and a[p] == d[p]:
            p += 1
        if p == min(len(a), len(d)):
            return None
        positions.append(len(seq) + p - 1)
        a_ids.append(a[p])
        d_ids.append(d[p])
        seq += a
    return Checked(ids=torch.tensor(seq[:positions[-1] + 1]), positions=positions, answer_ids=a_ids, distractor_ids=d_ids)
