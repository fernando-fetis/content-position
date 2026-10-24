'''Attention-only RoPE transformers and the three toy tasks: positional (the symbol n places before the query, which is the number n), symbolic (the value of the pair whose key is the query key) and interaction (the number equal to its own position). Heads are built by hand (the constructions of `prop:three-pure-heads`) or trained; their logits can be replaced by parts of the counterfactual matrix at the query.'''
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from src.analysis import counterfactual as cf
from src.core.models import Geometry, rotate

TASKS = ('positional', 'symbolic', 'interaction')
EDITS = {'full': 'cpe', 'drop_e': 'cp', 'only_c': 'c', 'only_p': 'p', 'only_e': 'e', 'drop_c': 'pe', 'drop_p': 'ce'}


class Vocab:
    '''Token ids: BOS 0, symbols, numbers 1..N, key queries, pair tokens (key, value), the question mark.'''
    def __init__(self, length: int, symbols: int, keys: int) -> None:
        self.N, self.S, self.K = length - 2, symbols, keys
        self.sym0, self.num0 = 1, 1 + symbols
        self.keyq0 = self.num0 + self.N
        self.pair0 = self.keyq0 + keys
        self.qmark = self.pair0 + keys * symbols
        self.size = self.qmark + 1

    def num(self, n: np.ndarray) -> np.ndarray:
        return self.num0 + n - 1

    def pair(self, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        return self.pair0 + key * self.S + value


def make(task: str, voc: Vocab, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    '''n examples of length N + 2 (BOS, N context tokens, the query): ids, answer token and answer position.'''
    N = voc.N
    ids = np.zeros((n, N + 2), int)
    pos = rng.integers(1, N + 1, n)
    if task == 'positional':
        ids[:, 1:N + 1] = voc.sym0 + rng.integers(0, voc.S, (n, N))
        k = N + 1 - pos
        ids[:, -1] = voc.num(k)
    elif task == 'symbolic':
        keys = np.argsort(rng.random((n, voc.K)), 1)[:, :N]
        vals = rng.integers(0, voc.S, (n, N))
        ids[:, 1:N + 1] = voc.pair(keys, vals)
        ids[:, -1] = voc.keyq0 + keys[np.arange(n), pos - 1]
    elif task == 'interaction':
        m = rng.integers(1, N, (n, N))
        j = np.arange(1, N + 1)[None]
        m = np.where(m >= j, m + 1, m)
        m[np.arange(n), pos - 1] = pos
        ids[:, 1:N + 1] = voc.num(m)
        ids[:, -1] = voc.qmark
    else:
        raise ValueError(task)
    answer = ids[np.arange(n), pos] if task != 'symbolic' else voc.sym0 + vals[np.arange(n), pos - 1]
    return {'ids': ids, 'answer': answer, 'pos': pos}


def mixture(voc: Vocab, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    parts = [make(t, voc, n, rng) for t in TASKS]
    which = rng.integers(0, len(TASKS), n)
    return {k: np.stack([p[k] for p in parts])[which, np.arange(n)] for k in parts[0]}


class Toy(nn.Module):
    '''Attention-only transformer: embedding, one layer of n_heads RoPE heads (head vectors as F planes of 2 coordinates) on the residual stream, unembedding; no normalization, no MLP.'''
    def __init__(self, vocab: int, d_model: int, n_heads: int, head_dim: int, base: float) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        lin = lambda a, b: nn.ModuleList([nn.Linear(a, b, bias=False)])
        self.wq, self.wk, self.wv, self.wo = lin(d_model, n_heads * head_dim), lin(d_model, n_heads * head_dim), lin(d_model, n_heads * head_dim), lin(n_heads * head_dim, d_model)
        self.unembed = nn.Linear(d_model, vocab, bias=False)
        self.geometry = Geometry(1, n_heads, n_heads, head_dim, base)
        self.scaling = head_dim ** -0.5

    @property
    def device(self) -> torch.device:
        return self.embed.weight.device

    def forward(self, ids: torch.Tensor, edit: Callable | None = None, capture: dict | None = None) -> torch.Tensor:
        '''Logits (batch, length, vocab). edit(layer, logits, q, k) may replace the attention logits (batch, heads, length, length); capture receives the unrotated q, k (batch, length, heads, F, 2) and the attention of each layer.'''
        g = self.geometry
        B, T = ids.shape
        x = self.embed(ids)
        ang = (torch.arange(T, dtype=torch.float64)[:, None] * g.theta()[None]).float().to(ids.device)[None, :, None, :]
        mask = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
        for l in range(g.n_layers):
            shape = (B, T, g.n_heads, g.n_freqs, 2)
            q, k = self.wq[l](x).view(shape), self.wk[l](x).view(shape)
            v = self.wv[l](x).view(B, T, g.n_heads, g.head_dim)
            lg = torch.einsum('bthfa,bshfa->bhts', rotate(q, ang), rotate(k, ang)) * self.scaling
            if edit is not None:
                lg = edit(l, lg, q, k)
            att = torch.softmax(lg.masked_fill(~mask, float('-inf')), -1)
            if capture is not None:
                capture[l] = {'q': q.detach(), 'k': k.detach(), 'att': att.detach()}
            x = x + self.wo[l](torch.einsum('bhts,bshd->bthd', att, v).reshape(B, T, -1))
        return self.unembed(x)


def part_edit(model: Toy, keep: str) -> Callable:
    '''An edit of the logits of the query at position T = N + 1 on the context positions 1, ..., N: lambda_t = A_tt becomes mu plus the diagonal parts in keep (letters c, p, e) of every head's counterfactual matrix (`def:content-positional-and-interaction-parts`); the first token and the query keep their logits.'''
    def edit(layer: int, lg: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        lg = lg.clone()
        r = lg.shape[-1] - 1
        for b in range(lg.shape[0]):
            A = cf.matrix(cf.factors(model, q[b, r], k[b], r))
            mu = A.mean((1, 2), keepdim=True)
            c, p = A.mean(2, keepdim=True) - mu, A.mean(1, keepdim=True) - mu
            e = A - mu - c - p
            new = mu + sum({'c': c.expand_as(A), 'p': p.expand_as(A), 'e': e}[x] for x in keep)
            lg[b, :, r, 1:r] = new.diagonal(0, 1, 2).to(lg.dtype)
        return lg
    return edit


def hand_built(task: str, voc: Vocab, head_dim: int, base: float, beta: float, gamma: float, seed: int = 0) -> Toy:
    '''One head built by hand on one-hot embeddings. positional: every symbol has the key 1 in every plane and the number n the query beta e^(-i theta_f n), so the logit at offset d is beta sum_f cos(theta_f (d - n)); symbolic: pair tokens and query keys share a random unit code of the key in the slowest half of the planes, times beta for the query; interaction: number m has the key e^(-i theta_f m) and the question mark at position T = N + 1 the query beta e^(-i theta_f T), so the logit of m at position j is beta sum_f cos(theta_f (m - j)). Values copy the answer, scaled by gamma.'''
    F = head_dim // 2
    model = Toy(voc.size, voc.size, 1, head_dim, base)
    theta = model.geometry.theta().numpy()
    Wq, Wk, Wv, Wo = (np.zeros((head_dim, voc.size)) for _ in range(4))
    plane = lambda z: np.stack([z.real, z.imag], -1).reshape(-1)
    rng = np.random.default_rng(seed)
    if task == 'positional':
        for s in range(voc.S):
            Wk[:, voc.sym0 + s] = plane(np.ones(F, complex))
            Wv[s, voc.sym0 + s], Wo[s, voc.sym0 + s] = 1, gamma
        for n in range(1, voc.N + 1):
            Wq[:, voc.num(n)] = beta * plane(np.exp(-1j * theta * n))
    elif task == 'symbolic':
        half = F // 2
        slow = np.zeros((voc.K, head_dim))
        slow[:, 2 * (F - half):] = rng.normal(size=(voc.K, 2 * half))
        slow /= np.linalg.norm(slow, axis=1, keepdims=True)
        for key in range(voc.K):
            Wq[:, voc.keyq0 + key] = beta * slow[key]
            for s in range(voc.S):
                Wk[:, voc.pair(key, s)] = slow[key]
                Wv[s, voc.pair(key, s)] = 1
        for s in range(voc.S):
            Wo[s, voc.sym0 + s] = gamma
    elif task == 'interaction':
        for m in range(1, voc.N + 1):
            Wk[:, voc.num(m)] = plane(np.exp(-1j * theta * m))
            Wv[m - 1, voc.num(m)], Wo[m - 1, voc.num(m)] = 1, gamma
        Wq[:, voc.qmark] = beta * plane(np.exp(-1j * theta * (voc.N + 1)))
    with torch.no_grad():
        model.embed.weight.copy_(torch.eye(voc.size))
        model.unembed.weight.copy_(torch.eye(voc.size))
        model.wq[0].weight.copy_(torch.tensor(Wq) / model.scaling)
        model.wk[0].weight.copy_(torch.tensor(Wk))
        model.wv[0].weight.copy_(torch.tensor(Wv))
        model.wo[0].weight.copy_(torch.tensor(Wo).T)
    return model


def train(model: Toy, sample: Callable[[int, np.random.Generator], dict], steps: int, batch: int, lr: float, seed: int, eval_every: int, n_eval: int) -> list[tuple[int, float, float]]:
    '''Adam with cosine decay to zero on the cross-entropy of the answer at the last position; returns (step, loss, held-out accuracy on n_eval examples of seed 10000 + seed) every eval_every steps.'''
    rng, held = np.random.default_rng(seed), sample(n_eval, np.random.default_rng(10_000 + seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    log = []
    for step in range(1, steps + 1):
        d = sample(batch, rng)
        logits = model(torch.as_tensor(d['ids'], device=model.device))[:, -1]
        loss = nn.functional.cross_entropy(logits, torch.as_tensor(d['answer'], device=model.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            log.append((step, loss.item(), accuracy(model, held)))
    return log


@torch.no_grad()
def accuracy(model: Toy, d: dict, edit: Callable | None = None, chunk: int = 500) -> float:
    hits = []
    for a in range(0, len(d['ids']), chunk):
        logits = model(torch.as_tensor(d['ids'][a:a + chunk], device=model.device), edit)[:, -1]
        hits.append((logits.argmax(-1).cpu().numpy() == d['answer'][a:a + chunk]))
    return float(np.concatenate(hits).mean())


@torch.no_grad()
def measure(model: Toy, d: dict, chunk: int = 500) -> dict:
    '''At the last query of each example, per layer and head (examples, L, H): the variances of the parts on the context positions 1, ..., N (the query at T = N + 1) and the attention on the answer's position.'''
    g = model.geometry
    out = {k: [] for k in ('var_c', 'var_p', 'var_e', 'att_answer')}
    for a in range(0, len(d['ids']), chunk):
        ids = torch.as_tensor(d['ids'][a:a + chunk], device=model.device)
        cap: dict = {}
        model(ids, capture=cap)
        r = ids.shape[1] - 1
        rows = {k: np.zeros((len(ids), g.n_layers, g.n_heads)) for k in out}
        for l in range(g.n_layers):
            for b in range(len(ids)):
                P = cf.parts(cf.factors(model, cap[l]['q'][b, r], cap[l]['k'][b], r))
                for k in ('var_c', 'var_p', 'var_e'):
                    rows[k][b, l] = P[k].cpu().numpy()
            rows['att_answer'][:, l] = cap[l]['att'][:, :, r].cpu().numpy()[np.arange(len(ids)), :, d['pos'][a:a + chunk]]
        for k in out:
            out[k].append(rows[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def edits(model: Toy, d: dict) -> dict[str, float]:
    '''The accuracy with each edit of EDITS in every head (full: intact).'''
    return {name: accuracy(model, d, part_edit(model, keep) if name != 'full' else None) for name, keep in EDITS.items()}
