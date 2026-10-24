from __future__ import annotations

import gc
import math
import os
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
from dotenv import load_dotenv
from transformers import AttentionInterface, AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from src import ROOT, RUN, registry

PASSES: dict[str, list[float]] = {}


@dataclass(frozen=True)
class Geometry:
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_base: float
    rope_scaling: dict | None = None

    @property
    def n_freqs(self) -> int:
        return self.head_dim // 2

    @property
    def groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def theta(self) -> torch.Tensor:
        inv = self.rope_base ** (-2.0 * torch.arange(self.n_freqs, dtype=torch.float64) / self.head_dim)
        rs = self.rope_scaling or {}
        if rs.get('rope_type', rs.get('type')) == 'llama3':
            factor, lo, hi = float(rs.get('factor', 8.0)), float(rs.get('low_freq_factor', 1.0)), float(rs.get('high_freq_factor', 4.0))
            orig = float(rs.get('original_max_position_embeddings', 8192))
            wavelen = 2 * math.pi / inv
            scaled = torch.where(wavelen > orig / lo, inv / factor, inv)
            smooth = ((orig / wavelen - lo) / (hi - lo)).clamp(0.0, 1.0)
            mid = (1 - smooth) * inv / factor + smooth * inv
            inv = torch.where((wavelen <= orig / lo) & (wavelen >= orig / hi), mid, scaled)
        return inv


def resolve_device(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def _rope_base(cfg: PretrainedConfig) -> float:
    if getattr(cfg, 'rope_theta', None) is not None:
        return float(cfg.rope_theta)
    rp = getattr(cfg, 'rope_parameters', None)
    return float(rp['rope_theta']) if isinstance(rp, dict) and 'rope_theta' in rp else 10000.0


class Model:

    def __init__(self, name: str, device: str = 'auto', dtype: str = 'float32', gpus: int = 1, mlp_chunk: int | None = None) -> None:
        load_dotenv(ROOT / '.env')
        spec = registry()[name]
        token, rev = os.environ.get('HF_TOKEN') or None, spec.get('revision')
        self.name = name
        self.device = resolve_device(device)
        dt = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        if self.device.type != 'cuda' and dt is torch.bfloat16:
            warnings.warn(f'{name}: bfloat16 is only used on CUDA; running in float32 on {self.device.type}', stacklevel=2)
            dt = torch.float32
        cfg = AutoConfig.from_pretrained(spec['hf'], token=token, revision=rev)
        self.config = getattr(cfg, 'text_config', None) or cfg
        self.softcap = getattr(self.config, 'attn_logit_softcapping', None)
        self.sliding_window = getattr(self.config, 'sliding_window', None) if getattr(self.config, 'use_sliding_window', True) else None
        self.tokenizer = AutoTokenizer.from_pretrained(spec['hf'], token=token, revision=rev)
        placement, max_memory = self.device, None
        if gpus > 1:
            if self.device.type != 'cuda' or torch.cuda.device_count() < gpus:
                raise RuntimeError(f'{name}: {gpus} GPUs requested, {torch.cuda.device_count() if torch.cuda.is_available() else 0} visible')
            placement, max_memory = 'auto', {i: int(0.9 * torch.cuda.get_device_properties(i).total_memory) for i in range(gpus)}
        self.model = AutoModelForCausalLM.from_pretrained(spec['hf'], token=token, revision=rev, dtype=dt, attn_implementation='eager' if self.softcap else 'sdpa', device_map=placement, max_memory=max_memory).eval()
        self.device = self.model.get_input_embeddings().weight.device
        heads = int(self.config.num_attention_heads)
        hd = int(getattr(self.config, 'head_dim', None) or self.config.hidden_size // heads)
        self.geometry = Geometry(n_layers=int(self.config.num_hidden_layers), n_heads=heads, n_kv_heads=int(getattr(self.config, 'num_key_value_heads', heads)), head_dim=hd, rope_base=_rope_base(self.config), rope_scaling=getattr(self.config, 'rope_scaling', None))
        self.scaling = float(getattr(self.attention(0), 'scaling', self.geometry.head_dim ** -0.5))
        self.model.register_forward_hook(lambda *_: count(name))
        if mlp_chunk:
            chunk_mlps(self, mlp_chunk)
        RUN['model'] = {'name': name, 'hf': spec['hf'], 'revision': rev, 'dtype': str(dt).removeprefix('torch.'), 'devices': sorted({str(p.device) for p in self.model.parameters()})}

    def layers(self) -> list[nn.Module]:
        return list(self.model.model.layers)

    def attention(self, layer: int) -> nn.Module:
        return self.layers()[layer].self_attn

    def qk_targets(self, layer: int) -> tuple[nn.Module, nn.Module]:
        attn = self.attention(layer)
        q = attn.q_norm if isinstance(getattr(attn, 'q_norm', None), nn.Module) else attn.q_proj
        k = attn.k_norm if isinstance(getattr(attn, 'k_norm', None), nn.Module) else attn.k_proj
        return q, k


def chunk_mlps(m: Model, size: int) -> None:
    '''Runs every MLP on at most size tokens at a time: exact, since the MLP acts per token, and it bounds activation memory on long sequences.'''
    for block in m.layers():
        mlp = block.mlp
        forward = mlp.forward

        def chunked(x: torch.Tensor, forward: Callable = forward) -> torch.Tensor:
            if x.shape[-2] <= size:
                return forward(x)
            return torch.cat([forward(x[..., a:a + size, :]) for a in range(0, x.shape[-2], size)], -2)
        mlp.forward = chunked


def count(name: str) -> None:
    rec = PASSES.setdefault(name, [0, time.time(), 0.0])
    rec[0] += 1
    rec[2] = time.time()
    if rec[0] % 5000 == 0:
        print(f'{time.strftime("%H:%M:%S")} {name}: {rec[0]} forward passes, {(rec[2] - rec[1]) / rec[0]:.3f} s per pass', flush=True)


def pairs(x: torch.Tensor) -> torch.Tensor:
    '''The last axis of x (head_dim) as the RoPE planes (F, 2); rotate_half convention: plane f pairs coordinates f and f + F.'''
    return torch.stack(x.chunk(2, -1), dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)


def rotate(v: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack([cos * v[..., 0] - sin * v[..., 1], sin * v[..., 0] + cos * v[..., 1]], dim=-1)


def head_planes(m: Model, output: torch.Tensor | tuple, heads: int) -> torch.Tensor:
    '''The output of a query or key projection (or norm) as (batch, positions, heads, planes, 2).'''
    x = (output[0] if isinstance(output, tuple) else output).detach()
    if x.dim() == 4:
        x = x.permute(0, 2, 1, 3) if x.shape[1] == heads and x.shape[2] != heads else x
    else:
        x = x.reshape(x.shape[0], x.shape[1], heads, m.geometry.head_dim)
    return pairs(x)


@contextmanager
def stream_qk(m: Model, fn: Callable[[int, torch.Tensor, torch.Tensor], None]) -> Iterator[None]:
    '''Inside it, calls fn(layer, Q, K) as soon as each layer's keys are computed, with the unrotated queries and keys of the first sequence in float32, (positions, query heads, planes, 2), keys repeated over their query heads; nothing is kept across layers.'''
    pending: dict[int, torch.Tensor] = {}

    def on_q(layer: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple) -> None:
            pending[layer] = head_planes(m, output, m.geometry.n_heads)[0].float()
        return hook

    def on_k(layer: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple) -> None:
            q = pending.pop(layer)
            k = repeat_kv(head_planes(m, output, m.geometry.n_kv_heads)[0].float().to(q.device), m.geometry.groups)
            fn(layer, q, k)
        return hook

    hooks = []
    try:
        for layer in range(m.geometry.n_layers):
            q_t, k_t = m.qk_targets(layer)
            hooks += [q_t.register_forward_hook(on_q(layer)), k_t.register_forward_hook(on_k(layer))]
        yield
    finally:
        for h in hooks:
            h.remove()


def attention_probs(lg: torch.Tensor, softcap: float | None, sliding_window: int | None) -> torch.Tensor:
    '''Attention probabilities of the logits lg with softcap, causal and sliding-window masks; entries already at -inf stay masked.'''
    S = lg.shape[-1]
    masked = torch.isneginf(lg)
    if softcap:
        lg = torch.tanh(lg / softcap) * softcap
    mask = torch.ones(S, S, dtype=torch.bool, device=lg.device).tril() & ~masked
    if sliding_window:
        i = torch.arange(S, device=lg.device)
        mask &= (i[:, None] - i[None, :]) < sliding_window
    return torch.softmax(lg.masked_fill(~mask, float('-inf')), -1)


@contextmanager
def use_attention(m: Model, fn: Callable[..., tuple]) -> Iterator[None]:
    '''Registers fn as the attention of every layer for the duration of the context.'''
    name = f'psym_{id(fn)}'
    AttentionInterface.register(name, fn)
    original = m.model.config._attn_implementation
    m.model.set_attn_implementation(name)
    try:
        yield
    finally:
        m.model.set_attn_implementation(original)
        AttentionInterface._global_mapping.pop(name, None)


@contextmanager
def override_attention(m: Model) -> Iterator[None]:
    '''Replaces every layer's attention by eager attention in float32 logits, returning the weights. It applies the causal mask, softcap and sliding window itself and ignores attention_mask: valid without padding, or with right padding when only the positions before it are read.'''
    def attn(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None, dropout: float = 0.0, scaling: float | None = None, softcap: float | None = None, **kw: object) -> tuple[torch.Tensor, torch.Tensor]:
        scaling = scaling if scaling is not None else module.head_dim ** -0.5
        k = repeat_kv(key, query.shape[1] // key.shape[1]).float()
        v = repeat_kv(value, query.shape[1] // value.shape[1])
        q = query.float()
        sw = kw.get('sliding_window') or getattr(module, 'sliding_window', None)
        lg = torch.matmul(q, k.transpose(-1, -2)) * scaling
        p = attention_probs(lg, softcap, sw)
        return torch.matmul(p.to(v.dtype), v).transpose(1, 2).contiguous(), p

    with use_attention(m, attn):
        yield


def random_ids(m: Model, gen: torch.Generator, length: int, token_range: list[int], repeats: int = 1) -> torch.Tensor:
    vocab = m.model.get_input_embeddings().weight.shape[0]
    body = torch.randint(token_range[0], min(vocab, token_range[1]), (length,), generator=gen).repeat(repeats)
    bos = m.tokenizer.bos_token_id
    return body if bos is None else torch.cat([torch.tensor([bos]), body])


def for_each_model(cfg: dict, names: list[str], body: Callable[[Model], None]) -> None:
    '''Loads each model in turn with the device, dtype, gpus (dtype and gpus may be {default: ..., <model>: ...}) and mlp_chunk of cfg, calls body on it and frees it before the next (a model split over GPUs sits in a reference cycle through its device hooks, so only gc frees it).'''
    from src.utils.results import per_model
    for name in names:
        body(Model(name, cfg['device'], per_model(cfg, 'dtype', name), per_model(cfg, 'gpus', name), cfg.get('mlp_chunk')))
        gc.collect()
        torch.cuda.empty_cache()
