'''Partial RoPE as an exact edit of the attention: in each KV head the kept planes rotate and the others lose their rotation in query and key, with gain one (MHA2MLA's NoPE planes). partial_rope reads the unrotated projections through hooks and computes attention in query chunks; partial_rope_train undoes the rotation of the removed planes on the rotated projections, which is differentiable.'''
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models import Model, head_planes, repeat_kv, use_attention


def _check_mask(m: Model, kept: np.ndarray) -> np.ndarray:
    g = m.geometry
    kept = np.asarray(kept, bool)
    if kept.shape != (g.n_layers, g.n_kv_heads, g.n_freqs):
        raise ValueError(f'kept has shape {kept.shape}, expected (layers, KV heads, planes) = {(g.n_layers, g.n_kv_heads, g.n_freqs)}')
    if m.softcap or m.sliding_window:
        raise NotImplementedError(f'{m.name}: logit softcap or sliding-window layers are not handled by partial RoPE')
    return kept


def unrotated(m: Model, output: torch.Tensor | tuple, heads: int) -> torch.Tensor:
    '''A query or key projection (or norm) output as unrotated heads (batch, heads, positions, head_dim), in the model's dtype.'''
    x = head_planes(m, output, heads)  # (batch, positions, heads, planes, 2)
    return torch.cat([x[..., 0], x[..., 1]], -1).transpose(1, 2)


@contextmanager
def store_unrotated(m: Model, layers: set[int] | None = None) -> Iterator[dict[int, dict[str, torch.Tensor]]]:
    '''Inside the context, the yielded dict receives the unrotated queries ('q') and keys ('k') of each layer of layers (default all), (batch, heads, positions, head_dim); the attention that consumes them pops its layer.'''
    unrot: dict[int, dict[str, torch.Tensor]] = {}
    g = m.geometry

    def store(layer: int, which: str, heads: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple) -> None:
            unrot.setdefault(layer, {})[which] = unrotated(m, output, heads)
        return hook

    hooks = []
    try:
        for layer in range(g.n_layers):
            if layers is None or layer in layers:
                q_t, k_t = m.qk_targets(layer)
                hooks += [q_t.register_forward_hook(store(layer, 'q', g.n_heads)), k_t.register_forward_hook(store(layer, 'k', g.n_kv_heads))]
        yield unrot
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def partial_rope(m: Model, kept: np.ndarray, chunk: int = 512) -> Iterator[None]:
    '''Inside the context, every forward pass (one sequence at a time) runs with partial RoPE: kept (layers, KV heads, F) marks the planes that keep their rotation; the others contribute the logits of the unrotated query and key.'''
    g = m.geometry
    kept = _check_mask(m, kept)
    keep_q = torch.as_tensor(np.repeat(kept, g.groups, axis=1), device=m.device)  # (layers, heads, F)
    keep_d = torch.cat([keep_q, keep_q], -1).float()[:, None, :, None, :]  # (layers, 1, heads, 1, head_dim)

    def attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None, dropout: float = 0.0, scaling: float | None = None, **kw: object) -> tuple[torch.Tensor, None]:
        layer = module.layer_idx
        B, H, T, D = query.shape
        if B != 1:
            raise ValueError('partial_rope runs one sequence at a time')
        scaling = scaling if scaling is not None else D ** -0.5
        G = H // key.shape[1]
        u = unrot.pop(layer)
        qu, ku = u['q'].float(), repeat_kv(u['k'], G).float()
        qr, kr, v = query.float(), repeat_kv(key, G).float(), repeat_kv(value, G)
        kd = keep_d[layer].to(query.device)
        q_keep, q_rem = qr * kd, qu * (1 - kd)
        out = torch.empty_like(qr)
        pos = torch.arange(T, device=query.device)
        for a in range(0, T, chunk):
            b = min(T, a + chunk)
            causal = pos[None, :b] <= pos[a:b, None]
            lg = (q_keep[:, :, a:b] @ kr[:, :, :b].transpose(-1, -2) + q_rem[:, :, a:b] @ ku[:, :, :b].transpose(-1, -2)) * scaling
            out[:, :, a:b] = (torch.softmax(lg.masked_fill(~causal, float('-inf')), -1).to(v.dtype) @ v[:, :, :b]).float()
        return out.transpose(1, 2).to(query.dtype).contiguous(), None

    with store_unrotated(m) as unrot, use_attention(m, attention):
        yield


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)


@contextmanager
def partial_rope_train(m: Model, kept: np.ndarray) -> Iterator[None]:
    '''Inside the context every forward pass runs with partial RoPE (kept as in partial_rope), differentiably: in the removed planes the model's rotation is undone (x cos - rotate_half(x) sin, with the model's own float32 cos and sin), then SDPA. Positions are 0, ..., T - 1: unpadded sequences, no cache.'''
    g = m.geometry
    kept = _check_mask(m, kept)
    keep_k = torch.as_tensor(np.concatenate([kept, kept], -1), dtype=torch.float32, device=m.device)  # (layers, KV heads, head_dim)
    keep_q = keep_k.repeat_interleave(g.groups, 1)
    rotary = m.model.model.rotary_emb
    tables: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def angles(T: int) -> tuple[torch.Tensor, torch.Tensor]:
        if T not in tables:
            cos, sin = rotary(torch.zeros(1, device=m.device), torch.arange(T, device=m.device)[None])
            tables[T] = (cos[0] / rotary.attention_scaling, sin[0] / rotary.attention_scaling)
        return tables[T]

    def attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None, dropout: float = 0.0, scaling: float | None = None, **kw: object) -> tuple[torch.Tensor, None]:
        layer, T = module.layer_idx, query.shape[2]
        cos, sin = (t.to(query.device, query.dtype) for t in angles(T))
        kq, kk = keep_q[layer][None, :, None].to(query.device, query.dtype), keep_k[layer][None, :, None].to(key.device, key.dtype)
        q = query * kq + (query * cos - rotate_half(query) * sin) * (1 - kq)
        k = key * kk + (key * cos - rotate_half(key) * sin) * (1 - kk)
        out = F.scaled_dot_product_attention(q, k, value, is_causal=True, scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None

    with use_attention(m, attention):
        yield


@torch.no_grad()
def _logits(m: Model, ids: torch.Tensor) -> torch.Tensor:
    return m.model(ids[None].to(m.device), use_cache=False).logits[0].double()


@torch.no_grad()
def check_partial_rope(m: Model, ids: torch.Tensor) -> float:
    '''Largest absolute deviation of the logits of ids from the intact model's when every plane is kept, under partial_rope and under partial_rope_train (zero up to float32 rounding, about 1e-7).'''
    g = m.geometry
    base = _logits(m, ids)
    kept = np.ones((g.n_layers, g.n_kv_heads, g.n_freqs), bool)
    with partial_rope(m, kept, chunk=7):
        d1 = float((_logits(m, ids) - base).abs().max())
    with partial_rope_train(m, kept):
        d2 = float((_logits(m, ids) - base).abs().max())
    return max(d1, d2)


@torch.no_grad()
def check_partial_rope_agreement(m: Model, ids: torch.Tensor) -> float:
    '''Largest absolute deviation between the logits of ids under partial_rope and partial_rope_train with every other plane removed; they agree up to the float32 rounding of the rotation (about 1e-6 at a few thousand tokens).'''
    g = m.geometry
    kept = np.zeros((g.n_layers, g.n_kv_heads, g.n_freqs), bool)
    kept[..., ::2] = True
    with partial_rope(m, kept, chunk=7):
        a = _logits(m, ids)
    with partial_rope_train(m, kept):
        b = _logits(m, ids)
    return float((a - b).abs().max())
