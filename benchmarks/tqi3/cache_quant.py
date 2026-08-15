from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CacheQuantConfig:
    key_bits: int = 3
    value_bits: int = 4
    value_group_size: int = 32
    residual_length: int = 128
    seed: int = 42


@lru_cache(maxsize=16)
def sphere_lloyd_levels(dim: int, bits: int, samples: int = 262_144) -> np.ndarray:
    if dim < 2:
        raise ValueError("dim must be at least two")
    if bits not in (1, 2, 3, 4):
        raise ValueError("bits must be between one and four")
    rng = np.random.default_rng(0x5451 + dim * 17 + bits)
    magnitude = np.sqrt(rng.beta(0.5, (dim - 1) / 2, size=samples))
    values = magnitude * rng.choice(np.asarray([-1.0, 1.0]), size=samples)
    count = 1 << bits
    levels = np.quantile(values, (np.arange(count) + 0.5) / count)
    for _ in range(64):
        boundaries = (levels[:-1] + levels[1:]) * 0.5
        assignments = np.searchsorted(boundaries, values)
        totals = np.bincount(assignments, weights=values, minlength=count)
        counts = np.bincount(assignments, minlength=count)
        updated = np.divide(totals, counts, out=levels.copy(), where=counts > 0)
        if np.max(np.abs(updated - levels)) < 1e-10:
            levels = updated
            break
        levels = updated
    levels = 0.5 * (levels - levels[::-1])
    result = np.asarray(levels, dtype=np.float32)
    result.setflags(write=False)
    return result


def _repeat_kv(values: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return values
    batch, heads, length, dim = values.shape
    expanded = values[:, :, None, :, :].expand(batch, heads, repeats, length, dim)
    return expanded.reshape(batch, heads * repeats, length, dim)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _rotation(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    matrix = torch.randn((dim, dim), generator=_generator(seed), dtype=torch.float32)
    q, r = torch.linalg.qr(matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q.to(device)


def _projection(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    matrix = torch.randn((dim, dim), generator=_generator(seed), dtype=torch.float32)
    return matrix.to(device)


def _mse_quantize(
    values: torch.Tensor,
    bits: int,
    rotation: torch.Tensor,
) -> torch.Tensor:
    dim = values.shape[-1]
    levels = torch.tensor(sphere_lloyd_levels(dim, bits), device=values.device)
    boundaries = (levels[:-1] + levels[1:]) * 0.5
    source = values.float()
    norms = torch.linalg.vector_norm(source, dim=-1, keepdim=True).clamp_min(1e-12)
    rotated = torch.matmul(source / norms, rotation.T)
    indices = torch.bucketize(rotated.contiguous(), boundaries)
    restored = torch.matmul(levels[indices], rotation) * norms
    return restored


def _product_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    bits: int,
    rotation: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    if bits < 2:
        raise ValueError("TurboQuant product estimation needs at least two bits")
    key_mse = _mse_quantize(key, bits - 1, rotation)
    residual = key.float() - key_mse
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    signs = torch.where(
        torch.matmul(residual, projection.T) >= 0,
        torch.ones((), device=key.device),
        -torch.ones((), device=key.device),
    )
    mse_scores = torch.matmul(query.float(), key_mse.transpose(2, 3))
    query_sketch = torch.matmul(query.float(), projection.T)
    residual_scores = torch.matmul(query_sketch, signs.transpose(2, 3))
    residual_scores *= math.sqrt(math.pi / 2.0) / key.shape[-1]
    residual_scores *= residual_norm.unsqueeze(-2)
    return mse_scores + residual_scores


def quantize_values(values: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    if bits not in (2, 4, 8):
        raise ValueError("value bits must be two, four, or eight")
    if values.shape[-1] % group_size:
        raise ValueError("value dimension must be divisible by group size")
    source = values.float()
    groups = source.reshape(*source.shape[:-1], -1, group_size)
    minimum = groups.amin(dim=-1, keepdim=True)
    maximum = groups.amax(dim=-1, keepdim=True)
    scale = ((maximum - minimum) / ((1 << bits) - 1)).clamp_min(1e-12)
    codes = torch.round((groups - minimum) / scale).clamp(0, (1 << bits) - 1)
    return (codes * scale + minimum).reshape_as(source)


def packed_cache_bytes_per_token(
    head_dim: int,
    key_bits: int,
    value_bits: int,
    value_group_size: int,
) -> int:
    if head_dim % value_group_size:
        raise ValueError("head dimension must be divisible by value group size")
    key_bytes = math.ceil(head_dim * key_bits / 8) + 8  # FP32 input and QJL residual norms.
    value_bytes = math.ceil(head_dim * value_bits / 8) + head_dim // value_group_size * 4
    return key_bytes + value_bytes


def resident_cache_bytes(
    sequence_tokens: int,
    residual_length: int,
    bf16_bytes_per_token: int,
    packed_bytes_per_token: int,
) -> int:
    residual_tokens = min(sequence_tokens, residual_length)
    packed_tokens = sequence_tokens - residual_tokens
    return residual_tokens * bf16_bytes_per_token + packed_tokens * packed_bytes_per_token


class TurboQuantAttention:
    def __init__(self, config: CacheQuantConfig):
        self.config = config
        self._matrices: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_matrices(
        self,
        layer_idx: int,
        dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer_idx, dim, str(device))
        if key not in self._matrices:
            seed = self.config.seed + layer_idx * 1009
            self._matrices[key] = (
                _rotation(dim, seed, device),
                _projection(dim, seed + 1, device),
            )
        return self._matrices[key]

    def __call__(
        self,
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        repeats = module.num_key_value_groups
        key_states = _repeat_kv(key, repeats)
        value_states = _repeat_kv(value, repeats)
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        query_positions = torch.arange(
            key_length - query_length,
            key_length,
            device=query.device,
        ).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=query.device).unsqueeze(0)
        causal = (key_positions <= query_positions).reshape(1, 1, query_length, key_length)
        if self.config.residual_length >= key.shape[-2]:
            scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            scores = scores.masked_fill(~causal, float("-inf"))
            if attention_mask is not None:
                scores += attention_mask
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = F.dropout(weights, p=dropout, training=module.training)
            output = torch.matmul(weights, value_states)
            return output.transpose(1, 2).contiguous(), weights

        rotation, projection = self._get_matrices(module.layer_idx, key.shape[-1], key.device)
        quantized_scores = _product_scores(
            query,
            key_states,
            self.config.key_bits,
            rotation,
            projection,
        )
        quantized_values = quantize_values(
            value_states,
            self.config.value_bits,
            self.config.value_group_size,
        )

        recent = key_positions >= query_positions - self.config.residual_length + 1
        recent = recent.reshape(1, 1, query_length, key_length)

        if self.config.residual_length > 0:
            full_scores = torch.matmul(query.float(), key_states.float().transpose(2, 3))
            scores = torch.where(recent, full_scores, quantized_scores)
        else:
            scores = quantized_scores
        scores *= scaling
        scores = scores.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            scores += attention_mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=dropout, training=module.training)
        if self.config.residual_length > 0:
            recent_weights = weights.masked_fill(~recent, 0)
            old_weights = weights.masked_fill(recent, 0)
            output = torch.matmul(old_weights, quantized_values.to(query.dtype))
            output += torch.matmul(recent_weights, value_states)
        else:
            output = torch.matmul(weights, quantized_values.to(query.dtype))
        return output.transpose(1, 2).contiguous(), weights


