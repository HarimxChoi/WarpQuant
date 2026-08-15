"""Reference components for WarpQuant."""

from __future__ import annotations

import math

import torch


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Apply a normalized Walsh-Hadamard transform on the last axis."""
    width = x.shape[-1]
    if width < 1 or width & (width - 1):
        raise ValueError("the last dimension must be a positive power of two")

    prefix = x.shape[:-1]
    y = x.clone()
    block = 1
    while block < width:
        y = y.reshape(*prefix, -1, 2, block)
        left, right = y[..., 0, :], y[..., 1, :]
        y = torch.cat((left + right, left - right), dim=-1)
        block *= 2
    return y.reshape_as(x) / math.sqrt(width)


def signed_hadamard(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Apply the deterministic signed Hadamard rotation R = HD."""
    if signs.shape != (x.shape[-1],):
        raise ValueError("signs must match the last dimension")
    return fwht(x * signs.to(device=x.device, dtype=x.dtype))


def output_fisher_score(
    weight: torch.Tensor,
    base_weight: torch.Tensor,
    activations: torch.Tensor,
    output_fisher: torch.Tensor,
    index_bits: int = 32,
) -> torch.Tensor:
    """Score recovery columns by activation and end-loss sensitivity."""
    if weight.ndim != 2 or base_weight.shape != weight.shape:
        raise ValueError("weight and base_weight must have the same 2D shape")
    if activations.shape[-1] != weight.shape[1]:
        raise ValueError("activation width must match weight columns")
    if output_fisher.shape != (weight.shape[0],):
        raise ValueError("output_fisher must match weight rows")

    residual = weight.float() - base_weight.float()
    input_energy = activations.float().square().mean(dim=0)
    loss_weighted_error = (
        output_fisher.float()[:, None] * residual.square()
    ).sum(dim=0)
    cost_bits = 16 * weight.shape[0] + index_bits
    return input_energy * loss_weighted_error / cost_bits


def select_weak_columns(scores: torch.Tensor, count: int) -> torch.Tensor:
    """Return the highest-scoring recovery-column indices."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 0 <= count <= scores.numel():
        raise ValueError("count must be within the number of columns")
    return torch.topk(scores, count, sorted=True).indices


def recover_columns(
    base_weight: torch.Tensor,
    weight: torch.Tensor,
    columns: torch.Tensor,
) -> torch.Tensor:
    """Restore selected original-domain columns in a quantized matrix."""
    if base_weight.shape != weight.shape or base_weight.ndim != 2:
        raise ValueError("base_weight and weight must have the same 2D shape")
    recovered = base_weight.clone()
    recovered[:, columns] = weight[:, columns]
    return recovered


def quantize_activation_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations with a dynamic per-token absolute-maximum scale."""
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127
    values = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
    return values, scale
