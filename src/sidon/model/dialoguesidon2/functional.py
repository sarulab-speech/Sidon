"""Length masks, the masked latent MSE and the two-speaker permutation-invariant assignment."""

import torch
from torch import Tensor


def length_mask(lengths: Tensor, size: int) -> Tensor:
    """[B] valid lengths -> [B, size] boolean mask of the valid prefix of each row."""
    if lengths.ndim != 1 or bool(((lengths <= 0) | (lengths > size)).any()):
        raise ValueError("Lengths must be positive and within the padded sequence")
    return torch.arange(size, device=lengths.device)[None] < lengths[:, None]


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error per example, [B, S, T, D] and [B, T] -> [B], over the valid frames only."""
    error = (pred.float() - target.float()).square()
    error = error.masked_fill(~mask[:, None, :, None], 0)
    return error.sum((1, 2, 3)) / (mask.sum(1) * pred.size(1) * pred.size(3))


def pit_assignment(pred: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Permutation-invariant training for two speakers.

    For each example, the speaker order of `target` that gives the lower error over the whole valid duration is
    chosen (one decision per example, not per frame). Returns the targets in that order and the mean of the
    per-example minimum errors (the loss).
    """
    direct = masked_mse(pred, target, mask)
    swapped = masked_mse(pred, target.flip(1), mask)
    swap = (swapped < direct).detach()
    ordered = torch.where(swap[:, None, None, None], target.flip(1), target)
    return ordered, torch.minimum(direct, swapped).mean()


def finite_or_raise(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"Non-finite values in {name}")
