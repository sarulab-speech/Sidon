"""DDP helpers for the manual-optimization GAN stages.

These stages call ``manual_backward`` twice per batch (discriminator, then
generator) inside a single DDP forward. Lightning prepares the DDP reducer
before every ``manual_backward``, so with ``static_graph=False`` each of the
two backward passes all-reduces its gradients, provided every trainable
parameter receives a gradient in it (``find_unused_parameters=False``).
"""

from __future__ import annotations

from typing import Iterable

import torch
from lightning import LightningModule
from torch.nn.parallel import DistributedDataParallel


def ghost_loss(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor | float:
    """Zero-valued term that gives every parameter an exact zero gradient.

    Added to the discriminator loss so that DDP sees the generator parameters
    in the discriminator backward. It touches the parameters directly, so it
    neither changes the loss or any gradient nor back-propagates through the
    generator.
    """
    return sum(p.sum() for p in parameters if p.requires_grad) * 0.0


def check_ddp_syncs_manual_backward(module: LightningModule) -> None:
    """Fail fast when DDP would silently skip the gradient all-reduce.

    With ``static_graph=True`` DDP queues its all-reduce from the backward of
    its own forward outputs. The losses passed to ``manual_backward`` never
    flow through those outputs, so gradients would stay local and every rank
    would train its own replica.
    """
    model = module.trainer.strategy.model
    if (
        isinstance(model, DistributedDataParallel)
        and model.static_graph
        and module.trainer.world_size > 1
    ):
        raise RuntimeError(
            f"{type(module).__name__} uses manual optimization, and "
            "DDPStrategy(static_graph=True) never all-reduces gradients from "
            "manual_backward, so each rank would train its own copy. Train it "
            "with train=gan or set train.trainer.strategy.static_graph=false."
        )
