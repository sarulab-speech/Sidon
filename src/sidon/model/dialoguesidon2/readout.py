"""The per-speaker readout head and its permutation-invariant training objective."""

import torch
from torch import Tensor, nn

from .functional import pit_assignment


def split_speakers(joint: Tensor, speakers: int = 2) -> Tensor:
    """[B, T, S*D] -> [B, S, T, D]: speaker s occupies channels s*D .. (s+1)*D of each frame."""
    if joint.ndim != 3 or joint.size(-1) == 0 or joint.size(-1) % speakers:
        raise ValueError(f"Expected a joint latent [B, T, {speakers}D]")
    b, frames, channels = joint.shape
    return joint.reshape(b, frames, speakers, channels // speakers).transpose(1, 2).contiguous()


class ReadoutHead(nn.Module):
    """Mixture encoding [B, T, C] -> both speakers' standardized latents as one [B, T, 2D] tensor:
    a projection to `hidden` channels, then Linear-GELU-Linear."""

    def __init__(self, condition_dim: int, hidden: int, latent_dim: int, speakers: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.speakers = speakers
        self.projection = nn.Linear(condition_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, speakers * latent_dim))

    def forward(self, condition: Tensor) -> Tensor:
        return self.head(self.projection(condition))

    @torch.no_grad()
    def predict(self, condition: Tensor, mask: Tensor) -> Tensor:
        """Inference: [B, T, C] with the [B, T] valid mask -> [B, T, 2D], zero on padded frames."""
        valid = mask[:, :, None]
        return self(condition.masked_fill(~valid, 0)).masked_fill(~valid, 0)

    def objective(self, target: Tensor, condition: Tensor, mask: Tensor,
                  assignment_weight: float = 1.0) -> dict[str, Tensor]:
        """The training loss: the mean squared error of the readout against the clean speakers' latents
        [B, 2, T, D], in the speaker order with the lower error per example (PIT)."""
        if target.ndim != 4 or target.size(1) != self.speakers or target.size(-1) != self.latent_dim:
            raise ValueError(f"Expected clean targets [B, {self.speakers}, T, {self.latent_dim}]")
        b, _, frames, _ = target.shape
        if condition.shape[:2] != (b, frames) or mask.shape != (b, frames):
            raise ValueError("Target, mixture and mask timelines must agree")
        target = target.masked_fill(~mask[:, None, :, None], 0)
        condition = condition.masked_fill(~mask[:, :, None], 0)
        prediction = self(condition)
        _, assignment = pit_assignment(split_speakers(prediction, self.speakers), target, mask)
        return {"loss": assignment_weight * assignment, "assignment": assignment}
