"""Reusable loss utilities for Sidon models."""

from __future__ import annotations

from typing import Dict

import torch
from audiotools import AudioSignal
from dac.model.discriminator import Discriminator
from dac.nn.loss import L1Loss, MelSpectrogramLoss, MultiScaleSTFTLoss
from torch import nn

try:  # pragma: no cover - optional dependency typing
    from omegaconf import DictConfig  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - controlled by dependencies
    DictConfig = Dict[str, object]  # type: ignore


class DACLoss(nn.Module):
    """Aggregate losses used for training Descript Audio Codec decoders."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.stft_loss = MultiScaleSTFTLoss(**cfg.stft_loss)
        self.mel_loss = MelSpectrogramLoss(**cfg.mel_loss)
        self.wav_loss = L1Loss()

    def forward(self, target: AudioSignal, predicted: AudioSignal) -> Dict[str, torch.Tensor]:
        """Compute STFT, mel, and waveform reconstruction losses."""
        stft_loss = self.stft_loss(predicted, target)
        mel_loss = self.mel_loss(predicted, target)
        wav_loss = self.wav_loss(predicted, target)
        return {
            "stft_loss": stft_loss,
            "mel_loss": mel_loss,
            "wav_loss": wav_loss,
        }


class GANLoss(nn.Module):
    """Least-squares GAN losses for DAC discriminators."""

    def __init__(self, discriminator: Discriminator) -> None:
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake: torch.Tensor, real: torch.Tensor):
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        d_fake, d_real = self.forward(fake.clone().detach(), real)
        loss_d = 0.0
        for out_fake, out_real in zip(d_fake, d_real):
            loss_d += torch.mean(out_fake[-1] ** 2)
            loss_d += torch.mean((1 - out_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake: torch.Tensor, real: torch.Tensor):
        d_fake, d_real = self.forward(fake, real)
        loss_g = 0.0
        for out_fake in d_fake:
            loss_g += torch.mean((1 - out_fake[-1]) ** 2)

        feature_loss = 0.0
        for idx in range(len(d_fake)):
            for feature_idx in range(len(d_fake[idx]) - 1):
                feature_loss += torch.nn.functional.l1_loss(
                    d_fake[idx][feature_idx],
                    d_real[idx][feature_idx].detach(),
                )
        return loss_g, feature_loss
