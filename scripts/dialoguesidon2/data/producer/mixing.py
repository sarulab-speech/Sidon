"""The mixing step of the producer: one record's two channels and their weighted mixture.

Only PyTorch and the standard library, so the step can be used and tested without the degradation dependencies.
"""

from __future__ import annotations

import random
from typing import Any

import torch

RATIO_RANGE = (0.3, 0.7)


def mix(wavs: torch.Tensor, ratio: float) -> torch.Tensor:
    """(batch, 2, samples) -> (batch, samples): ratio * channel 0 + (1 - ratio) * channel 1."""
    return wavs[:, 0, :] * ratio + (1 - ratio) * wavs[:, 1, :]


@torch.inference_mode()
def collate(samples: list[dict[str, Any]], sampling_rate: int, use_noise: bool = True,
            max_duration: float | None = None, force_max_length: bool = True) -> dict[str, Any]:
    """Batch of producer samples -> the record written to the shards (before torch.save / pickling).

    Every sample holds "clean", "noisy", "clean_16k" and "noisy_16k" as ((2, samples) tensor, rate). One mixing
    ratio r ~ U(0.3, 0.7) is drawn per batch (from Python's `random`) and used for all four mixtures."""
    clean_wavs = torch.nn.utils.rnn.pad_sequence([x["clean"][0] for x in samples], batch_first=True).float()
    noisy_wavs = torch.nn.utils.rnn.pad_sequence([x["noisy"][0] for x in samples], batch_first=True).float()
    clean_wav_lens = torch.tensor([x["clean"][0].size(-1) for x in samples], dtype=torch.long)
    clean_16k_wavs = torch.nn.utils.rnn.pad_sequence([x["clean_16k"][0] for x in samples], batch_first=True).float()
    noisy_16k_wavs = torch.nn.utils.rnn.pad_sequence([x["noisy_16k"][0] for x in samples], batch_first=True).float()
    ratio = random.uniform(*RATIO_RANGE)
    clean_mixture = mix(clean_wavs, ratio)
    noisy_mixture = mix(noisy_wavs, ratio)
    clean_16k_mixture = mix(clean_16k_wavs, ratio)
    noisy_16k_mixture = mix(noisy_16k_wavs, ratio)
    if force_max_length and max_duration is not None:
        max_length = int(max_duration * sampling_rate)
        clean_wavs = torch.nn.functional.pad(clean_wavs, (0, max_length - clean_wavs.size(-1)))
    sample = {
        "input_wav": clean_wavs,
        "input_wav_lens": clean_wav_lens,
        "sr": sampling_rate,
        "names": [x["__key__"] for x in samples],
        "sample_rate": [x["clean"][1] for x in samples],
        "clean_mixture": clean_mixture,
        "clean_16k_mixture": clean_16k_mixture,
        "noisy_mixture": noisy_mixture,
        "noisy_16k_mixture": noisy_16k_mixture,
    }
    if use_noise:
        sample["noisy_input_wav16k"] = noisy_16k_wavs
        sample["noisy_input_wav"] = noisy_wavs
    return sample
