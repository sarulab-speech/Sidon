import random
from typing import Any, Callable, Sequence

import pyroomacoustics as pra
import torch
import torchaudio

from .rir_utils import estimate_early_rir

from .degrations import align_codec_waveform, clipping


@torch.inference_mode()
def random_apply(
    samples: Sequence[dict[str, Any]], prob: float, transform_fn: Callable, **kwargs
):
    for sample in samples:
        if random.random() < prob:
            sample = transform_fn(sample, **kwargs)
        yield sample


def codec(
    sample: dict[str, Any],
    codec_effectors: Sequence[torchaudio.io.AudioEffector],
    input_key: str,
    output_key: str,
) -> dict[str, Any]:
    x, sr = sample[input_key]
    x = x.view(1, -1)
    effector = random.sample(codec_effectors, 1)[0]
    try:
        codec_applied = effector.apply(x.T, sr).T
    except:
        return sample
    codec_applied = align_codec_waveform(x, codec_applied=codec_applied)
    new_sample = sample.copy()
    new_sample[output_key] = (codec_applied, sr)
    assert codec_applied.ndim == 2
    return new_sample


def packet_loss(
    sample: dict[str, Any],
    input_key: str,
    output_key: str,
    loss_rate: float = 0.1,
) -> dict[str, Any]:
    x, sr = sample[input_key]
    # Randomly mask 10% of the audio with chunks of length 0.1 to 0.5 seconds
    x = x.view(1, -1)
    total_duration = x.size(1) / sr
    num_chunks = int(total_duration * 3 / 10)  # 10% of the total duration

    for _ in range(num_chunks):
        chunk_duration_ms = random.uniform(20, 200)  # 0.1 to 0.5 seconds
        start_time = random.uniform(0, total_duration - chunk_duration_ms / 1000)
        start_sample = int(start_time * sr)
        end_sample = int((start_time + chunk_duration_ms / 1000) * sr)
        x[:, start_sample:end_sample] = 0
    new_sample = sample.copy()
    new_sample[output_key] = (x, sr)
    assert x.ndim == 2
    return new_sample


@torch.inference_mode()
def band_limit(
    sample: dict[str, Any],
    candidate_srs: Sequence[int],
    output_key: str = "noisy",
    input_key: str = "audio",
) -> dict[str, Any]:
    x, sr = sample[input_key]
    x = x.view(1, -1)
    target_sr = random.sample(candidate_srs, 1)[0]
    down_sampled = torchaudio.functional.resample(x, sr, target_sr)
    new_sample = sample.copy()
    new_sample[output_key] = (
        torchaudio.functional.resample(down_sampled, target_sr, sr),
        sr,
    )
    assert x.ndim == 2
    return new_sample


@torch.inference_mode()
def add_non_parametric_noise(
    sample: dict[str, Any],
    input_key: str,
    output_key: str,
    noise_ds: Any,
) -> dict[str, Any]:
    noise_sample = next(noise_ds)
    noise_key = [k for k in noise_sample.keys() if "audio" in k][0]
    noise, noise_sr = noise_sample[noise_key]
    if noise.ndim == 1:
        noise = noise.unsqueeze(0)
    noise = noise[0].unsqueeze(0)
    x, sr = sample[input_key]
    x = x.view(1, -1)
    noise = torchaudio.functional.resample(noise, noise_sr, sr)
    noise = noise.repeat(1, x.size(1) // noise.size(1) + 1)[:, : x.size(1)]
    x = torchaudio.functional.add_noise(
        x, noise, snr=torch.tensor(random.uniform(-5, 20)).unsqueeze(0)
    )
    new_sample = sample.copy()
    new_sample[output_key] = (x, sr)
    assert x.ndim == 2
    return new_sample


def clip(sample: dict[str, Any], input_key: str, output_key: str) -> dict[str, Any]:
    x, sr = sample[input_key]
    x = x.view(1, -1)
    original_shape = x.shape
    min_q = random.uniform(0.0, 0.1)
    max_q = random.uniform(0.9, 1.0)
    x = torch.tensor(
        clipping(x.view(1, -1).numpy(), min_quantile=min_q, max_quantile=max_q)
    )
    new_sample = sample.copy()
    new_sample[output_key] = (x.view(*original_shape), sr)
    assert x.ndim == 2
    return new_sample


@torch.inference_mode()
def convolve_rir_pra(
    sample: dict[str, Any],
    input_key: str,
    direct_key: str,
    reverb_key: str,
    rir_ds=None,
) -> dict[str, Any]:
    del rir_ds
    x, sr = sample[input_key]
    x = x.view(1, -1)
    rt60 = random.uniform(0.1, 2.0)
    while True:
        try:
            room_dim = (
                random.uniform(2, 20),
                random.uniform(2, 20),
                random.uniform(2, 20),
            )
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            break
        except:
            continue
    room = pra.ShoeBox(
        room_dim,
        fs=sr,
        max_order=max_order,
        materials=pra.Material(e_absorption),
    )
    margin = 0.01
    source_pos = [
        random.uniform(margin, room_dim[0] - margin),
        random.uniform(margin, room_dim[1] - margin),
        random.uniform(margin, room_dim[2] - margin),
    ]
    room.add_source(source_pos)
    mic_pos = [
        random.uniform(margin, room_dim[0] - margin),
        random.uniform(margin, room_dim[1] - margin),
        random.uniform(margin, room_dim[2] - margin),
    ]
    room.add_microphone(mic_pos)
    anechoic_room = pra.ShoeBox(
        room_dim,
        fs=sr,
        max_order=0,
        materials=pra.Material(1.0, scattering=0.0),
    )
    anechoic_room.add_source(source_pos)
    anechoic_room.add_microphone(mic_pos)
    room.compute_rir()
    anechoic_room.compute_rir()
    rir = torch.tensor(room.rir[0][0]).unsqueeze(0)  # type: ignore
    direct_rir = torch.tensor(anechoic_room.rir[0][0]).unsqueeze(0)  # type: ignore
    x = x.view(1, -1)
    rir = rir.view(1, -1)
    original_maximum_amp = x.abs().max()
    direct_rir = direct_rir.view(1, -1)
    new_sample = sample.copy()
    rir_convolved = torchaudio.functional.fftconvolve(x, rir)[:, : x.size(1)]
    direct_rir_convolved = torchaudio.functional.fftconvolve(x, direct_rir)[
        :, : x.size(1)
    ]
    convolved_maximum_amp = max(
        rir_convolved.abs().max().item(), direct_rir_convolved.abs().max().item()
    )
    rir_convolved = (rir_convolved / convolved_maximum_amp) * original_maximum_amp
    direct_rir_convolved = (
        direct_rir_convolved / convolved_maximum_amp
    ) * original_maximum_amp

    new_sample[reverb_key] = (
        rir_convolved,
        sr,
    )
    new_sample[direct_key] = (
        direct_rir_convolved,
        sr,
    )
    assert new_sample[reverb_key][0].ndim == 2
    assert new_sample[direct_key][0].ndim == 2

    return new_sample


@torch.inference_mode()
def convolve_rir(
    sample: dict[str, Any],
    input_key: str,
    direct_key: str,
    reverb_key: str,
    rir_ds: Any,
) -> dict[str, Any]:
    x, sr = sample[input_key]
    x = x.view(1, -1)
    sample = next(rir_ds)
    audio_key = [k for k in sample.keys() if "audio" in k][0]
    rir, rir_sr = sample[audio_key]
    rir = rir[random.randint(0, rir.shape[0] - 1)].unsqueeze(0)
    rir = rir.view(1, -1)
    rir = torchaudio.functional.resample(rir, rir_sr, sr)
    direct_rir = torch.tensor(
        estimate_early_rir(rir.numpy(), early_rir_sec=0.05, fs=sr)
    )  # Estimate the early reflections
    new_sample = sample.copy()
    rir_convolved = torchaudio.functional.fftconvolve(x, rir)[:, : x.size(1)]
    direct_rir_convolved = torchaudio.functional.fftconvolve(x, direct_rir)[
        :, : x.size(1)
    ]

    new_sample[reverb_key] = (
        rir_convolved,
        sr,
    )
    new_sample[direct_key] = (
        direct_rir_convolved,
        sr,
    )
    assert new_sample[reverb_key][0].ndim == 2
    assert new_sample[direct_key][0].ndim == 2

    return new_sample
