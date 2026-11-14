"""Per-channel degradations of the producer.

The released shards were written with the degradation code of the Sidon repository as it was before two later
fixes. To reproduce the data, the producer uses these versions by default (`degradations: legacy`):

* packet_loss zeroes the samples in place. The producer starts "clean" and "noisy" as views of one tensor, so when
  no earlier degradation has replaced "noisy" with a new tensor, the zeroed packets also appear in the clean target.
  (Upstream now masks a copy.)
* band_limit returns the resampled signal without restoring the input length, and the codec output is aligned by
  slicing, which keeps a shorter output shorter. (Upstream now matches lengths.) In the producer both are followed
  by padding/cutting to the window length, so they only shift or shorten the degraded signal by a few samples.
* convolve_rir_pra takes the RT60 range as an argument; the producer draws RT60 from U(0.1, 1.0) s.

`degradations: upstream` selects the fixed packet_loss, band_limit and codec of src/sidon/data/preprocess (RT60
keeps the producer's range). The unchanged functions (random_apply, add_non_parametric_noise, clip) are imported
from there.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import pyroomacoustics as pra
import torch
import torchaudio

from sidon.data.preprocess import functional_degrations as upstream
from sidon.data.preprocess.functional_degrations import add_non_parametric_noise, clip, random_apply  # noqa: F401


def align_codec_waveform(original: torch.Tensor, codec_applied: torch.Tensor) -> torch.Tensor:
    """Legacy alignment of a codec output to its input: a longer output is cut to the input length; a shorter one is
    sliced at the offset that minimises the MSE (and so stays shorter)."""
    if original.size(1) == codec_applied.size(1):
        return codec_applied
    if original.size(1) < codec_applied.size(1):
        return codec_applied[:, : original.size(1)]
    mse = []
    for i in range(original.size(1) - codec_applied.size(1) + 1):
        mse.append(torch.mean((original[:, i : i + codec_applied.size(1)] - codec_applied) ** 2))
    start_index = torch.argmin(torch.tensor(mse)).item()
    return codec_applied[:, start_index : start_index + original.size(1)]


def codec(
    sample: dict[str, Any],
    codec_effectors: Sequence[torchaudio.io.AudioEffector],
    input_key: str,
    output_key: str,
) -> dict[str, Any]:
    """Encode and decode with one randomly chosen effector (mp3); on an encoder error the sample is unchanged."""
    x, sr = sample[input_key]
    x = x.view(1, -1)
    effector = random.sample(codec_effectors, 1)[0]
    try:
        codec_applied = effector.apply(x.T, sr).T
    except Exception:
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
    """Zero 3 packets of U(20, 200) ms per 10 s at uniform positions, in place (legacy behaviour)."""
    x, sr = sample[input_key]
    x = x.view(1, -1)
    total_duration = x.size(1) / sr
    num_chunks = int(total_duration * 3 / 10)
    for _ in range(num_chunks):
        chunk_duration_ms = random.uniform(20, 200)
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
    """Resample down to a random rate and back (legacy: the length is not restored)."""
    x, sr = sample[input_key]
    x = x.view(1, -1)
    target_sr = random.sample(candidate_srs, 1)[0]
    down_sampled = torchaudio.functional.resample(x, sr, target_sr)
    new_sample = sample.copy()
    new_sample[output_key] = (torchaudio.functional.resample(down_sampled, target_sr, sr), sr)
    assert x.ndim == 2
    return new_sample


@torch.inference_mode()
def convolve_rir_pra(
    sample: dict[str, Any],
    input_key: str,
    direct_key: str,
    reverb_key: str,
    rir_ds=None,
    rt60_min_max: tuple[float, float] = (0.1, 2.0),
) -> dict[str, Any]:
    """Reverberation with a shoebox room (pyroomacoustics image source method): RT60 ~ U(rt60_min_max), room sides
    U(2, 20) m, source and microphone uniform in the room. `reverb_key` gets the reverberant signal and `direct_key`
    the direct path, both scaled so the louder of the two keeps the input's peak."""
    del rir_ds
    x, sr = sample[input_key]
    x = x.view(1, -1)
    rt60_min, rt60_max = rt60_min_max
    rt60 = random.uniform(rt60_min, rt60_max)
    while True:
        try:
            room_dim = (random.uniform(2, 20), random.uniform(2, 20), random.uniform(2, 20))
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            break
        except Exception:
            continue
    room = pra.ShoeBox(room_dim, fs=sr, max_order=max_order, materials=pra.Material(e_absorption))
    margin = 0.01
    source_pos = [random.uniform(margin, room_dim[0] - margin), random.uniform(margin, room_dim[1] - margin),
                  random.uniform(margin, room_dim[2] - margin)]
    room.add_source(source_pos)
    mic_pos = [random.uniform(margin, room_dim[0] - margin), random.uniform(margin, room_dim[1] - margin),
               random.uniform(margin, room_dim[2] - margin)]
    room.add_microphone(mic_pos)
    anechoic_room = pra.ShoeBox(room_dim, fs=sr, max_order=0, materials=pra.Material(1.0, scattering=0.0))
    anechoic_room.add_source(source_pos)
    anechoic_room.add_microphone(mic_pos)
    room.compute_rir()
    anechoic_room.compute_rir()
    rir = torch.tensor(room.rir[0][0]).unsqueeze(0)
    direct_rir = torch.tensor(anechoic_room.rir[0][0]).unsqueeze(0)
    x = x.view(1, -1)
    rir = rir.view(1, -1)
    original_maximum_amp = x.abs().max()
    direct_rir = direct_rir.view(1, -1)
    new_sample = sample.copy()
    rir_convolved = torchaudio.functional.fftconvolve(x, rir)[:, : x.size(1)]
    direct_rir_convolved = torchaudio.functional.fftconvolve(x, direct_rir)[:, : x.size(1)]
    convolved_maximum_amp = max(rir_convolved.abs().max().item(), direct_rir_convolved.abs().max().item())
    rir_convolved = (rir_convolved / convolved_maximum_amp) * original_maximum_amp
    direct_rir_convolved = (direct_rir_convolved / convolved_maximum_amp) * original_maximum_amp
    new_sample[reverb_key] = (rir_convolved, sr)
    new_sample[direct_key] = (direct_rir_convolved, sr)
    assert new_sample[reverb_key][0].ndim == 2
    assert new_sample[direct_key][0].ndim == 2
    return new_sample


def functions(mode: str = "legacy") -> dict:
    """The degradation functions of a mode: "legacy" (as the released data was made) or "upstream"."""
    if mode == "legacy":
        return {"codec": codec, "packet_loss": packet_loss, "band_limit": band_limit}
    if mode == "upstream":
        return {"codec": upstream.codec, "packet_loss": upstream.packet_loss, "band_limit": upstream.band_limit}
    raise ValueError(f"degradations must be 'legacy' or 'upstream', not {mode!r}")
