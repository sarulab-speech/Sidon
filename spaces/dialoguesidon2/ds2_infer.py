"""DialogueSidon2 inference from its torch.export release (no training code needed).

Two programs and a metadata file:
- separator.pt2: w2v-BERT 2.0 input features [1, T, 160] -> decoder-ready latents [1, 2, T, D] (the fine-tuned
  w2v-BERT 2.0 conditioner and the per-speaker readout; deterministic, no sampling);
- vae_decoder.pt2: latents [2, D, T] -> waveforms [2, N] at 24 kHz;
- metadata.json: sample rates, hop and the long-input settings.

The front end is the model's input path: channel mean, peak 0.9, resampling to 24 kHz and then to
16 kHz, 160 samples of zero padding on each side, Kaldi-style 80-bin log-mel features (the w2v-BERT 2.0 settings)
normalized per utterance, and frame pairs stacked. Inputs up to the metadata's window_seconds are separated in
one shot. Longer inputs are separated in windows of that length with an overlap; each window is aligned to
the previous one by the correlation of the two outputs in the overlap (the speaker order of a window is
arbitrary) and cross-faded into it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.export.passes import move_to_device_pass
from torchaudio.compliance import kaldi

SAMPLE_RATE = 24_000
FEATURE_RATE = 16_000


def _program(path: Path, device: torch.device) -> torch.nn.Module:
    """An exported program as a module on `device`. The programs were exported on the CPU, and some graph nodes
    create tensors on an explicit device; move_to_device_pass moves those as well as the weights."""
    program = torch.export.load(str(path))
    if device.type != "cpu":
        program = move_to_device_pass(program, device)
    return program.module()


def load(directory: str | Path, device: torch.device | str = "cpu") -> dict:
    """The two programs on `device` and the metadata, from a directory holding the release files."""
    directory, device = Path(directory), torch.device(device)
    if device.type == "cuda" and device.index is None:
        # The graphs check tensor devices exactly ("cuda:0", not "cuda").
        device = torch.device("cuda", torch.cuda.current_device())
    meta = json.loads((directory / "metadata.json").read_text())
    return {
        "separator": _program(directory / "separator.pt2", device),
        "decoder": _program(directory / "vae_decoder.pt2", device),
        "meta": meta,
        "device": device,
    }


def prepare(wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """[C, N] or [N] audio at sample_rate -> [1, N24] mono at 24 kHz with a peak of 0.9 (before resampling)."""
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if wave.ndim != 2:
        raise ValueError("Expected audio of shape (channels, samples) or (samples,)")
    wave = wave.float().mean(dim=0, keepdim=True)
    peak = wave.abs().max()
    if peak > 0:
        wave = wave / peak * 0.9
    if sample_rate != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sample_rate, SAMPLE_RATE)
    return wave


_CACHE: dict = {}


def _window_and_mel(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = ("fbank", device, dtype)
    if key not in _CACHE:
        window = kaldi._feature_window_function("povey", 400, 0.42, device, dtype)
        mel = F.pad(kaldi.get_mel_banks(80, 512, float(FEATURE_RATE), 20.0, 0.0, 100.0, -500.0, 1.0)[0], (0, 1))
        _CACHE[key] = (window, mel.to(device=device, dtype=dtype))
    return _CACHE[key]


def _to_feature_rate(wave: torch.Tensor) -> torch.Tensor:
    """24 kHz -> 16 kHz with the transform the model was trained with (torchaudio.transforms.Resample)."""
    key = ("resample", wave.device)
    if key not in _CACHE:
        _CACHE[key] = torchaudio.transforms.Resample(SAMPLE_RATE, FEATURE_RATE).to(wave.device)
    return _CACHE[key](wave)


def features(wave: torch.Tensor) -> torch.Tensor:
    """[1, N] at 24 kHz -> [1, T, 160] stacked features: torchaudio's Kaldi fbank with the w2v-BERT 2.0 settings
    (25 ms povey frames every 10 ms, DC removal, 0.97 preemphasis, 512-point power spectrum, 80 mel bins from
    20 Hz), each bin normalized over the utterance, and pairs of frames stacked."""
    with torch.autocast(device_type=wave.device.type, enabled=False):
        x = _to_feature_rate(wave.float())
        x = F.pad(x, (160, 160 + max(0, 320 - x.size(-1))))
        frames = x.unfold(-1, 400, 160)
        frames = frames - frames.mean(-1, keepdim=True)
        frames = frames - 0.97 * torch.cat((frames[..., :1], frames[..., :-1]), -1)
        window, mel = _window_and_mel(x.device, x.dtype)
        power = torch.fft.rfft(F.pad(frames * window, (0, 512 - 400))).abs().pow(2)
        fbank = torch.matmul(power, mel.T).clamp_min(torch.finfo(torch.float).eps).log()  # [1, frames, 80]
        fbank = (fbank - fbank.mean(1, keepdim=True)) / torch.sqrt(fbank.var(1, keepdim=True) + 1e-5)
        pairs = fbank.size(1) // 2  # an odd last frame is dropped, as the feature extractor's attention mask does
        return fbank[:, :2 * pairs].reshape(1, pairs, 160)


@torch.inference_mode()
def separate_window(wave: torch.Tensor, models: dict) -> torch.Tensor:
    """[1, N] at 24 kHz (at most the model's maximum) -> [2, N] separated speakers at 24 kHz."""
    device = models["device"]
    samples = wave.size(-1)
    latent = models["separator"](features(wave.to(device)))  # [1, 2, T, D]
    out = models["decoder"](latent[0].transpose(1, 2).contiguous())  # [2, N']
    if abs(out.size(-1) - samples) > models["meta"]["hop"]:
        raise RuntimeError("Feature frames and waveform duration disagree")
    out = out[:, :samples]
    return F.pad(out, (0, samples - out.size(-1)))


def _similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1) - a.mean(), b.reshape(-1) - b.mean()
    norm = torch.linalg.norm(a) * torch.linalg.norm(b)
    return float(torch.dot(a, b) / norm) if float(norm) > 1e-8 else 0.0


def window_starts(total: int, window: int, overlap: int) -> list[int]:
    """Window starts covering [0, total): a hop of window - overlap, and a last window ending at total."""
    if total <= window:
        return [0]
    starts = list(range(0, total - window, window - overlap))
    return starts + [total - window]


@torch.inference_mode()
def separate(wave: torch.Tensor, sample_rate: int, models: dict, log: list | None = None) -> torch.Tensor:
    """Audio [C, N] or [N] at sample_rate -> [2, N24] separated speakers at 24 kHz (on the CPU).
    log, if given, receives one record per window boundary: the overlap and both alignment scores."""
    meta = models["meta"]
    wave = prepare(wave, sample_rate)
    window = int(meta["window_seconds"] * SAMPLE_RATE)
    overlap = int(meta["overlap_seconds"] * SAMPLE_RATE)
    total = wave.size(-1)
    if total <= window:
        return separate_window(wave, models).float().cpu()
    output = None
    end_before = 0
    for start in window_starts(total, window, overlap):
        chunk = wave[:, start:start + window]
        # Each window is separated at the model's input level (peak 0.9) and scaled back, so the windows keep
        # their relative level.
        peak = float(chunk.abs().max())
        if peak <= 0:
            out = torch.zeros(2, chunk.size(-1))
        else:
            out = separate_window(chunk / peak * 0.9, models).float().cpu() * (peak / 0.9)
        if output is None:
            output, end_before = out, start + chunk.size(-1)
            continue
        shared = end_before - start
        before, now = output[:, -shared:], out[:, :shared]
        direct = _similarity(before[0], now[0]) + _similarity(before[1], now[1])
        swapped = _similarity(before[0], now[1]) + _similarity(before[1], now[0])
        if swapped > direct:
            out = out[[1, 0]]
        if log is not None:
            log.append({"start": start / SAMPLE_RATE, "overlap": shared / SAMPLE_RATE, "direct": direct,
                        "swapped": swapped, "swap": swapped > direct})
        fade = torch.linspace(0.0, 1.0, shared).unsqueeze(0)
        blended = output[:, -shared:] * (1 - fade) + out[:, :shared] * fade
        output = torch.cat([output[:, :-shared], blended, out[:, shared:]], dim=-1)
        end_before = start + chunk.size(-1)
    return output
