"""The w2v-BERT 2.0 feature front end, batched on the model device.

Input features follow the w2v-BERT 2.0 feature extractor: audio at 16 kHz, 80-bin Kaldi log-mel filter banks
(25 ms povey windows every 10 ms, DC removal, 0.97 preemphasis, 512-point power spectrum, 20 Hz to Nyquist), each
bin normalized over the utterance, and pairs of frames stacked into 160-dim vectors (50 Hz, one frame per 480
samples at 24 kHz). Waveforms get 160 samples of zero padding on each side before the filter bank.
"""

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor

from sidon.model.dialoguesidon2.functional import finite_or_raise, length_mask

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def kaldi_fbank(waves: Tensor) -> Tensor:
    """Batched torchaudio.compliance.kaldi.fbank with the w2v-BERT 2.0 settings: [k, n] 16 kHz waveforms ->
    [k, frames, 80] log mel energies (snip_edges framing). The window and mel banks are built once per device."""
    from torchaudio.compliance import kaldi

    size, shift, padded = 400, 160, 512
    frames = waves.unfold(-1, size, shift)
    frames = frames - frames.mean(-1, keepdim=True)
    frames = frames - 0.97 * torch.cat((frames[..., :1], frames[..., :-1]), -1)
    window, mel = _cached(("fbank", waves.device, waves.dtype), lambda: (
        kaldi._feature_window_function("povey", size, 0.42, waves.device, waves.dtype),
        F.pad(kaldi.get_mel_banks(80, padded, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)[0], (0, 1))
        .to(device=waves.device, dtype=waves.dtype)))
    power = torch.fft.rfft(F.pad(frames * window, (0, padded - size))).abs().pow(2)
    return torch.matmul(power, mel.T).clamp_min(torch.finfo(torch.float).eps).log()


def stack_features(features: list[Tensor], stride: int = 2) -> dict[str, Tensor]:
    """Pad to an even frame count and stack frame pairs; the attention mask marks pairs whose second frame is
    real (an odd last frame is dropped)."""
    sizes = [feature.size(0) for feature in features]
    target = -(-max(sizes) // stride) * stride
    padded = features[0].new_zeros(len(features), target, features[0].size(1))
    mask = torch.zeros(len(features), target, dtype=torch.long, device=padded.device)
    for i, (feature, size) in enumerate(zip(features, sizes)):
        padded[i, :size] = feature
        mask[i, :size] = 1
    return {"input_features": padded.reshape(len(features), target // stride, -1),
            "attention_mask": mask[:, 1::stride]}


def extract_features(mixture: Tensor, lengths: Tensor, sample_rate: int) -> tuple[dict[str, Tensor], Tensor]:
    """Mono waveforms [B, N] with [B] lengths -> input features on the waveforms' device and the [B] lengths at
    24 kHz. Waveforms of equal length are processed together."""
    if mixture.ndim != 2 or len(lengths) != mixture.size(0) or sample_rate <= 0:
        raise ValueError("Expected mono mixture [B,N], lengths [B], and a positive sample rate")
    length_mask(lengths, mixture.size(1))
    if (lengths > 120 * sample_rate).any():
        raise ValueError("Input exceeds 120 seconds; split it into windows (sidon.dialoguesidon2.infer)")
    finite_or_raise(mixture, "input waveform")
    resample = None if sample_rate == 16000 else _cached(
        ("resample", sample_rate, mixture.device),
        lambda: torchaudio.transforms.Resample(sample_rate, 16000).to(mixture.device))
    sizes = lengths.tolist()
    features = [None] * len(sizes)
    # Filter banks and normalization run in float32 even under mixed precision.
    with torch.autocast(device_type=mixture.device.type, enabled=False):
        for size in sorted(set(sizes)):
            index = [i for i, n in enumerate(sizes) if n == size]
            wave = mixture[index, :size].detach().float()
            if resample is not None:
                wave = resample(wave)
            # At least two filter-bank frames, for the normalization and the stacking.
            wave = F.pad(wave, (160, 160 + max(0, 320 - wave.size(-1))))
            fbank = kaldi_fbank(wave)
            fbank = (fbank - fbank.mean(1, keepdim=True)) / torch.sqrt(fbank.var(1, keepdim=True) + 1e-5)
            for i, value in zip(index, fbank):
                features[i] = value
        inputs = stack_features(features)
    finite_or_raise(inputs["input_features"], "input_features")
    output_lengths = ((lengths.long() * 24000 + sample_rate - 1) // sample_rate).clamp_min(1)
    return inputs, output_lengths


def prepare_ssl_inputs(batch: dict, device: torch.device | str, include_mixture: bool = True) -> dict:
    """Add the input features of both clean speakers (clean_ssl_inputs_0/1, from input_wav [B, 2, N]) and, with
    include_mixture, of the mixture (noisy_mixture_ssl_inputs, from noisy_mixture [B, N]) to a copy of the batch.

    Groups already present are moved to `device`; the missing ones are computed in one extraction call (they share
    the batch's lengths), on the model device rather than in data workers.
    """
    result = dict(batch)
    keys = ["clean_ssl_inputs_0", "clean_ssl_inputs_1"]
    if include_mixture:
        keys.append("noisy_mixture_ssl_inputs")
    missing = []
    for key in keys:
        if key in result:
            result[key] = {name: value.to(device) for name, value in result[key].items()}
        else:
            missing.append(key)
    if missing:
        sources = [batch["noisy_mixture"] if key == "noisy_mixture_ssl_inputs" else batch["input_wav"][:, int(key[-1])]
                   for key in missing]
        size = max(source.size(-1) for source in sources)
        stacked = torch.cat([F.pad(source, (0, size - source.size(-1))) for source in sources]).to(device)
        lengths = batch["input_wav_lens"].to(device)
        inputs, _ = extract_features(stacked, lengths.repeat(len(missing)), int(batch["sr"]))
        count = len(lengths)
        for j, key in enumerate(missing):
            result[key] = {name: value[j * count:(j + 1) * count] for name, value in inputs.items()}
    return result
