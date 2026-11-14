"""Speech restoration of per-speaker tracks with the public Sidon v0.1 model (sarulab-speech/sidon-v0.1).

The training data of DialogueSidon2 was restored track by track with this procedure (CallHome, Fisher, OpenYap-1k
and Basis; the simulated LibriTTS-R conversations are not restored). Per channel of an input file:

1. peak-normalise to 0.9, high-pass at 50 Hz (biquad), resample to 16 kHz and append 0.5 s of silence;
2. cut into chunks of 96 s; per chunk, 80-bin Kaldi filterbank features (25 ms windows, 10 ms hop, no dither,
   pre-emphasis 0.97, Povey window) normalised per bin, padded with 160 samples on both sides and stacked in pairs
   of frames (the feature layout of facebook/w2v-bert-2.0);
3. run the TorchScript feature extractor and decoder; the last feature frame of a chunk is prepended to the next
   chunk's features, and the decoder's last 960 output samples are dropped;
4. concatenate the chunks and cut to the input length at 48 kHz.

The output is a 48 kHz FLAC with the input's channels, quantised to 16 bits as round(x * 32768) clipped to the
int16 range. Inputs are read with soundfile (16-bit PCM as x / 32768).

Files: <input_dir>/**/*.flac (--pattern) -> <output_dir>/<input_dir.stem>/<relative path>.flac. Existing outputs
are overwritten; the pipelines pass a tree holding only the tracks that still need restoration (common/restore.py).

  python scripts/dialoguesidon2/data/common/sidon_restore.py --input_dir TREE/openyap-1k --output_dir RESTORED \\
      --num_shards 8 --shard_idx 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

MODEL_REPO = "sarulab-speech/sidon-v0.1"
MODEL_REVISION = "89624152d11da2d86b61dbb43d5b0ccc72b2cd9b"  # the snapshot used for the released data
MODEL_FILES = {"feature_extractor_cuda.pt": "c3739c332da4ce00cc9019a328bcb19ee8f863ac5eb0b52386672772aab836ce",
               "decoder_cuda.pt": "a789c71f54ca7db4b28becdc87c5e8b9260cb568cca462d502c54900db74a1cc"}
TARGET_SAMPLE_RATE = 48_000
CHUNK_SIZE = 16_000 * 96
RESAMPLE_RATE = 16_000
POST_PAD = 24_000
FEATURE_PAD = (160, 160)

fe = None
decoder = None


def load_models(device: str = "cuda", revision: str = MODEL_REVISION) -> None:
    """Load the two TorchScript modules once (cached in this module)."""
    global fe, decoder
    if fe is not None and decoder is not None:
        return
    from huggingface_hub import hf_hub_download
    model_device = torch.device(device)
    paths = {name: hf_hub_download(MODEL_REPO, filename=name, revision=revision) for name in MODEL_FILES}
    fe_module = torch.jit.load(paths["feature_extractor_cuda.pt"], map_location=model_device).to(model_device)
    decoder_module = torch.jit.load(paths["decoder_cuda.pt"], map_location=model_device).to(model_device)
    fe_module.eval()
    decoder_module.eval()
    fe, decoder = fe_module, decoder_module


def fbank_features(waveform: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Normalised 80-bin filterbank of one 16 kHz waveform, padded to an even frame count and stacked in pairs:
    (1, frames / 2, 160)."""
    torch_device = torch.device("cpu") if device == "cuda" and not torch.cuda.is_available() else torch.device(device)
    waveform = torch.as_tensor(waveform, dtype=torch.float32, device=torch_device)
    if waveform.ndim > 1:
        waveform = waveform[0]
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=waveform.unsqueeze(0), sample_frequency=16_000, num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, preemphasis_coefficient=0.97, remove_dc_offset=True, window_type="povey", use_energy=False,
        energy_floor=1.192092955078125e-07)
    mean = feature.mean(0, keepdim=True)
    var = feature.var(0, keepdim=True)
    feature = (feature - mean) / torch.sqrt(var + 1e-5)
    frames = (feature.shape[0] + 1) // 2 * 2
    padded = torch.zeros((1, frames, feature.shape[1]), dtype=torch.float32, device=feature.device)
    padded[0, :feature.shape[0]] = feature
    return padded.reshape(1, frames // 2, feature.shape[1] * 2)


def denoise_speech(waveform: torch.Tensor, sample_rate: int, device: str = "cuda") -> torch.Tensor:
    """Restore every channel of `waveform` ((channels, samples) float) and return (channels, samples at 48 kHz)."""
    load_models(device)
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    restored_wavs = []
    for index in range(waveform.shape[0]):
        channel = waveform[index].unsqueeze(0)
        max_val = channel.abs().max()
        if max_val > 0:
            channel = 0.9 * (channel / max_val)
        target_n_samples = int(TARGET_SAMPLE_RATE / sample_rate * channel.shape[-1])
        filtered = torchaudio.functional.highpass_biquad(channel, sample_rate, 50)
        resampled = torchaudio.functional.resample(filtered, sample_rate, RESAMPLE_RATE)
        resampled = torch.nn.functional.pad(resampled, (0, POST_PAD))
        segments = []
        feature_cache = None
        for chunk in resampled.view(-1).split(CHUNK_SIZE):
            features = fbank_features(torch.nn.functional.pad(chunk, FEATURE_PAD), device=device)
            with torch.inference_mode():
                feature = fe(features.to(device))["last_hidden_state"]
                if feature_cache is not None:
                    feature = torch.cat([feature_cache, feature], dim=1)
                restored_chunk = decoder(feature.transpose(1, 2)).view(-1)[:-960]
            segments.append(restored_chunk)
            feature_cache = feature[:, -1:]
        restored = torch.cat(segments, dim=0)
        restored_wavs.append(restored.view(1, -1)[:, :target_n_samples])
    return torch.cat(restored_wavs, dim=0).cpu()


def quantise(restored: torch.Tensor) -> np.ndarray:
    """(channels, samples) float -> (samples, channels) int16: round(x * 32768), clipped."""
    x = restored.detach().cpu().numpy().astype(np.float32)
    return np.clip(np.round(x * np.float32(32768)), -32768, 32767).astype(np.int16).T


def read_audio(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sample_rate)


def write_flac(path: Path, restored: torch.Tensor) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), quantise(restored), TARGET_SAMPLE_RATE, format="FLAC", subtype="PCM_16")


class AudioFileDataset(torch.utils.data.Dataset):
    """Loads the input files inside DataLoader workers."""

    def __init__(self, files):
        self.files = [Path(f) for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        waveform, sample_rate = read_audio(path)
        return path, waveform, sample_rate


def _single_item_collate(batch):
    return batch[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--revision", default=MODEL_REVISION, help="model revision on the Hugging Face Hub")
    parser.add_argument("--pattern", default="**/*.flac", help="glob of the input files below --input_dir")
    args = parser.parse_args(argv)
    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("shard_idx must be in [0, num_shards)")
    from tqdm import tqdm
    files = sorted(args.input_dir.glob(args.pattern))
    if args.num_shards > 1:
        files = [f for i, f in enumerate(files) if i % args.num_shards == args.shard_idx]
    dataset = AudioFileDataset(files)
    if len(dataset) == 0:
        print("No audio files to process for the selected shard.")
        return 0
    load_models(args.device, args.revision)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=_single_item_collate, persistent_workers=args.num_workers > 0)
    for path, waveform, sample_rate in tqdm(loader, total=len(dataset)):
        restored = denoise_speech(waveform, int(sample_rate), device=args.device)
        relative = path.relative_to(args.input_dir).with_suffix(".flac")
        write_flac(args.output_dir / args.input_dir.stem / relative, restored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
