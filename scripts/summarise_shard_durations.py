"""Summarise total audio duration stored in WebDataset shards."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Iterable, Optional

import webdataset as wds

from sidon.data import torch_audio


def iter_shards(root: Path) -> Iterable[Path]:
    """Yield all .tar/.tar.gz shards underneath the given directory."""
    if root.is_file() and (root.suffix == ".tar" or root.name.endswith(".tar.gz")):
        yield root
        return
    for suffix in ("**/*.tar", "**/*.tar.gz"):
        yield from root.glob(suffix)


def load_waveform(sample, audio_key: str):
    if audio_key not in sample:
        raise KeyError(f"Sample missing key '{audio_key}'")
    waveform = sample[audio_key]
    if isinstance(waveform, tuple) and len(waveform) == 2:
        return waveform
    return waveform, None


def accumulate_duration(
    shard: Path,
    audio_key: str,
    sample_rate_key: Optional[str],
    default_sample_rate: Optional[float],
) -> float:
    """Return total duration in seconds for all audio entries in `shard`."""
    dataset = (
        wds.WebDataset(
            str(shard),
            shardshuffle=False,
            nodesplitter=lambda urls: urls,
            handler=wds.warn_and_continue,
        )
        .decode(wds.autodecode.basichandlers, torch_audio, handler=wds.warn_and_continue)
    )
    seconds = 0.0
    for sample in dataset:
        try:
            waveform, sample_rate = load_waveform(sample, audio_key)
        except KeyError:
            continue
        if sample_rate is None:
            if sample_rate_key and sample_rate_key in sample:
                sample_rate = sample[sample_rate_key]
            else:
                sample_rate = default_sample_rate
        if sample_rate is None:
            raise ValueError(
                f"Unable to determine sample rate for {shard}; set --sample-rate-key or --default-sample-rate"
            )
        if hasattr(waveform, "ndim") and waveform.ndim > 1:
            waveform = waveform.reshape(-1)
        seconds += float(waveform.numel()) / float(sample_rate)
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=str, help="Path to a shard or directory of shards")
    parser.add_argument(
        "--audio-key",
        type=str,
        default="input_wav.pth",
        help="Key of the audio tensor inside the WebDataset sample",
    )
    parser.add_argument(
        "--sample-rate-key",
        type=str,
        default="sr.index",
        help="Key holding the sample rate if the audio tensor lacks one",
    )
    parser.add_argument(
        "--default-sample-rate",
        type=float,
        default=None,
        help="Fallback sample rate when the key is missing",
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    total = 0.0
    for shard in iter_shards(root):
        try:
            shard_seconds = accumulate_duration(
                shard,
                audio_key=args.audio_key,
                sample_rate_key=args.sample_rate_key,
                default_sample_rate=args.default_sample_rate,
            )
        except Exception as exc:  # pragma: no cover - surface offending shard
            print(f"Failed to read {shard}: {exc}")
            continue
        print(f"{shard}: {shard_seconds / 3600:.2f} h")
        total += shard_seconds

    print("---")
    print(f"Total duration: {total / 3600:.2f} h")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
