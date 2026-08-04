"""Data module implementations for Sidon."""
from __future__ import annotations

import importlib
import io
import os
import random
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
from urllib.parse import urlsplit

import torch
import torchaudio
from lightning.pytorch import LightningDataModule

wds = importlib.import_module("webdataset")


def torch_audio(key: str, data: bytes):
    """Decode common audio formats using torchaudio."""
    extension = re.sub(r".*[.]", "", key)
    if extension not in {"flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"}:
        return None
    buffer = io.BytesIO(data)
    buffer.seek(0)
    return torchaudio.load(buffer, backend="soundfile")


def glob_wds(paths: Sequence[str] | str) -> list[str]:
    """Materialise webdataset shards from a glob or directory list."""
    if isinstance(paths, str):
        paths = [paths]
    shard_paths: list[str] = []
    for path in paths:
        shard_paths.extend(map(str, Path(path).glob("**/*.tar.gz")))
        shard_paths.extend(map(str, Path(path).glob("**/*.tar")))
    return shard_paths


def get_urls(path: Sequence[str] | str) -> list[str]:
    """Expand a list of S3 URIs stored in a text file into aws cli pipe URLs."""
    if isinstance(path, str):
        path = [path]
    urls: list[str] = []
    for entry in path:
        with Path(entry).open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle.read().splitlines(), start=1):
                uri = line.strip()
                parsed = urlsplit(uri)
                if (
                    parsed.scheme != "s3"
                    or not parsed.netloc
                    or parsed.path in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                    or any(ord(character) < 32 or ord(character) == 127 for character in uri)
                ):
                    raise ValueError(
                        f"invalid S3 URI in {entry} at line {line_number}: {uri!r}"
                    )
                urls.append(
                    "pipe:aws --endpoint-url https://s3ds.mdx.jp s3 cp "
                    f"{shlex.quote(uri)} -"
                )
    return urls


def random_crop(samples, n_crops, seconds, input_key=None):
    for sample in samples:
        if input_key is None:
            audio_key = [k for k in sample.keys() if "audio" in k][0]
        else:
            audio_key = input_key
        wav, sr = sample[audio_key]
        if wav.shape[0] > 1:
            wav = wav[0, None, :]  # if stereo, take only one channel
        wav = wav.view(1, -1)
        duration = wav.size(1) / sr
        n_crops = int(max(min(duration / n_crops, n_crops), 1))
        for i in range(n_crops):
            start = random.randint(  # noqa: S311
                0,
                max(0, wav.size(1) - int(sr * seconds)),
            )
            cropped = wav[
                :,
                start : start + round(sr * seconds),
            ].squeeze(0)
            new_sample = sample.copy()
            new_sample[audio_key] = (cropped, sr)
            yield new_sample


def _contains_nan(value: Any) -> bool:
    """Recursively inspect tensors nested in mappings/sequences for NaNs."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or torch.is_complex(value):
            return bool(torch.isnan(value).any().item())
        return False
    if isinstance(value, dict):
        return any(_contains_nan(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nan(v) for v in value)
    return False


def skip_nan_preprocessed(samples: Iterable[Dict[str, Any]]):
    """Drop samples with NaNs in any tensor payload."""
    for sample in samples:
        if _contains_nan(sample):
            continue
        yield sample


class PreprocessedDataModule(LightningDataModule):
    """Loads preprocessed torch tensors packaged as WebDataset shards."""

    def __init__(
        self,
        train_urls: Sequence[str] | str,
        val_urls: Sequence[str] | str,
        batch_size: int,
        val_batch_size: int,
        train_num_workers: int = 0,
        val_num_workers: int = 0,
        preprocessed: bool = True,
        is_s3: bool = False,
    ) -> None:
        super().__init__()
        del preprocessed  # maintained for backwards-compatible signature
        if is_s3:
            self.train_urls = get_urls(train_urls)
            self.val_urls = get_urls(val_urls)
        else:
            self.train_urls = glob_wds(train_urls)
            self.val_urls = glob_wds(val_urls)
        self.batch_size = batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.val_batch_size = val_batch_size
    @property
    def get_shuffle_buffer_size(self):
        return 1000

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = (
            wds.WebDataset(
                self.train_urls,
                shardshuffle=True,
                nodesplitter=lambda x: x,
                workersplitter=wds.split_by_worker,
                repeat=True,
                empty_check=True,
                handler=wds.warn_and_continue,
            )
            .decode(
                wds.autodecode.basichandlers,
                torch_audio,
                handler=wds.warn_and_continue,
            )
            .compose(skip_nan_preprocessed)
            .shuffle(self.get_shuffle_buffer_size)
            .batched(self.batch_size, collation_fn=self.collate_fn)
        )
        self.val_dataset = (
            wds.WebDataset(
                self.val_urls,
                shardshuffle=True,
                nodesplitter=lambda x: x,
                workersplitter=wds.split_by_worker,
                repeat=True,
                empty_check=True,
                handler=wds.warn_and_continue,
            )
            .decode(
                wds.autodecode.basichandlers,
                torch_audio,
                handler=wds.warn_and_continue,
            )
            .compose(skip_nan_preprocessed)
            .batched(self.val_batch_size, collation_fn=self.collate_fn)
        )

    def train_dataloader(self) -> Any:
        return wds.WebLoader(
            self.train_dataset,
            num_workers=self.train_num_workers,
            collate_fn=lambda batch: batch[0],
            pin_memory=True,
            persistent_workers=self.train_num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> Any:
        return wds.WebLoader(
            self.val_dataset,
            num_workers=self.val_num_workers,
            collate_fn=lambda batch: batch[0],
            pin_memory=True,
            persistent_workers=self.val_num_workers > 0,
            drop_last=True,
        )

    def collate_fn(self, samples: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble pickled SSL features and wave tensors into a batch."""
        output_sample: Dict[str, Any] = {
            "input_wav": torch.stack(
                [sample["input_wav.pth"].view(-1) for sample in samples]
            ),
            "noisy_input_wav": torch.stack(
                [sample["noisy_input_wav.pth"].view(-1) for sample in samples]
            ),
            "input_wav_lens": torch.tensor(
                [sample["input_wav.pth"].numel() for sample in samples],
                dtype=torch.long,
            ),
            "sr": samples[0]["sr.index"],
            "names": [sample["__key__"] for sample in samples],
        }

        ssl_inputs: Dict[str, list[torch.Tensor]] = {}
        if "ssl_inputs.pickle" in samples[0]:
            for key in samples[0]["ssl_inputs.pickle"].keys():
                ssl_inputs[key] = []
            for sample in samples:
                tensors = sample["ssl_inputs.pickle"]
                for key, value in tensors.items():
                    ssl_inputs[key].append(value[0])
            output_sample["ssl_inputs"] = {
                key: torch.stack(stack) for key, stack in ssl_inputs.items()
            }

        if "noisy_ssl_inputs.pickle" in samples[0]:
            noisy_ssl_inputs: Dict[str, list[torch.Tensor]] = {
                key: [] for key in samples[0]["noisy_ssl_inputs.pickle"].keys()
            }
            for sample in samples:
                tensors = sample["noisy_ssl_inputs.pickle"]
                for key, value in tensors.items():
                    noisy_ssl_inputs[key].append(value[0])
            output_sample["noisy_ssl_inputs"] = {
                key: torch.stack(stack) for key, stack in noisy_ssl_inputs.items()
            }
        else:
            output_sample["noisy_ssl_inputs"] = {}

        return output_sample
