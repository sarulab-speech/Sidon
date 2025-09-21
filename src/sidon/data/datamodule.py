"""Data module implementations for Sidon."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torchaudio
import webdataset as wds
from lightning.pytorch import LightningDataModule


def torch_audio(key: str, data: bytes):
    """Decode common audio formats using torchaudio."""
    extension = re.sub(r".*[.]", "", key)
    if extension not in {"flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"}:
        return None

    with tempfile.TemporaryDirectory() as dirname:
        filename = os.path.join(dirname, f"file.{extension}")
        with open(filename, "wb") as stream:
            stream.write(data)
        return torchaudio.load(filename, backend="soundfile")


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
            for line in file_handle.read().splitlines():
                urls.append(
                    f"pipe:aws --endpoint-url https://s3ds.mdx.jp s3 cp {line} -"
                )
    return urls


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
            .shuffle(1000)
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
            .batched(self.val_batch_size, collation_fn=self.collate_fn)
        )

    def train_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.train_dataset,
            num_workers=self.train_num_workers,
            collate_fn=lambda batch: batch[0],
            drop_last=True,
        )

    def val_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.val_dataset,
            num_workers=self.val_num_workers,
            collate_fn=lambda batch: batch[0],
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
                key: torch.stack(stack)
                for key, stack in noisy_ssl_inputs.items()
            }
        else:
            output_sample["noisy_ssl_inputs"] = {}

        return output_sample
