"""Dialogue WebDataset loading for the separation model.

Each record of a shard holds one dialogue (at most 120 s at 24 kHz):
  input_wav.pth       clean speakers, float32 [2, N] (or [1, 2, N])
  noisy_mixture.pth   the degraded single-channel mixture, [N] (or [1, N])
  input_wav_lens.pth  valid length(s), [1]
  sr.index            24000
Other members are ignored. Records are cut into near-equal, frame-aligned segments of at most segment_seconds,
shuffled in bounded buffers and collated into batches: input_wav [B, 2, N], noisy_mixture [B, N] (zero padded),
input_wav_lens [B], sr, names. Input features are computed on the model device (sidon.dialoguesidon2.audio).

The training order (shards, records, segments) is a function of data_seed, the rank and the start step: runs with
one data_seed see the same order. Without data_seed every launch draws a new order.
"""

import math
import os
import zlib
from functools import partial
from pathlib import Path

import torch
import webdataset as wds
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence

from sidon.model.dialoguesidon2.functional import finite_or_raise, length_mask

MEMBERS = {"__key__", "__url__", "input_wav.pth", "input_wav_lens.pth", "noisy_mixture.pth", "sr.index"}


def launch_identity(data_seed=None) -> tuple[str, object]:
    """(source, value) of the training data-order identity: data_seed, else the process id."""
    if data_seed is not None and data_seed != "":
        return "data_seed", data_seed
    return "pid", os.getpid()


def shard_seed(step: int = 0, data_seed=None) -> int:
    """Shard-order seed, distinct per rank and start step (a resumed run reads a new order)."""
    rank = wds.utils.pytorch_worker_info()[0]
    launch = launch_identity(data_seed)[1]
    return zlib.crc32(f"{launch}:{rank}:{step}".encode())


def sample_seed(step: int, stage: str, data_seed=None) -> int | None:
    """Seed of a sample shuffle buffer, derived from data_seed; None without one (webdataset then seeds from the
    process and time)."""
    if data_seed is None or data_seed == "":
        return None
    return zlib.crc32(f"{shard_seed(step, data_seed)}:{stage}".encode())


def dialogue_members(record: dict) -> dict:
    """The members the loader reads; others (e.g. cached features) are dropped before decoding."""
    return {key: value for key, value in record.items() if key in MEMBERS}


def split_records(records, seconds: float):
    """Split each dialogue into ceil(frames / limit) contiguous, frame-aligned segments of near-equal length,
    limit = round(seconds * 50) frames of 480 samples: a 120 s record gives 4 segments of 30 s."""
    limit = max(1, round(seconds * 50))
    for record in records:
        if int(record["sr.index"]) != 24000:
            raise ValueError("DialogueSidon2 requires 24 kHz waveforms")
        wav = record["input_wav.pth"]
        if wav.ndim == 2:
            wav = wav[None]
        lens = record.get("input_wav_lens.pth", torch.full((wav.size(0),), wav.size(-1)))
        lens = torch.as_tensor(lens).reshape(-1).long()
        length_mask(lens, wav.size(-1))
        if (lens > 120 * 24000).any():
            raise ValueError("Dialogue exceeds 120 seconds; prepare bounded records")
        mixture = record.get("noisy_mixture.pth")
        if mixture is not None:
            if mixture.ndim == 1:
                mixture = mixture[None]
            elif mixture.ndim == 3 and mixture.size(1) == 1:
                mixture = mixture[:, 0]
        key = str(record.get("__key__", "unknown"))
        for i, length in enumerate(lens.tolist()):
            frames = -(-length // 480)
            count = -(-frames // limit)
            bounds = [min(k * frames // count * 480, length) for k in range(count + 1)]
            for start, end in zip(bounds, bounds[1:]):
                # Copies release the complete record while segments wait in the shuffle buffer.
                segment = {"__key__": f"{key}/{i}/{start}", "sr.index": 24000,
                           "input_wav.pth": wav[i, :, start:end].clone(),
                           "input_wav_lens.pth": torch.tensor([end - start])}
                if mixture is not None:
                    segment["noisy_mixture.pth"] = mixture[i, start:end].clone()
                yield segment


def collate_dialogues(records: list[dict]) -> dict:
    """Records (or segments) -> input_wav [B, 2, N], noisy_mixture [B, N], input_wav_lens [B], sr, names."""
    waves, lengths, mixtures, names = [], [], [], []
    for record in records:
        if int(record["sr.index"]) != 24000:
            raise ValueError("DialogueSidon2 requires 24 kHz waveforms")
        wav = record["input_wav.pth"]
        if wav.ndim == 2:
            wav = wav[None]
        if wav.ndim != 3 or wav.size(1) != 2:
            raise ValueError("input_wav must have shape [B,2,N] or [2,N]")
        lens = record.get("input_wav_lens.pth", torch.full((wav.size(0),), wav.size(-1)))
        lens = torch.as_tensor(lens).reshape(-1).long()
        if len(lens) != wav.size(0):
            raise ValueError("Waveform length count does not match record batch")
        length_mask(lens, wav.size(-1))
        if (lens > 120 * 24000).any():
            raise ValueError("Dialogue exceeds 120 seconds; prepare bounded records")
        mixture = record.get("noisy_mixture.pth")
        if mixture is None:
            raise ValueError(f"{record.get('__key__', 'record')} has no noisy_mixture.pth")
        if mixture.ndim == 1:
            mixture = mixture[None]
        elif mixture.ndim == 3 and mixture.size(1) == 1:
            mixture = mixture[:, 0]
        if mixture.ndim != 2 or mixture.size(0) != wav.size(0):
            raise ValueError("noisy_mixture must have shape [B,N] or [N]")
        length_mask(lens, mixture.size(-1))
        for i, length in enumerate(lens.tolist()):
            audio = wav[i, :, :length].float()
            finite_or_raise(audio, "clean audio")
            waves.append(audio.T)
            lengths.append(length)
            names.append(str(record.get("__key__", "unknown")))
            current_mixture = mixture[i, :length].float()
            finite_or_raise(current_mixture, "noisy mixture")
            mixtures.append(current_mixture)
    return {"input_wav": pad_sequence(waves, batch_first=True).transpose(1, 2),
            "input_wav_lens": torch.tensor(lengths), "sr": 24000, "names": names,
            "noisy_mixture": pad_sequence(mixtures, batch_first=True)}


def every_shard(urls):
    """A node splitter that gives every rank every shard (picklable for spawned workers)."""
    yield from urls


def expand_urls(entries: list[str]) -> list[str]:
    """Directories expand to every *.tar and *.tar.gz under them (sorted); files are kept as given."""
    urls = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            urls.extend(str(p) for p in sorted(path.rglob("*.tar")))
            urls.extend(str(p) for p in sorted(path.rglob("*.tar.gz")))
        elif path.is_file():
            urls.append(str(path))
        else:
            raise ValueError(f"Shard path does not exist: {entry}")
    return urls


class DialogueDataModule(LightningDataModule):
    """Training and validation dialogues, cut into segments of segment_seconds (null: complete records)."""

    def __init__(self, train_urls: list[str], val_urls: list[str], batch_size: int = 1, num_workers: int = 0,
                 segment_seconds: float | None = None, val_batch_size: int | None = None,
                 data_seed: str | None = None):
        super().__init__()
        self.train_urls = expand_urls(train_urls)
        self.val_urls = expand_urls(val_urls)
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.num_workers = num_workers
        if segment_seconds is not None and segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive or null")
        self.segment_seconds = segment_seconds
        self.data_seed = None if data_seed is None or str(data_seed) == "" else str(data_seed)

    expand = staticmethod(expand_urls)

    def loader(self, urls: list[str], training: bool, repeat: bool = True, segment_seconds: float | None = None):
        if not urls:
            raise ValueError("Set data.datamodule.train_urls and val_urls to prepared dialogue shards")
        if training:
            # detshuffle reseeds every pass with seed + pass index.
            step = self.trainer.global_step if self.trainer is not None else 0
            seed = shard_seed(step, self.data_seed)
            source, launch = launch_identity(self.data_seed)
            rank = wds.utils.pytorch_worker_info()[0]
            print(f"[data] rank={rank} launch_source={source} launch={launch} shard_seed={seed}", flush=True)
            shuffle = dict(shardshuffle=100, detshuffle=True, seed=seed)
        else:
            shuffle = dict(shardshuffle=False)
        nodesplitter = wds.split_by_node
        if not training and wds.utils.pytorch_worker_info()[1] > len(urls):
            # Fewer validation shards than ranks: split by node, some ranks would get none and wait forever in
            # validation's collectives. Every rank reads every shard instead (the logged mean is unchanged).
            nodesplitter = every_shard
        dataset = wds.WebDataset(urls, **shuffle, nodesplitter=nodesplitter, workersplitter=wds.split_by_worker,
                                 empty_check=True)
        dataset = dataset.map(dialogue_members)
        dataset = dataset.decode(wds.autodecode.basichandlers)
        if training:
            # A record can hold 120 s of audio; keep the buffer small.
            dataset = dataset.shuffle(4, seed=sample_seed(step, "records", self.data_seed))
        if segment_seconds is not None:
            dataset = dataset.compose(partial(split_records, seconds=segment_seconds))
            if training:
                # Mixes the segments of about four records.
                dataset = dataset.shuffle(4 * math.ceil(120 / segment_seconds),
                                          seed=sample_seed(step, "segments", self.data_seed))
        dataset = dataset.batched(self.batch_size if training else self.val_batch_size,
                                  collation_fn=collate_dialogues, partial=not training)
        if repeat:
            # Without repetition, ranks with fewer records would end a DDP epoch before the others.
            dataset = dataset.repeat()
        workers = self.num_workers
        if not training:
            rank, world_size, _, _ = wds.utils.pytorch_worker_info()
            shards = len(urls) if world_size > len(urls) else len(range(rank, len(urls), world_size))
            workers = min(workers, shards)
        options = {}
        if workers > 0:
            # CUDA and NCCL must not be inherited by forked workers; bounded prefetch (records are long).
            options = dict(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
        return wds.WebLoader(dataset, batch_size=None, num_workers=workers,
                             pin_memory=torch.cuda.is_available(), **options)

    def train_dataloader(self):
        return self.loader(self.train_urls, True, segment_seconds=self.segment_seconds)

    def val_dataloader(self):
        # Validation uses the training segment length, so both losses share one context length.
        return self.loader(self.val_urls, False, segment_seconds=self.segment_seconds)
