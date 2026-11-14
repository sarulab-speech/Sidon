"""Single-speaker training data of the SSL-VAE stage, from Sidon's preprocessed shards.

Sidon's preprocessing (config/data/preprocessed*.yaml) writes one clean utterance crop per record:
input_wav.pth [1, N] float32 at sr.index Hz (48 kHz or 24 kHz) and input_wav_lens.pth [1]. The VAE stage treats
the two tracks of a [B, 2, N] batch independently (one posterior, one decoder crop and one KL term per track), so
two segments of different utterances stacked as two tracks train it exactly as two single-track examples do.

Each record's clean track is resampled to 24 kHz and cut into consecutive segments of segment_seconds (a shorter
remainder is dropped). Segments are paired (never two of one utterance) after a seeded shuffle in training, in
reading order in validation, and collated as input_wav [B, 2, N], input_wav_lens [B], sr 24000, names, and
noisy_mixture [B, N] (the mean of the two tracks, only used to log an input in validation).
"""

import random
from functools import partial

import torch
import torchaudio
import webdataset as wds
from lightning import LightningDataModule

from sidon.dialoguesidon2.data import expand_urls, launch_identity, sample_seed, shard_seed
from sidon.model.dialoguesidon2.functional import finite_or_raise

SAMPLE_RATE = 24000
HOP = 480  # samples per SSL frame at 24 kHz
KEEP = {"__key__", "__url__", "input_wav.pth", "input_wav_lens.pth", "sr.index"}


def clean_members(record: dict) -> dict:
    """The members segments() reads (module level, so spawned loader workers can pickle it)."""
    return {key: value for key, value in record.items() if key in KEEP}


def segments(records, seconds: float):
    """Records -> (name, index, [N] 24 kHz float32 segment) of exactly `seconds`, a multiple of HOP samples."""
    size = int(round(seconds * SAMPLE_RATE)) // HOP * HOP
    if size <= 0:
        raise ValueError("segment_seconds must cover at least one SSL frame")
    for record in records:
        rate = int(record["sr.index"])
        wav = torch.as_tensor(record["input_wav.pth"]).float()
        if wav.ndim == 2:
            if wav.size(0) != 1:
                raise ValueError("Sidon records hold one track: input_wav [1,N]")
            wav = wav[0]
        if wav.ndim != 1:
            raise ValueError("input_wav must have shape [1,N] or [N]")
        length = int(torch.as_tensor(record.get("input_wav_lens.pth", wav.numel())).reshape(-1)[0])
        wav = wav[:length]
        if rate != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, rate, SAMPLE_RATE)
        finite_or_raise(wav, "clean audio")
        name = str(record.get("__key__", "unknown"))
        for index in range(wav.numel() // size):
            yield name, index, wav[index * size:(index + 1) * size].contiguous()


def pair_segments(items, seed: int | None):
    """Segments -> pairs of segments of different utterances. With a seed (training) the first of each pair is
    drawn at random from a buffer of 16 segments; without one (validation) the reading order is kept."""
    rng = random.Random(seed) if seed is not None else None
    buffer = []

    def drain(limit: int | None):
        """Pairs from the buffer: at most `limit` (None: until no two utterances are left)."""
        made = 0
        while len(buffer) >= 2 and (limit is None or made < limit):
            first = buffer.pop(rng.randrange(len(buffer)) if rng else 0)
            partner = next((i for i, other in enumerate(buffer) if other[0] != first[0]), None)
            if partner is None:
                buffer.insert(0, first)
                return
            yield first, buffer.pop(partner)
            made += 1

    for item in items:
        buffer.append(item)
        if rng is None:
            yield from drain(None)
        elif len(buffer) >= 16:
            yield from drain(1)
    # The end of a pass (a worker's shards): what is left is paired too.
    yield from drain(None)


def collate_pairs(pairs) -> dict:
    """[(first, second)] of (name, index, wav) -> input_wav [B, 2, N], input_wav_lens [B], sr, names, and
    noisy_mixture [B, N], the mean of the two tracks."""
    waves = [torch.stack((first[2], second[2])) for first, second in pairs]
    if len({wav.size(-1) for wav in waves}) != 1:
        raise ValueError("Segments of one batch must share their length")
    wav = torch.stack(waves)
    return {"input_wav": wav, "input_wav_lens": torch.full((len(waves),), waves[0].size(-1)),
            "noisy_mixture": wav.mean(1), "sr": SAMPLE_RATE,
            "names": [f"{a[0]}#{a[1]}+{b[0]}#{b[1]}" for a, b in pairs]}


class SidonTrackDataModule(LightningDataModule):
    """Data module of the SSL-VAE stage. Directories expand to every *.tar under them (sorted);
    expected_train_shards (optional) refuses another count."""

    def __init__(self, train_urls: list[str], val_urls: list[str], batch_size: int = 4, num_workers: int = 0,
                 segment_seconds: float = 10.0, val_batch_size: int | None = None, data_seed: str | None = None,
                 expected_train_shards: int | None = None):
        super().__init__()
        self.train_urls = expand_urls(train_urls)
        self.val_urls = expand_urls(val_urls)
        if expected_train_shards is not None and len(self.train_urls) != int(expected_train_shards):
            raise ValueError(f"Found {len(self.train_urls)} training shards, expected {expected_train_shards}")
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.num_workers = num_workers
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        self.segment_seconds = float(segment_seconds)
        self.data_seed = None if data_seed is None or str(data_seed) == "" else str(data_seed)

    def loader(self, urls: list[str], training: bool):
        if not urls:
            raise ValueError("Set data.datamodule.train_urls and val_urls to Sidon's shard directories")
        pair_seed = None
        if training:
            step = self.trainer.global_step if self.trainer is not None else 0
            seed = shard_seed(step, self.data_seed)
            source, launch = launch_identity(self.data_seed)
            rank = wds.utils.pytorch_worker_info()[0]
            print(f"[sidon-tracks] rank={rank} launch_source={source} launch={launch} shard_seed={seed} "
                  f"shards={len(urls)}", flush=True)
            shuffle = dict(shardshuffle=100, detshuffle=True, seed=seed)
            pair_seed = sample_seed(step, "pairs", self.data_seed) or seed
        else:
            shuffle = dict(shardshuffle=False)
        dataset = wds.WebDataset(urls, **shuffle, nodesplitter=wds.split_by_node,
                                 workersplitter=wds.split_by_worker, empty_check=True)
        # Only the clean track is decoded; the other members are dropped undecoded.
        dataset = dataset.map(clean_members)
        dataset = dataset.decode(wds.autodecode.basichandlers)
        if training:
            dataset = dataset.shuffle(8, seed=sample_seed(step, "records", self.data_seed))
        dataset = dataset.compose(partial(segments, seconds=self.segment_seconds))
        dataset = dataset.compose(partial(pair_segments, seed=pair_seed))
        dataset = dataset.batched(self.batch_size if training else self.val_batch_size, collation_fn=collate_pairs,
                                  partial=not training)
        dataset = dataset.repeat()
        workers = self.num_workers
        if not training:
            rank, world_size, _, _ = wds.utils.pytorch_worker_info()
            workers = min(workers, len(range(rank, len(urls), world_size)))
        options = {}
        if workers > 0:
            options = dict(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
        return wds.WebLoader(dataset, batch_size=None, num_workers=workers,
                             pin_memory=torch.cuda.is_available(), **options)

    def train_dataloader(self):
        return self.loader(self.train_urls, True)

    def val_dataloader(self):
        return self.loader(self.val_urls, False)
