"""DialogueProducerDataModule: the data pipeline that turns 120 s stereo source windows into training records.

Per source record (a stereo 48 kHz FLAC under the member `flac`, one speaker per channel):

1. decode, high-pass at 50 Hz (biquad), peak-normalise each channel to 0.9 and then the pair to 0.9;
2. cut into chunks of `max_duration` seconds (keys <source key>_<start sample>); drop chunks in which neither channel
   exceeds 0.2 in absolute value; pad or cut to `max_duration`;
3. per channel, each with probability 0.5 and in this order: reverberation (pyroomacoustics shoebox room, RT60 ~
   U(rt60_min, rt60_max); the clean target becomes the direct path), additive noise from the noise shards at SNR ~
   U(-5, 20) dB, band limitation (down to one of 8, 16, 22.05, 24, 44.1 or 48 kHz and back), quantile clipping
   (lower quantile U(0, 0.1), upper U(0.9, 1.0)), mp3 (qscale 10, 8, 4, 2 or 1) and packet loss; then pad or cut
   both "clean" and "noisy" to `max_duration` again;
4. join the channels, resample to `sampling_rate` and to 16 kHz, peak-normalise each signal (per channel, then
   jointly), drop records with NaNs;
5. collate (producer.mixing.collate): the targets, the degraded channels and the mixtures with r ~ U(0.3, 0.7).

Nothing is seeded: Python's `random` drives the degradations and the mixing ratio, so two runs give different
degradations of the same windows.

The data module reuses src/sidon/data/preprocess where the code is unchanged; the pieces that differ from it are
defined here (chunk keys, the quiet-chunk filter, the RT60 range, the degradation versions, the shard split and the
collate without SSL inputs).
"""

from __future__ import annotations

import random
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
import webdataset as wds

from sidon.data.datamodule import random_crop, torch_audio
from sidon.data.preprocess.webdataset_datamodule import (
    WebDatasetDataModule,
    lowcut,
    normalize,
    rename_audio,
    skip_nan,
)

from producer import degradations as dg
from producer import mixing

CODEC_QSCALES = (10, 8, 4, 2, 1)
BAND_LIMIT_RATES = [8000, 16000, 22050, 24000, 44100, 48000]


# ---- sample transforms -------------------------------------------------------------------------------------------

def chunk_crop(samples, input_key=None, chunk_seconds: float = 30):
    """Fixed-size chunks of each sample's audio, keyed <key>_<start sample>; the last chunk may be shorter."""
    for sample in samples:
        audio_key = [k for k in sample.keys() if "audio" in k][0] if input_key is None else input_key
        wav, sr = sample[audio_key]
        n_samples = wav.shape[1]
        step = int(chunk_seconds * sr)
        if step <= 0:
            yield sample
            continue
        if n_samples <= step:
            new_sample = sample.copy()
            new_sample["__key__"] = f"{sample['__key__']}_{0}"
            new_sample[audio_key] = (wav, sr)
            yield new_sample
            continue
        for i in range(0, n_samples, step):
            new_sample = sample.copy()
            new_sample["__key__"] = f"{sample['__key__']}_{i}"
            new_sample[audio_key] = (wav[:, i : i + step], sr)
            yield new_sample


def select_non_silent_chunk(sample, key: str, threshold: float = 0.2) -> bool:
    """Keep a chunk when at least one channel's peak exceeds `threshold`."""
    wav, _ = sample[key]
    return bool((wav.abs().max(dim=1).values > threshold).any())


def split_by_channel(sample, channel_idx: int, split_sample_key: str):
    wav, sr = sample[split_sample_key]
    new_sample = sample.copy()
    new_sample[split_sample_key] = (wav[channel_idx].unsqueeze(0), sr)
    return new_sample


def drop_key(sample, key_to_drop: str):
    sample.pop(key_to_drop)
    return sample


def pad_to_duration(sample, key, pad_to_duration: int):
    wav, sr = sample[key]
    wav_len = wav.shape[-1]
    if wav_len > pad_to_duration * sr:
        wav = wav[:, : pad_to_duration * sr]
    else:
        wav = torch.nn.functional.pad(wav, (0, int(pad_to_duration * sr - wav_len)))
    sample[key] = (wav, sr)
    return sample


def merge_dataset(sample: dict[str, Any], merge_keys: list[str], other_ds: Any, randomize_order: bool = True):
    """Join channel 0 (this sample) and channel 1 (the next sample of `other_ds`, the same window) into stereo.
    The channel order is always kept; the random draw is kept because it advances Python's RNG."""
    if randomize_order:
        random.random()
    other_sample = next(other_ds)
    if sample["__key__"] != other_sample["__key__"]:
        raise ValueError
    output_sample = sample.copy()
    for merge_key in merge_keys:
        wav1, sr = sample[merge_key]
        wav2, _ = other_sample[merge_key]
        output_sample[merge_key] = (torch.cat([wav1, wav2], dim=0), sr)
    return output_sample


# ---- data module -----------------------------------------------------------------------------------------------

class DialogueProducerDataModule(WebDatasetDataModule):
    """WebDatasetDataModule with the dialogue pipeline of the producer (module docstring).

    Extra arguments: n_shards / shard_idx split the training shards into tasks (np.array_split of the shard
    listing; the validation shards are not split), rt60_min / rt60_max the reverberation range, and degradations
    "legacy" (default, as the released data was made) or "upstream" (producer.degradations)."""

    def __init__(self, *args, n_shards: int = 1, shard_idx: int = 0, rt60_min: float = 0.1, rt60_max: float = 1.0,
                 degradations: str = "legacy", **kwargs):
        super().__init__(*args, **kwargs)
        self.train_wds_patterns = np.array_split(self.train_wds_patterns, n_shards)[shard_idx]
        self.n_shards = n_shards
        self.shard_idx = shard_idx
        self.rt60_min_max = (rt60_min, rt60_max)
        self.degradations = degradations
        self._functions = dg.functions(degradations)

    def _source(self, patterns):
        return (
            wds.WebDataset(patterns, shardshuffle=True, nodesplitter=lambda x: x, workersplitter=self.workersplitter,
                           repeat=True, empty_check=True)
            .repeat(self.n_repeats)
            .decode(wds.autodecode.basichandlers, wds.autodecode.torch_audio)
            .map(partial(rename_audio, input_key="flac", output_key="audio"))
            .map(partial(lowcut, input_key="audio", cutoff=50))
            .map(partial(normalize, input_key="audio", output_key="audio"))
            .compose(partial(chunk_crop, chunk_seconds=self.max_duration))
            .select(partial(select_non_silent_chunk, key="audio"))
            .map(partial(pad_to_duration, key="audio", pad_to_duration=self.max_duration))
            .map(partial(rename_audio, input_key="audio", output_key="clean"))
            .map(partial(rename_audio, input_key="audio", output_key="noisy"))
            .map(partial(drop_key, key_to_drop="flac"))
            .map(partial(drop_key, key_to_drop="audio"))
        )

    def _degraded_stereo(self, dataset):
        channels = []
        for channel in (0, 1):
            one = dataset.map(partial(split_by_channel, channel_idx=channel, split_sample_key="noisy")).map(
                partial(split_by_channel, channel_idx=channel, split_sample_key="clean"))
            one = self.add_noise_pipeline(one).map(
                partial(pad_to_duration, key="clean", pad_to_duration=self.max_duration)).map(
                partial(pad_to_duration, key="noisy", pad_to_duration=self.max_duration))
            channels.append(one)
        return channels[0].map(partial(merge_dataset, merge_keys=["clean", "noisy"], other_ds=iter(channels[1])),
                               handler=wds.handlers.warn_and_continue)

    def _finish(self, dataset, batch_size: int):
        dataset = self.add_resample_pipeline(dataset)
        dataset = (
            dataset.map(partial(normalize, input_key="clean", output_key="clean"))
            .map(partial(normalize, input_key="noisy", output_key="noisy"))
            .map(partial(normalize, input_key="clean_16k", output_key="clean_16k"))
            .map(partial(normalize, input_key="noisy_16k", output_key="noisy_16k"))
        ).compose(skip_nan)
        return dataset.batched(batch_size, collation_fn=self.collate_fn)

    def setup(self, stage=None):
        train = self._source(self.train_wds_patterns)
        val = self._source(self.val_wds_patterns)
        if self.use_noise:
            train = self._degraded_stereo(train)
            val = self._degraded_stereo(val)
        self.train_dataset = self._finish(train, self.batch_size)
        self.val_dataset = self._finish(val, 1)

    def add_noise_pipeline(self, dataset: wds.WebDataset):
        f = self._functions
        noise_ds = (
            wds.WebDataset(self.noise_urls, shardshuffle=True, nodesplitter=lambda x: x, workersplitter=lambda x: x,
                           repeat=True, empty_check=True)
            .decode(torch_audio)
            .compose(partial(random_crop, n_crops=30, seconds=self.max_duration))
            .shuffle(10)
            .repeat()
        )
        effectors = [torchaudio.io.AudioEffector(format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=q))
                     for q in CODEC_QSCALES]
        return (
            dataset.compose(partial(dg.random_apply, prob=0.5, transform_fn=dg.convolve_rir_pra, input_key="clean",
                                    direct_key="clean", reverb_key="noisy", rt60_min_max=self.rt60_min_max))
            .compose(partial(dg.random_apply, prob=0.5, transform_fn=dg.add_non_parametric_noise, input_key="noisy",
                             output_key="noisy", noise_ds=iter(noise_ds)))
            .compose(partial(dg.random_apply, prob=0.5, transform_fn=f["band_limit"], candidate_srs=BAND_LIMIT_RATES,
                             output_key="noisy", input_key="noisy"))
            .compose(partial(dg.random_apply, prob=0.5, transform_fn=dg.clip, input_key="noisy", output_key="noisy"))
            .compose(partial(dg.random_apply, prob=0.5, transform_fn=f["codec"], codec_effectors=effectors,
                             input_key="noisy", output_key="noisy"))
            .compose(partial(dg.random_apply, prob=0.5, transform_fn=f["packet_loss"], input_key="noisy",
                             output_key="noisy"))
        )

    @torch.inference_mode()
    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return mixing.collate(samples, self.sampling_rate, self.use_noise, self.max_duration, self.force_max_length)
