import pathlib
from functools import partial
from typing import List, Optional, Sequence

import torch
import torchaudio
import webdataset as wds
from lightning.pytorch import LightningDataModule
from transformers import AutoFeatureExtractor

from .datamodule import random_crop, torch_audio
from .functional_degrations import (
    add_non_parametric_noise,
    band_limit,
    clip,
    codec,
    convolve_rir,
    convolve_rir_pra,
    packet_loss,
    random_apply,
)


@torch.inference_mode()
def resample(sample, target_sr, input_key: str, output_key: str):
    x, sr = sample[input_key]
    if sr != target_sr:
        x = torchaudio.functional.resample(x, sr, target_sr)
    sample[output_key] = (x.view(1, -1), target_sr)
    return sample


def rename_audio(sample, output_key: str, input_key: Optional[str] = None):
    if input_key is None:
        audio_key = [k for k in sample.keys() if "audio" in k][0]
    else:
        audio_key = input_key
    sample[output_key] = sample[audio_key]
    return sample


@torch.inference_mode()
def lowcut(sample, input_key: str, cutoff=50):
    wav, sr = sample[input_key]
    wav = torchaudio.functional.highpass_biquad(wav, sr, cutoff)
    new_sample = sample.copy()
    new_sample[input_key] = (wav.view(1, -1), sr)
    return new_sample


def glob_wds(paths: Sequence[str]) -> List[str]:
    wds_paths = []
    for path in paths:
        wds_paths.extend(list(map(str, pathlib.Path(path).glob("**/*.tar.gz"))))
    for path in paths:
        wds_paths.extend(list(map(str, pathlib.Path(path).glob("**/*.tar"))))
    return wds_paths


def get_urls(path: pathlib.Path) -> List[str]:
    if isinstance(path, str):
        path = pathlib.Path(path)
    with path.open("r") as f:
        urls = f.read().splitlines()
    urls = [
        f"pipe:aws --endpoint-url https://s3ds.mdx.jp s3 cp {url} -" for url in urls
    ]
    return urls


def normalize(sample, input_key: str, output_key: str):
    wav, sr = sample[input_key]
    wav = (wav / wav.abs().max() + 1e-7) * 0.9
    new_sample = sample.copy()
    new_sample[output_key] = (wav, sr)
    return new_sample


def skip_nan(samples):
    for sample in samples:
        if torch.isnan(sample["clean"][0]).any():
            continue
        if torch.isnan(sample["noisy"][0]).any():
            continue
        if "noisy_16k" in sample.keys():
            if torch.isnan(sample["noisy_16k"][0]).any():
                continue
        if "clean_16k" in sample.keys():
            if torch.isnan(sample["clean_16k"][0]).any():
                continue
        yield sample


def merge_samples_to_target_length(samples, seconds, audio_keys):
    output_samples = []
    for sample in samples:
        first_key = audio_keys[0]
        output_samples.append(sample)
        total_duration = sum(
            [x[first_key][0].shape[-1] / x[first_key][1] for x in output_samples]
        )
        if total_duration >= seconds:
            outputs = dict()
            for key in audio_keys:
                sr = output_samples[0][key][1]
                outputs[key] = (
                    torch.cat(
                        [x[key][0] for x in output_samples],
                        dim=-1,
                    )[:, : int(seconds * sr)],
                    sr,
                )
            outputs["__key__"] = output_samples[0]["__key__"]
            output_samples = []
            yield outputs


class WebDatasetDataModule(LightningDataModule):
    def __init__(
        self,
        train_wds_patterns: str | list[str],
        val_wds_patterns: str | list[str],
        batch_size: int,
        sampling_rate: int,
        ssl_model_name: str,
        force_max_length: bool = True,
        noise_path: Sequence[str] = [""],
        use_noise: bool = True,
        speaker_ssl_model_name: Optional[str] = None,
        max_duration: float = 20.0,
        merge_samples: bool = False,
        n_repeats: int = -1,
        train_num_workers: int = 8,
        val_num_workers: int = 0,
        use_pra=False,
        split_by_worker=False,
    ):
        super().__init__()
        self.train_wds_patterns = glob_wds(train_wds_patterns)
        self.val_wds_patterns = glob_wds(val_wds_patterns)
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.processor = AutoFeatureExtractor.from_pretrained(ssl_model_name)
        self.use_speaker = False
        if speaker_ssl_model_name is not None:
            self.speaker_processor = AutoFeatureExtractor.from_pretrained(
                speaker_ssl_model_name
            )
            self.use_speaker = True
        self.force_max_length = force_max_length
        self.use_noise = use_noise
        self.noise_urls = glob_wds(noise_path)
        self.merge_samples = merge_samples
        self.n_repeats = n_repeats
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.use_pra = use_pra
        # WebDataset expects a callable splitter; accept a bool for convenience.
        self.split_by_worker = split_by_worker
        self.workersplitter = wds.split_by_worker if split_by_worker else False

    def train_dataloader(self) -> wds.WebLoader:
        def identity(x):
            return x[0]

        return wds.WebLoader(
            self.train_dataset,
            num_workers=self.train_num_workers,
            collate_fn=identity,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = (
            wds.WebDataset(
                self.train_wds_patterns,
                shardshuffle=True,
                nodesplitter=lambda x: x,
                workersplitter=self.workersplitter,
                repeat=True,
                empty_check=True,
            )
            .repeat(self.n_repeats)
            .decode(wds.autodecode.basichandlers, torch_audio)
            .map(partial(rename_audio, output_key="audio"))
            .map(partial(lowcut, input_key="audio", cutoff=50))
            .compose(
                partial(
                    random_crop,
                    n_crops=30,
                    seconds=self.max_duration,
                    input_key="audio",
                )
            )
            .map(partial(normalize, input_key="audio", output_key="audio"))
            .shuffle(100)
            .map(partial(rename_audio, input_key="audio", output_key="clean"))
            .map(partial(rename_audio, input_key="audio", output_key="noisy"))
        )
        self.val_dataset = (
            wds.WebDataset(
                self.val_wds_patterns,
                shardshuffle=True,
                nodesplitter=lambda x: x,
                workersplitter=self.workersplitter,
                repeat=True,
                empty_check=True,
            )
            .repeat(self.n_repeats)
            .decode(wds.autodecode.basichandlers, torch_audio)
            .map(partial(rename_audio, output_key="audio"))
            .map(partial(lowcut, input_key="audio", cutoff=50))
            .compose(
                partial(
                    random_crop,
                    n_crops=30,
                    seconds=self.max_duration,
                    input_key="audio",
                )
            )
            .map(partial(normalize, input_key="audio", output_key="audio"))
            .map(partial(rename_audio, input_key="audio", output_key="clean"))
            .map(partial(rename_audio, input_key="audio", output_key="noisy"))
        )
        if self.use_noise:
            self.train_dataset = self.add_noise_pipeline(self.train_dataset)
            self.val_dataset = self.add_noise_pipeline(self.val_dataset)
        self.train_dataset = self.add_resample_pipeline(self.train_dataset)
        self.train_dataset = (
            self.train_dataset.compose(skip_nan)
            .map(partial(normalize, input_key="clean", output_key="clean"))
            .map(partial(normalize, input_key="noisy", output_key="noisy"))
            .map(partial(normalize, input_key="clean_16k", output_key="clean_16k"))
            .map(partial(normalize, input_key="noisy_16k", output_key="noisy_16k"))
        )
        if self.merge_samples:
            self.train_dataset = self.train_dataset.compose(
                partial(
                    merge_samples_to_target_length,
                    seconds=self.max_duration,
                    audio_keys=["clean", "noisy", "clean_16k", "noisy_16k"],
                )
            )
        self.train_dataset = self.train_dataset.compose(skip_nan).batched(
            self.batch_size, collation_fn=self.collate_fn
        )
        self.val_dataset = self.add_resample_pipeline(self.val_dataset)
        self.val_dataset = (
            self.val_dataset.compose(skip_nan)
            .map(partial(normalize, input_key="clean", output_key="clean"))
            .map(partial(normalize, input_key="noisy", output_key="noisy"))
            .map(partial(normalize, input_key="clean_16k", output_key="clean_16k"))
            .map(partial(normalize, input_key="noisy_16k", output_key="noisy_16k"))
        )
        if self.merge_samples:
            self.val_dataset = self.val_dataset.compose(
                partial(
                    merge_samples_to_target_length,
                    seconds=self.max_duration,
                    audio_keys=["clean", "noisy", "clean_16k", "noisy_16k"],
                )
            )
        self.val_dataset = self.val_dataset.compose(skip_nan).batched(
            1, collation_fn=self.collate_fn
        )

    def add_resample_pipeline(self, dataset: wds.WebDataset):
        dataset = (
            dataset.map(
                partial(
                    resample,
                    target_sr=self.sampling_rate,
                    input_key="clean",
                    output_key="clean",
                ),
            )
            .map(
                partial(
                    resample,
                    target_sr=self.sampling_rate,
                    input_key="noisy",
                    output_key="noisy",
                ),
            )
            .map(
                partial(
                    resample,
                    target_sr=16_000,
                    input_key="clean",
                    output_key="clean_16k",
                )
            )
            .map(
                partial(
                    resample,
                    target_sr=16_000,
                    input_key="noisy",
                    output_key="noisy_16k",
                )
            )
        )
        return dataset

    def add_noise_pipeline(self, dataset: wds.WebDataset):
        noise_ds = (
            wds.WebDataset(
                self.noise_urls,
                shardshuffle=True,
                nodesplitter=lambda x: x,  # no split
                workersplitter=lambda x: x,  # no split
                repeat=True,
                empty_check=True,
            )
            .decode(torch_audio)
            .compose(
                partial(
                    random_crop,
                    n_crops=30,
                    seconds=self.max_duration,
                )
            )
            .shuffle(10)
            .repeat()
        )
        dataset = (
            dataset.compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=convolve_rir_pra,
                    input_key="clean",
                    direct_key="clean",
                    reverb_key="noisy",
                )
            )
            .compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=add_non_parametric_noise,
                    input_key="noisy",
                    output_key="noisy",
                    noise_ds=iter(noise_ds),
                )
            )
            .compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=band_limit,
                    candidate_srs=[8000, 16000, 22050, 24000, 44100, 48000],
                    output_key="noisy",
                    input_key="noisy",
                )
            )
            .compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=clip,
                    input_key="noisy",
                    output_key="noisy",
                )
            )
            .compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=codec,
                    codec_effectors=[
                        torchaudio.io.AudioEffector(
                            format="mp3",
                            codec_config=torchaudio.io.CodecConfig(qscale=10),
                        ),
                        torchaudio.io.AudioEffector(
                            format="mp3",
                            codec_config=torchaudio.io.CodecConfig(qscale=8),
                        ),
                        torchaudio.io.AudioEffector(
                            format="mp3",
                            codec_config=torchaudio.io.CodecConfig(qscale=4),
                        ),
                        torchaudio.io.AudioEffector(
                            format="mp3",
                            codec_config=torchaudio.io.CodecConfig(qscale=2),
                        ),
                        torchaudio.io.AudioEffector(
                            format="mp3",
                            codec_config=torchaudio.io.CodecConfig(qscale=1),
                        ),
                    ],
                    input_key="noisy",
                    output_key="noisy",
                )
            )
            .compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=packet_loss,
                    input_key="noisy",
                    output_key="noisy",
                )
            )
        )
        return dataset

    def val_dataloader(self) -> wds.WebLoader:
        def identity(x):
            return x[0]

        return wds.WebLoader(
            self.val_dataset,
            num_workers=self.val_num_workers,
            collate_fn=identity,
            pin_memory=False,
            drop_last=True,
        )

    @torch.inference_mode()
    def collate_fn(
        self,
        samples: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor | int | list[torch.Tensor]]:
        """Collate samples into a batch."""
        clean_wavs = torch.nn.utils.rnn.pad_sequence(
            [x["clean"][0].squeeze() for x in samples],
            batch_first=True,
        ).float()
        noisy_wavs = torch.nn.utils.rnn.pad_sequence(
            [x["noisy"][0].squeeze() for x in samples],
            batch_first=True,
        ).float()
        clean_wav_lens = torch.tensor(
            [x["clean"][0].size(-1) for x in samples],
            dtype=torch.long,
        )
        clean_16k_wavs = torch.nn.utils.rnn.pad_sequence(
            [x["clean_16k"][0].squeeze() for x in samples],
            batch_first=True,
        ).float()
        noisy_16k_wavs = torch.nn.utils.rnn.pad_sequence(
            [x["noisy_16k"][0].squeeze() for x in samples],
            batch_first=True,
        ).float()
        ssl_inputs = self.processor(
            [
                torch.nn.functional.pad(x["clean_16k"][0].squeeze(), (40, 40)).numpy()
                for x in samples
            ],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        noisy_ssl_inputs = self.processor(
            [
                torch.nn.functional.pad(x["noisy_16k"][0].squeeze(), (40, 40)).numpy()
                for x in samples
            ],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        # pad to max length
        if self.force_max_length:
            max_length = int(self.max_duration * self.sampling_rate)
            clean_wavs = torch.nn.functional.pad(
                clean_wavs,
                (0, max_length - clean_wavs.size(-1)),
            )

        sample = {
            "input_wav": clean_wavs,
            "input_wav_lens": clean_wav_lens,
            "sr": self.sampling_rate,
            "ssl_inputs": ssl_inputs,
            "names": [x["__key__"] for x in samples],
            "sample_rate": [x["clean"][1] for x in samples],
        }

        if self.use_noise:
            sample["noisy_ssl_inputs"] = noisy_ssl_inputs
            sample["noisy_input_wav16k"] = noisy_16k_wavs
            sample["noisy_input_wav"] = noisy_wavs
            if self.force_max_length:
                max_length = int(self.max_duration * self.sampling_rate)
                noisy_wavs = torch.nn.functional.pad(
                    noisy_wavs,
                    (0, max_length - noisy_wavs.size(-1)),
                )
        if self.use_speaker:
            noisy_speaker_ssl_inputs = self.speaker_processor(
                noisy_16k_wavs.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                max_length=int(self.max_duration * 16000),
                padding="max_length" if self.force_max_length else "longest",
            )
            sample["noisy_speaker_ssl_inputs"] = noisy_speaker_ssl_inputs

        return sample


class PreprocessedDataModule(LightningDataModule):
    def __init__(
        self,
        train_urls: Sequence[str] | str,
        val_urls: Sequence[str] | str,
        batch_size: int,
        train_num_workers: int = 0,
        val_num_workers: int = 0,
        preprocessed: bool = True,
        is_s3: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        if is_s3:
            self.train_urls = get_urls(train_urls)
            self.val_urls = get_urls(val_urls)
        else:
            self.train_urls = glob_wds(train_urls)
            self.val_urls = glob_wds(val_urls)
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers

    def setup(self, stage: str = "fit") -> None:
        self.train_dataset = (
            wds.WebDataset(
                self.train_urls,
                shardshuffle=True,
                nodesplitter=lambda x: x,  # no split
                workersplitter=True,  # no split
                repeat=True,
                empty_check=True,
                handler=wds.warn_and_continue,
            )
            .decode(
                wds.autodecode.basichandlers, torch_audio, handler=wds.warn_and_continue
            )
            .shuffle(1000)
            .batched(self.batch_size, collation_fn=self.collate_fn)
        )
        self.val_dataset = (
            wds.WebDataset(
                self.val_urls,
                shardshuffle=True,
                nodesplitter=lambda x: x,  # no split
                workersplitter=True,  # no split
                repeat=True,
                empty_check=True,
                handler=wds.warn_and_continue,
            )
            .decode(
                wds.autodecode.basichandlers, torch_audio, handler=wds.warn_and_continue
            )
            .batched(8, collation_fn=self.collate_fn)
        )

    def train_dataloader(self) -> wds.WebLoader:
        def identity(x):
            return x[0]

        return wds.WebLoader(
            self.train_dataset,
            num_workers=self.train_num_workers,
            collate_fn=identity,
            drop_last=True,
        )

    def val_dataloader(self) -> wds.WebLoader:
        def identity(x):
            return x[0]

        return wds.WebLoader(
            self.val_dataset,
            num_workers=self.val_num_workers,
            collate_fn=identity,
            drop_last=True,
        )

    def collate_fn(
        self,
        samples: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor | int | list[torch.Tensor]]:
        output_sample = {
            "input_wav": torch.stack(
                [sample["input_wav.pth"].view(-1) for sample in samples]
            ),
            "noisy_input_wav": torch.stack(
                [sample["noisy_input_wav.pth"].view(-1) for sample in samples]
            ),
            "input_wav_lens": torch.tensor(
                [sample["input_wav.pth"].view(-1).size(-1) for sample in samples],
                dtype=torch.long,
            ),
            "sr": samples[0]["sr.index"],
            "names": [x["__key__"] for x in samples],
        }
        ssl_inputs_dict = {}

        for k, v in samples[0]["ssl_inputs.pickle"].items():
            ssl_inputs_dict[k] = []

        for sample in samples:
            if "ssl_inputs.pickle" in sample:
                ssl_inputs = sample["ssl_inputs.pickle"]
                for key, value in ssl_inputs.items():
                    ssl_inputs_dict[key].append(value[0])
        for key, value in ssl_inputs_dict.items():
            ssl_inputs_dict[key] = torch.stack(value)
        noisy_ssl_inputs_dict = {}
        if "noisy_ssl_inputs.pickle" in samples[0]:
            for k, v in samples[0]["noisy_ssl_inputs.pickle"].items():
                noisy_ssl_inputs_dict[k] = []

            for sample in samples:
                if "noisy_ssl_inputs.pickle" in sample:
                    noisy_ssl_inputs = sample["noisy_ssl_inputs.pickle"]
                    for key, value in noisy_ssl_inputs.items():
                        noisy_ssl_inputs_dict[key].append(value[0])
            for key, value in noisy_ssl_inputs_dict.items():
                noisy_ssl_inputs_dict[key] = torch.stack(value)
        output_sample["ssl_inputs"] = ssl_inputs_dict
        output_sample["noisy_ssl_inputs"] = noisy_ssl_inputs_dict

        return output_sample
