import random

import numpy as np
import torch
import torchaudio


# align two waveforms by padding the shorter one based on mse
def align_codec_waveform(original: torch.Tensor, codec_applied: torch.Tensor):
    if original.size(1) == codec_applied.size(1):
        return codec_applied
    else:
        # find an index which minimizes the mse
        if original.size(1) < codec_applied.size(1):
            return codec_applied[:, : original.size(1)]
        mse = []
        for i in range(original.size(1) - codec_applied.size(1) + 1):
            mse.append(
                torch.mean(
                    (original[:, i : i + codec_applied.size(1)] - codec_applied) ** 2
                )
            )
        start_index = torch.argmin(torch.tensor(mse)).item()
        return codec_applied[:, start_index : start_index + original.size(1)]


class DegrationApply:
    def __init__(self, noise_source, rir_source):
        self.codec_effector = [
            # torchaudio.io.AudioEffector(format="wav"),
            # torchaudio.io.AudioEffector(format="wav",encoder="pcm_mulaw"),
            torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=1)
            ),
            torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=2)
            ),
            torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=3)
            ),
            torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=4)
            ),
            torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=5)
            ),
            torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=6)
            ),
            torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=7)
            ),
            torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=8)
            ),
            torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=9)
            ),
            torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=10)
            ),
            #            torchaudio.io.AudioEffector(
            #                format="ogg",
            #                encoder="vorbis",
            #            ),
            #            torchaudio.io.AudioEffector(
            #                format="ogg",
            #                encoder="opus",
            #            ),
            #            torchaudio.io.AudioEffector(format="ogg"),
            #           torchaudio.io.AudioEffector(format="ogg", encoder='vorbis'),
            #           torchaudio.io.AudioEffector(format="ogg", encoder='opus'),
            #           torchaudio.io.AudioEffector(format="webm", encoder='opus'),
        ]
        self.downsample_sr = [8000, 16000, 22050, 24_000, 44100, 48000]
        self.noise_ds = iter(noise_source)
        self.rir_ds = iter(rir_source)

    def packet_loss(self, x: torch.Tensor, sr):
        # Randomly mask 10% of the audio with chunks of length 0.1 to 0.5 seconds
        total_duration = x.size(1) / sr
        num_chunks = int(total_duration * 3 / 10)  # 10% of the total duration

        for _ in range(num_chunks):
            chunk_duration_ms = random.uniform(100, 500)  # 0.1 to 0.5 seconds
            start_time = random.uniform(0, total_duration - chunk_duration_ms / 1000)
            start_sample = int(start_time * sr)
            end_sample = int((start_time + chunk_duration_ms / 1000) * sr)
            x[:, start_sample:end_sample] = 0

        return x

    def add_non_parametric_noise(self, x: torch.Tensor, sr: int):
        sample = next(self.noise_ds)
        audio_key = [k for k in sample.keys() if "audio" in k][0]
        noise, noise_sr = sample[audio_key]
        noise = noise[0].unsqueeze(0)

        noise = torchaudio.functional.resample(noise, noise_sr, sr)
        # repeat noise to the same length as x
        noise = noise.repeat(1, x.size(1) // noise.size(1) + 1)[:, : x.size(1)]
        x = torchaudio.functional.add_noise(
            x, noise, snr=torch.tensor(random.uniform(-5, 20)).unsqueeze(0)
        )
        return x

    def convolve_rir(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        sample = next(self.rir_ds)
        audio_key = [k for k in sample.keys() if "audio" in k][0]
        rir, rir_sr = sample[audio_key]
        rir = rir[random.randint(0, rir.shape[0] - 1)].unsqueeze(0)
        x = x.view(1, -1)
        rir = rir.view(1, -1)
        rir = torchaudio.functional.resample(rir, rir_sr, sr)
        rir = rir / rir.abs().max()
        return torchaudio.functional.fftconvolve(x, rir)[:, : x.size(1)]

    def band_limit(self, x: torch.Tensor, sr: int):
        target_sr = random.sample(self.downsample_sr, 1)[0]
        down_sampled = torchaudio.functional.resample(x, sr, target_sr)
        return torchaudio.functional.resample(down_sampled, target_sr, sr)

    def codec(self, x: torch.Tensor, sr: int):
        effector = random.sample(self.codec_effector, 1)[0]
        try:
            codec_applied = effector.apply(x.T, sr).T
        except TypeError:
            return x
        # _, codec_applied = align_waveform(x.view(1, -1), codec_applied.view(1, -1))
        codec_applied = align_codec_waveform(x, codec_applied=codec_applied)
        return codec_applied

    def clip(self, x: torch.Tensor):
        original_shape = x.shape
        min_q = random.uniform(0.0, 0.1)
        max_q = random.uniform(0.9, 1.0)
        x = torch.tensor(
            clipping(x.view(1, -1).numpy(), min_quantile=min_q, max_quantile=max_q)
        )
        return x.view(*original_shape)

    @torch.inference_mode()
    def apply(self, sample):
        audio_key = [k for k in sample.keys() if "audio" in k][0]
        new_sample = sample.copy()
        x, sr = sample[audio_key]
        x = x.clone()
        original_shape = x.size()
        x = x.view(1, -1)
        if random.random() > 0.5:
            x = self.convolve_rir(x, sr)
        if random.random() > 0.5:
            x = self.add_non_parametric_noise(x, sr)
        if random.random() > 0.5:
            x = self.band_limit(x, sr)
        if random.random() > 0.5:
            x = self.clip(x)
        if random.random() > 0.5:
            x = self.codec(x, sr)
        if random.random() > 0.5:
            x = self.packet_loss(x, sr)
        x = x.view(-1)
        x = torch.nan_to_num(x)
        new_sample["noisy"] = (x, sr)
        return new_sample


def clipping(speech_sample, min_quantile: float = 0.0, max_quantile: float = 0.9):
    """Apply the clipping distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        min_quantile (float): lower bound on the quantile of samples to be clipped
        max_quantile (float): upper bound on the quantile of samples to be clipped

    Returns:
        ret (np.ndarray): clipped speech sample (1, Time)
    """
    q = np.array([min_quantile, max_quantile])
    min_, max_ = np.quantile(speech_sample, q, axis=-1, keepdims=False)
    # per-channel clipping
    ret = np.stack(
        [
            np.clip(speech_sample[i], min_[i], max_[i])
            for i in range(speech_sample.shape[0])
        ],
        axis=0,
    )
    return ret
