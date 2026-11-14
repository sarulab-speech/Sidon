"""Tiny DialogueSidon2 models, batches and shards for CPU tests (no downloads, no GPU)."""

import math
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

TINY_W2V = dict(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                output_hidden_size=16, feature_projection_input_dim=160, conv_depthwise_kernel_size=3,
                position_embeddings_type="relative_key", layerdrop=0.0, apply_spec_augment=False)
LATENT = 4


def tiny_config(**overrides):
    from transformers import Wav2Vec2BertConfig

    return Wav2Vec2BertConfig(**{**TINY_W2V, **overrides})


def vae_cfg(target_layer: int = 2) -> dict:
    return {"ssl_model_name": "facebook/w2v-bert-2.0", "target_layer": target_layer, "input_dim": 16,
            "hidden_dim": 8, "latent_dim": LATENT, "decoder_channels": 16, "decoder_rates": [8, 5, 4, 3],
            "hf_config": tiny_config(num_hidden_layers=target_layer).to_dict()}


def separation_cfg(layers: int = 3) -> dict:
    return {"vae": vae_cfg(), "sample_posterior": True,
            "conditioner": {"name": "facebook/w2v-bert-2.0", "layers": layers, "gradient_checkpointing": False,
                            "hf_config": tiny_config(num_hidden_layers=layers).to_dict()},
            "readout": {"hidden": 12}, "efficient_ssl_attention": True, "performance": {},
            "loss": {"assignment_weight": 1.0},
            "optim": {"fused": None, "lr": 1e-4, "conditioner_lr": 2e-5, "betas": [0.9, 0.95], "weight_decay": 0.0,
                      "warmup_steps": 2000, "cooldown_start": 160000, "min_lr_ratio": 0.0, "max_steps": 200000,
                      "max_grad_norm": 1.0, "skip_grad_norm": None},
            "vae_checkpoint": None, "statistics_path": None}


def stage_cfg() -> dict:
    return {"vae": vae_cfg(), "efficient_ssl_attention": True, "performance": {}, "kl_weight": 1e-5,
            "decoder_crop_frames": 60, "loss_weights": {"mel": 15.0, "adv_gen": 2.0, "adv_feature": 1.0},
            "scheduler_gamma": 0.999996,
            "mel": {"n_mels": [5, 10, 20, 40, 80, 160, 320], "window_lengths": [32, 64, 128, 256, 512, 1024, 2048],
                    "mel_fmin": [0] * 7, "mel_fmax": [None] * 7, "pow": 1.0, "clamp_eps": 1e-5, "mag_weight": 0.0},
            "optim": {"fused": None, "lr": 1e-4, "betas": [0.8, 0.98], "weight_decay": 0.01}}


class FakeDiscriminator(nn.Module):
    """A one-scale stand-in for dac's Discriminator: [[feature, logits]] per scale."""

    def __init__(self, **kwargs):
        super().__init__()
        self.first = nn.Conv1d(1, 2, 5, padding=2)
        self.second = nn.Conv1d(2, 1, 3, padding=1)

    def forward(self, x):
        feature = torch.tanh(self.first(x))
        return [[feature, self.second(feature)]]


@contextmanager
def tiny_hub():
    """Wav2Vec2BertModel.from_pretrained builds the tiny config with random weights."""
    from transformers import Wav2Vec2BertModel

    def model(name=None, **kwargs):
        overrides = {k: v for k, v in kwargs.items() if k in ("num_hidden_layers", "layerdrop", "apply_spec_augment")}
        return Wav2Vec2BertModel(tiny_config(**overrides)).eval()

    with patch.object(Wav2Vec2BertModel, "from_pretrained", side_effect=model):
        yield


@contextmanager
def fake_discriminator():
    with patch("dac.model.discriminator.Discriminator", FakeDiscriminator):
        yield


def tiny_separator(seed: int = 0, statistics: bool = True):
    from sidon.model.dialoguesidon2.lightning_module import DialogueSidon2

    torch.manual_seed(seed)
    model = DialogueSidon2(separation_cfg(), initialize_pretrained=False)
    if statistics:
        generator = torch.Generator().manual_seed(seed + 1)
        model.vae.mean.copy_(0.1 * torch.randn(LATENT, generator=generator))
        model.vae.var.copy_(0.5 + torch.rand(LATENT, generator=generator))
        model.vae.stats_ready.fill_(True)
    return model


def tiny_stage(seed: int = 0):
    from sidon.model.dialoguesidon2.lightning_module import SSLVAEModule

    torch.manual_seed(seed)
    with fake_discriminator():
        return SSLVAEModule(stage_cfg(), initialize_pretrained=False)


def dialogue_batch(seconds=(2.0, 1.5), seed: int = 0) -> dict:
    """input_wav [B, 2, N] (zero padded), noisy_mixture [B, N], input_wav_lens [B] at 24 kHz."""
    generator = torch.Generator().manual_seed(seed)
    lengths = [int(round(s * 50)) * 480 for s in seconds]
    size = max(lengths)
    wav = torch.zeros(len(lengths), 2, size)
    for i, n in enumerate(lengths):
        t = torch.arange(n) / 24000
        for s in range(2):
            tone = 0.3 * torch.sin(2 * math.pi * (200 + 90 * s) * t)
            wav[i, s, :n] = tone + 0.02 * torch.randn(n, generator=generator)
    mixture = wav.sum(1) + 0.01 * torch.randn(len(lengths), size, generator=generator)
    for i, n in enumerate(lengths):
        mixture[i, n:] = 0
    return {"input_wav": wav, "input_wav_lens": torch.tensor(lengths), "noisy_mixture": mixture, "sr": 24000,
            "names": [f"example/{i}" for i in range(len(lengths))]}


def write_dialogue_shards(root: Path, shards: int = 2, records: int = 3, seed: int = 0) -> None:
    import webdataset as wds

    generator = torch.Generator().manual_seed(seed)
    for split, count in (("train", shards), ("valid", 1)):
        (root / split).mkdir(parents=True, exist_ok=True)
        for s in range(count):
            with wds.TarWriter(str(root / split / f"shard-{s:04d}.tar")) as sink:
                for r in range(records):
                    n = 24000 * 2 + int(24000 * float(torch.rand((), generator=generator)))
                    wav = 0.2 * torch.randn(1, 2, n, generator=generator)
                    sink.write({"__key__": f"{split}-{s}-{r}", "input_wav.pth": wav, "sr.index": 24000,
                                "noisy_mixture.pth": wav.sum(1), "input_wav_lens.pth": torch.tensor([n])})


def write_sidon_shards(root: Path, shards: int = 2, records: int = 4, seed: int = 0) -> None:
    import webdataset as wds

    generator = torch.Generator().manual_seed(seed)
    for split, count in (("train", shards), ("valid", 1)):
        (root / split).mkdir(parents=True, exist_ok=True)
        for s in range(count):
            with wds.TarWriter(str(root / split / f"shard-{s:04d}.tar")) as sink:
                for r in range(records):
                    n = int(48000 * (1.1 + 2 * float(torch.rand((), generator=generator))))
                    wav = 0.2 * torch.randn(1, n, generator=generator)
                    sink.write({"__key__": f"{split}-{s}-{r}", "input_wav.pth": wav,
                                "input_wav_lens.pth": torch.tensor([n]), "sr.index": 48000})
