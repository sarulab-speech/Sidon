"""The SSL-VAE: w2v-BERT 2.0 truncated to one layer, a VAE bottleneck and a DAC decoder.

The encoder maps w2v-BERT 2.0 input features (log-mel, ``sidon.dialoguesidon2.audio``) through the first
``target_layer`` layers of w2v-BERT 2.0 (frozen) and a small MLP to the mean and log-variance of a ``latent_dim``
posterior per 20 ms frame. The decoder turns latents back into 24 kHz audio (480 samples per frame).

The separation model is trained to predict standardized latents: ``z`` drawn from the posterior, minus the
training-set mean, divided by sqrt(var + 1e-5) (``latent statistics``, ``sidon.dialoguesidon2.statistics``).
The decoder takes raw latents, so its input is de-normalized first (``decoder_input``).
"""

import hashlib
import math
import zlib
from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from .functional import finite_or_raise, length_mask

# Keys of the `vae` config section (hf_config is added when the w2v-BERT weights are first loaded).
VAE_KEYS = ("ssl_model_name", "target_layer", "input_dim", "hidden_dim", "latent_dim", "decoder_channels")
STATISTICS_BUFFERS = ("mean", "var", "stats_ready")


def stable_seed(*parts) -> int:
    """A seed from the parts, equal in every process and interpreter run (unlike hash())."""
    return zlib.crc32(":".join(map(str, parts)).encode())


def encode_unpadded(model: nn.Module, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Run a w2v-BERT model on each sequence's valid frames only, so padding never reaches the convolutions.

    Sequences of equal length share one call. Returns the padded [B, T, C] outputs and the [B] frame counts.
    """
    features = inputs["input_features"]
    masks = inputs["attention_mask"].bool()
    lengths = masks.sum(1)
    if not torch.equal(masks, length_mask(lengths, features.size(1))):
        raise ValueError("SSL masks must be contiguous valid prefixes")
    outputs = [None] * len(lengths)
    for length in lengths.unique().tolist():
        index = (lengths == length).nonzero().flatten()
        x = features[index, :length]
        finite_or_raise(x, "SSL input")
        # Every row is fully valid, so no attention mask is needed.
        out = model(input_features=x, attention_mask=None).last_hidden_state
        finite_or_raise(out, "SSL representation")
        for i, value in zip(index.tolist(), out):
            outputs[i] = value
    sizes = torch.tensor([x.size(0) for x in outputs], device=features.device)
    return pad_sequence(outputs, batch_first=True), sizes


class VAEBottleneck(nn.Module):
    """Linear-ReLU-Linear from the SSL features to 2 * latent_dim channels: the posterior mean, then the
    log-variance."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * latent_dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        return mu, logvar


class SSLVAE(nn.Module):
    """SSL-VAE encoder and decoder with the latent statistics the separation model is trained on.

    cfg: ssl_model_name, target_layer, input_dim (the w2v-BERT width), hidden_dim, latent_dim, decoder_channels,
    decoder_rates (default [8, 5, 4, 3], 480 samples per frame) and, once known, hf_config (the truncated w2v-BERT
    config, so the model can be rebuilt without the hub). initialize_pretrained loads w2v-BERT from the hub;
    otherwise it is built from hf_config and its weights come from a checkpoint. decoder=False builds the encoder
    only (latent statistics).

    Posterior draws z = mu + exp(logvar / 2) * eps take eps from the global Torch stream, or inside
    ``posterior_draws(*parts)`` from a generator seeded with stable_seed(*parts).
    """

    hop = 480

    def __init__(self, cfg: dict, initialize_pretrained: bool = True, decoder: bool = True):
        super().__init__()
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

        missing = [key for key in VAE_KEYS if key not in cfg]
        if missing:
            raise ValueError(f"The vae config needs {missing}")
        self.cfg = {key: deepcopy(value) for key, value in cfg.items()}
        for key in VAE_KEYS[1:]:
            self.cfg[key] = int(self.cfg[key])
        self.cfg["decoder_rates"] = [int(rate) for rate in self.cfg.get("decoder_rates", [8, 5, 4, 3])]
        if math.prod(self.cfg["decoder_rates"]) != self.hop:
            raise ValueError(f"Decoder rates {self.cfg['decoder_rates']} must multiply to {self.hop}")
        if initialize_pretrained:
            ssl = Wav2Vec2BertModel.from_pretrained(self.cfg["ssl_model_name"],
                                                    num_hidden_layers=self.cfg["target_layer"],
                                                    layerdrop=0.0, apply_spec_augment=False)
            self.cfg["hf_config"] = ssl.config.to_dict()
        else:
            if "hf_config" not in self.cfg:
                raise ValueError("Without pretrained weights the vae config needs hf_config")
            ssl = Wav2Vec2BertModel(Wav2Vec2BertConfig.from_dict(deepcopy(self.cfg["hf_config"])))
        if (ssl.config.num_hidden_layers, ssl.config.hidden_size) != (self.cfg["target_layer"], self.cfg["input_dim"]):
            raise ValueError("The w2v-BERT config does not match target_layer and input_dim")
        self.ssl_model = ssl
        self.bottleneck = VAEBottleneck(self.cfg["input_dim"], self.cfg["hidden_dim"], self.cfg["latent_dim"])
        self.ssl_model.requires_grad_(False).eval()
        self.register_buffer("mean", torch.zeros(self.dim))
        self.register_buffer("var", torch.ones(self.dim))
        self.register_buffer("stats_ready", torch.tensor(False))
        self.decoder = None
        if decoder:
            from dac.model.dac import Decoder

            self.decoder = Decoder(input_channel=self.dim, channels=self.cfg["decoder_channels"],
                                   rates=self.cfg["decoder_rates"])
        self.sample_posterior = True  # targets are posterior draws (False: the posterior mean)
        self._draws = None  # (seed, generator) inside posterior_draws

    @property
    def dim(self) -> int:
        return self.cfg["latent_dim"]

    def train(self, mode: bool = True):
        super().train(mode)
        self.ssl_model.eval()  # frozen: no dropout
        return self

    # ---- weights and statistics ----

    def load_weights(self, state: dict[str, Tensor]) -> None:
        """Load encoder and decoder weights strictly; the statistics buffers are loaded with load_statistics."""
        state = {key: value for key, value in state.items() if key not in STATISTICS_BUFFERS}
        expected = {key for key in self.state_dict() if key not in STATISTICS_BUFFERS}
        if set(state) != expected:
            missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
            raise ValueError(f"VAE weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
        current = self.state_dict()
        self.load_state_dict({**{key: current[key] for key in STATISTICS_BUFFERS}, **state}, strict=True)

    def identity(self) -> str:
        """SHA-256 of the encoder weights (w2v-BERT and bottleneck), which latent statistics are tied to."""
        state = {**{f"ssl_model.{k}": v for k, v in self.ssl_model.state_dict().items()},
                 **{f"bottleneck.{k}": v for k, v in self.bottleneck.state_dict().items()}}
        digest = hashlib.sha256()
        for name, value in sorted(state.items()):
            value = value.detach().cpu().contiguous()
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy())
        return digest.hexdigest()

    def load_statistics(self, path: str) -> None:
        """Latent statistics written by sidon.dialoguesidon2.statistics for this encoder."""
        stats = torch.load(path, map_location="cpu", weights_only=True)
        recorded = stats.get("encoder_identity")
        if recorded is not None and recorded != self.identity():
            raise ValueError(f"The statistics {path} were computed with other encoder weights")
        for name in ("mean", "var"):
            value = stats[name].float()
            finite_or_raise(value, name)
            if value.shape != (self.dim,) or (name == "var" and (value < 0).any()):
                raise ValueError("Invalid latent statistics")
            getattr(self, name).copy_(value)
        if stats["count"] <= 1:
            raise ValueError("Too few frames in statistics")
        self.stats_ready.fill_(True)

    # ---- encoder ----

    @torch.no_grad()
    def posterior(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """mu and logvar [N, T, latent_dim] (zero after each length) and the [N] frame counts."""
        features, lengths = encode_unpadded(self.ssl_model, inputs)
        mu, logvar = self.bottleneck(features)
        valid = length_mask(lengths, features.size(1))[..., None]
        return mu.masked_fill(~valid, 0), logvar.masked_fill(~valid, 0), lengths

    def trainable_posterior(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """posterior() with gradients through the bottleneck (VAE training); w2v-BERT runs without them."""
        with torch.no_grad():
            features, lengths = encode_unpadded(self.ssl_model, inputs)
        mu, logvar = self.bottleneck(features)
        valid = length_mask(lengths, features.size(1))[..., None]
        return mu.masked_fill(~valid, 0), logvar.masked_fill(~valid, 0), lengths

    def _generator(self, device: torch.device) -> torch.Generator | None:
        if self._draws is None:
            return None
        seed, generator = self._draws
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(seed)
            self._draws = (seed, generator)
        elif generator.device != device:
            raise RuntimeError("posterior_draws started on another device")
        return generator

    def sample(self, mu: Tensor, logvar: Tensor, lengths: Tensor) -> Tensor:
        """z = mu + exp(logvar / 2) * eps in float32, with gradients; one draw over the whole padded tensor, zero
        after each length."""
        eps = torch.randn(mu.shape, generator=self._generator(mu.device), device=mu.device, dtype=torch.float32)
        z = mu.float() + torch.exp(0.5 * logvar.float()) * eps
        return z.masked_fill(~length_mask(lengths, z.size(1))[..., None], 0)

    @torch.no_grad()
    def draw(self, mu: Tensor, logvar: Tensor, lengths: Tensor) -> Tensor:
        """The target latent of a posterior: a draw (sample_posterior) or the float32 mean."""
        if not self.sample_posterior:
            return mu.float()
        return self.sample(mu, logvar, lengths)

    @torch.no_grad()
    def raw(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Raw (not standardized) target latents [N, T, D] and frame counts."""
        mu, logvar, lengths = self.posterior(inputs)
        return self.draw(mu, logvar, lengths), lengths

    @torch.no_grad()
    def encode(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Standardized target latents [N, T, D] (zero after each length) and frame counts."""
        if not bool(self.stats_ready):
            raise RuntimeError("Load the latent statistics before encoding targets")
        z, lengths = self.raw(inputs)
        z = (z.float() - self.mean) * torch.rsqrt(self.var + 1e-5)
        return z.masked_fill(~length_mask(lengths, z.size(1))[:, :, None], 0), lengths

    @torch.no_grad()
    def posterior_mean(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """The raw posterior mean (float32) and frame counts; draws nothing."""
        mu, _, lengths = self.posterior(inputs)
        return mu.float(), lengths

    @contextmanager
    def posterior_draws(self, *parts):
        """Repeatable posterior draws: inside, eps comes from one generator seeded with stable_seed(*parts)."""
        if self._draws is not None:
            raise RuntimeError("posterior_draws does not nest")
        self._draws = (stable_seed(*parts), None)
        try:
            yield self
        finally:
            self._draws = None

    # ---- decoder ----

    def decoder_input(self, latent: Tensor) -> Tensor:
        """Standardized latents [..., D] -> the raw latents the decoder was trained on."""
        return latent * torch.sqrt(self.var + 1e-5) + self.mean

    def decode(self, latent: Tensor, frame_lengths: Tensor, waveform_lengths: Tensor,
               normalized: bool = True) -> Tensor:
        """[B, S, T, D] latents -> [B, S, N] waveforms, decoding only the valid frames of each example.
        normalized: the latents are standardized (de-normalized here) rather than raw."""
        b, speakers, _, _ = latent.shape
        if len(frame_lengths) != b or len(waveform_lengths) != b:
            raise ValueError("One pair of lengths is required per example")
        length_mask(frame_lengths, latent.size(2))
        if (waveform_lengths <= 0).any():
            raise ValueError("Waveform lengths must be positive")
        max_length = int(waveform_lengths.max())
        waves = []
        for i, (frames, samples) in enumerate(zip(frame_lengths.tolist(), waveform_lengths.tolist())):
            z = latent[i, :, :frames]
            if normalized:
                z = self.decoder_input(z)
            wav = self.decoder(z.transpose(1, 2)).squeeze(1)  # [S, D, T] -> [S, N]
            if abs(wav.size(-1) - samples) > self.hop:
                raise ValueError("SSL frame count and waveform duration disagree")
            wav = F.pad(wav[..., :samples], (0, max_length - min(samples, wav.size(-1))))
            waves.append(wav)
        output = torch.stack(waves)
        finite_or_raise(output, "decoded waveform")
        return output


def load_vae_checkpoint(path) -> tuple[dict, dict[str, Tensor]]:
    """The `vae` config and the SSLVAE state dict of a VAE-stage checkpoint (SSLVAEModule, Lightning format)."""
    # Lightning checkpoints hold more than tensors (loop and callback states), so weights_only=False:
    # only load checkpoints you trust.
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    cfg = checkpoint["hyper_parameters"]["cfg"]
    if "vae" not in cfg or "hf_config" not in cfg["vae"]:
        raise ValueError(f"{path} is not a VAE-stage checkpoint")
    state = {key[len("vae."):]: value for key, value in checkpoint["state_dict"].items() if key.startswith("vae.")}
    return deepcopy(dict(cfg["vae"])), state
