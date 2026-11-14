"""Lightning modules of the two training stages: the SSL-VAE (``SSLVAEModule``) and the separation model
(``DialogueSidon2``)."""

import os
from copy import deepcopy

import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .functional import finite_or_raise, length_mask
from .performance import adamw, compile_training_blocks
from .readout import ReadoutHead, split_speakers
from .ssl_attention import optimize_relative_key_attention
from .vae import SSLVAE, encode_unpadded, load_vae_checkpoint, stable_seed

__all__ = ["DialogueSidon2", "SSLVAEModule", "stable_seed"]

UNSET = (None, "", "???")


def as_dict(cfg) -> dict:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return deepcopy(dict(cfg))


def set_conditioner_finetune(conditioner: nn.Module) -> None:
    """Every conditioner weight trains, except masked_spec_embed: without SpecAugment nothing uses it, and DDP
    would wait for its gradient."""
    for name, parameter in conditioner.named_parameters():
        parameter.requires_grad_("masked_spec_embed" not in name)


def optimizer_groups(model: nn.Module, conditioner: nn.Module, conditioner_lr: float | None) -> list:
    """The trainable parameters; with conditioner_lr, the conditioner's weights form a second group at that rate
    and every other parameter (the readout head) keeps optim.lr."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if conditioner_lr is None:
        return trainable
    base = {id(p) for p in conditioner.parameters() if p.requires_grad}
    if not base:
        raise ValueError("optim.conditioner_lr needs trainable conditioner weights")
    return [{"params": [p for p in trainable if id(p) not in base]},
            {"params": [p for p in trainable if id(p) in base], "lr": float(conditioner_lr)}]


def learning_rate_factor(step: int, warmup_steps: int, cooldown_start: int | None, max_steps: int,
                         min_lr_ratio: float = 0.0) -> float:
    """Linear warmup to 1 over warmup_steps, constant, then (from cooldown_start) linear decay to min_lr_ratio at
    max_steps."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    if cooldown_start is not None and step >= cooldown_start:
        progress = (step - cooldown_start) / max(1, max_steps - cooldown_start)
        return max(min_lr_ratio, 1 - progress)
    return 1.0


def reseed_training(module: LightningModule) -> None:
    """Reseed Torch (CPU and CUDA) per rank and start step when training starts.

    seed_everything gives every rank the same stream, and a resumed run would replay it from the start; DDP has
    already broadcast rank 0's initial weights at this point.
    """
    torch.manual_seed(stable_seed(os.environ.get("PL_GLOBAL_SEED", 0), module.global_rank, module.global_step))


def safe_gradients(module: nn.Module, max_norm: float = 1.0,
                   max_abs_norm: float | None = 20.0) -> tuple[bool, Tensor]:
    """Clip the gradients of `module`'s trainable parameters to a total norm of max_norm, or drop them (skipping the
    update, Adam moments included) when the norm is not finite or exceeds max_abs_norm. The decision is the same
    on every DDP rank. Returns (updated, norm before clipping)."""
    parameters = [p for p in module.parameters() if p.requires_grad]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = torch.nn.utils.get_total_norm(grads) if grads else torch.zeros((), device=parameters[0].device)
    bad = (~torch.isfinite(norm) | ((norm > max_abs_norm) if max_abs_norm is not None else False)).int()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
    if bool(bad):
        for p in parameters:
            p.grad = None
        return False, norm
    # The same scaling as clip_grad_norm_(max_norm), reusing the norm computed above.
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, norm)
    return True, norm


def encode_speakers(encode, batch: dict, device) -> tuple[Tensor, Tensor]:
    """Both clean speakers through one encoder call: `encode` maps a feature group of [2B] rows to ([2B, T, D],
    [2B]); returns [B, 2, T, D] latents and the [B] frame counts."""
    from sidon.dialoguesidon2.audio import prepare_ssl_inputs

    batch = prepare_ssl_inputs(batch, device, include_mixture=False)
    first, second = batch["clean_ssl_inputs_0"], batch["clean_ssl_inputs_1"]
    if first["input_features"].shape != second["input_features"].shape:
        raise ValueError("Speaker SSL lengths differ")
    z, lengths = encode({name: torch.cat((first[name], second[name])) for name in first})
    count = first["input_features"].size(0)
    if not torch.equal(lengths[:count], lengths[count:]):
        raise ValueError("Speaker SSL lengths differ")
    return torch.stack((z[:count], z[count:]), 1), lengths[:count]


def posterior_speakers(vae: SSLVAE, batch: dict) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """VAE training: both clean speakers' posteriors in one call with gradients through the bottleneck, and one
    draw z = mu + exp(logvar / 2) * eps; [B, 2, T, D] z, mu and logvar and the [B] frame counts."""
    from sidon.dialoguesidon2.audio import prepare_ssl_inputs

    batch = prepare_ssl_inputs(batch, vae.mean.device, include_mixture=False)
    first, second = batch["clean_ssl_inputs_0"], batch["clean_ssl_inputs_1"]
    if first["input_features"].shape != second["input_features"].shape:
        raise ValueError("Speaker SSL lengths differ")
    mu, logvar, lengths = vae.trainable_posterior({name: torch.cat((first[name], second[name])) for name in first})
    count = first["input_features"].size(0)
    if not torch.equal(lengths[:count], lengths[count:]):
        raise ValueError("Speaker SSL lengths differ")
    z = vae.sample(mu, logvar, lengths)
    return tuple(torch.stack((value[:count], value[count:]), 1) for value in (z, mu, logvar)) + (lengths[:count],)


def kl_divergence(mu: Tensor, logvar: Tensor, lengths: Tensor) -> Tensor:
    """-0.5 * sum_d(1 + logvar - mu^2 - exp(logvar)) per frame, averaged over the valid frames of both speakers of
    [B, 2, T, D] posteriors with [B] frame counts, in float32."""
    mu, logvar = mu.float(), logvar.float()
    per_frame = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    valid = length_mask(lengths, mu.size(2))[:, None].expand_as(per_frame)
    return per_frame[valid].mean()


class SSLVAEModule(LightningModule):
    """Stage 1: train the SSL-VAE's bottleneck and decoder on clean speech (w2v-BERT stays frozen).

    The loss is a GAN (multi-scale DAC discriminator, least squares), multi-scale mel and feature matching loss on
    decoded crops of at most decoder_crop_frames frames, plus kl_weight times the KL term. Optimization is manual:
    per batch, one discriminator step, then one generator (bottleneck and decoder) step.
    """

    def __init__(self, cfg, initialize_pretrained: bool = True):
        super().__init__()
        self.cfg = as_dict(cfg)
        self.vae = SSLVAE(self.cfg["vae"], initialize_pretrained)
        self.cfg["vae"] = self.vae.cfg
        weight = self.cfg.setdefault("kl_weight", 1e-5)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not weight >= 0:
            raise ValueError(f"kl_weight must be a non-negative number, not {weight!r}")
        if self.cfg.get("efficient_ssl_attention", True):
            optimize_relative_key_attention(self.vae)
        from dac.model.discriminator import Discriminator
        from dac.nn.loss import MelSpectrogramLoss

        from sidon.model.losses import GANLoss

        self.discriminator = GANLoss(Discriminator(sample_rate=24000))
        self.mel = MelSpectrogramLoss(**self.cfg["mel"])
        self.automatic_optimization = False
        self.save_hyperparameters({"cfg": self.cfg}, logger=False)

    def generator_modules(self) -> nn.Module:
        """What the generator optimizer steps: the bottleneck, then the decoder (not registered as a submodule)."""
        return nn.ModuleList([self.vae.bottleneck, self.vae.decoder])

    def configure_optimizers(self):
        options = self.cfg["optim"]
        betas = tuple(options.get("betas", (0.8, 0.98)))
        optimizers = [adamw(module.parameters(), fused=options.get("fused"), lr=options["lr"], betas=betas,
                            weight_decay=options["weight_decay"])
                      for module in (self.generator_modules(), self.discriminator)]
        return [{"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.get("scheduler_gamma",
                                                                                             0.999996)),
            "name": f"train/lr_{role}"}} for optimizer, role in zip(optimizers, ("generator", "discriminator"))]

    def on_fit_start(self):
        torch.set_float32_matmul_precision("medium")
        compile_training_blocks(self)

    def on_train_start(self):
        # Otherwise every rank (and every resumed run) draws the same decoder crops.
        reseed_training(self)

    def reconstruction(self, batch: dict, training: bool) -> tuple[list[tuple[Tensor, Tensor, int]], Tensor]:
        """Decoded crops and their targets, grouped by crop size as (prediction [2n, 1, N], target [2n, 1, N], n),
        and the KL term of the posterior whose draw they decode."""
        latent, mu, logvar, frame_lengths = posterior_speakers(self.vae, batch)
        kl = kl_divergence(mu, logvar, frame_lengths)
        # The encoder sees the whole record; the decoder learns on aligned crops of at most decoder_crop_frames
        # (random in training, the start in validation).
        limit = int(self.cfg["decoder_crop_frames"])
        hop = self.vae.hop
        groups = {}
        for i, frames in enumerate(frame_lengths.tolist()):
            samples = int(batch["input_wav_lens"][i])
            count = min(limit, frames)
            start = int(torch.randint(frames - count + 1, ())) if training else 0
            end_sample = min((start + count) * hop, samples)
            groups.setdefault((count, end_sample - start * hop), []).append((i, start))
        pairs = []
        # Crops of one size share a decoder call; rows are example-major, speaker-minor.
        for (count, length), members in groups.items():
            prediction = self.vae.decoder(
                torch.cat([latent[i, :, start:start + count] for i, start in members]).transpose(1, 2))
            target = torch.cat([batch["input_wav"][i, :, start * hop:start * hop + length]
                                for i, start in members]).unsqueeze(1)
            size = min(length, prediction.size(-1))
            if size < 2048:
                raise ValueError("Training crops must contain at least 2048 samples")
            pairs.append((prediction[..., :size], target[..., :size], len(members)))
        return pairs, kl

    def step(self, batch: dict, training: bool) -> Tensor:
        from audiotools import AudioSignal

        stage = "train" if training else "val"
        pairs, kl = self.reconstruction(batch, training)
        examples = sum(count for _, _, count in pairs)

        def average(values):
            # Each group's loss is a mean over its rows; weighting by group size gives the per-example mean.
            return sum(value * count for value, (_, _, count) in zip(values, pairs)) / examples

        if training:
            generator_opt, discriminator_opt = self.optimizers()
            generator_scheduler, discriminator_scheduler = self.lr_schedulers()
            discriminator_opt.zero_grad(set_to_none=True)
        d_loss = average([self.discriminator.discriminator_loss(pred.detach(), target) for pred, target, _ in pairs])
        finite_or_raise(d_loss, "discriminator loss")
        if training:
            # DDP (find_unused_parameters off) expects a gradient for every trainable parameter in each backward;
            # this zero term reaches the decoder and the bottleneck without changing any value.
            ghost = sum(pred.sum() * 0 for pred, _, _ in pairs)
            self.manual_backward(d_loss + ghost, retain_graph=True)
            valid, _ = safe_gradients(self.discriminator, max_abs_norm=None)
            self.log("train/discriminator_skipped", float(not valid))
            if valid:
                discriminator_opt.step()
                discriminator_scheduler.step()
        losses = []
        weights = self.cfg.get("loss_weights", {"mel": 15.0, "adv_gen": 2.0, "adv_feature": 1.0})
        components = []
        for pred, target, _ in pairs:
            mel = self.mel(AudioSignal(pred, sample_rate=24000), AudioSignal(target, sample_rate=24000))
            adv, feature = self.discriminator.generator_loss(pred, target)
            components.append(torch.stack([torch.as_tensor(value, device=mel.device)
                                           for value in (mel, adv, feature)]))
            losses.append(weights["mel"] * mel + weights["adv_gen"] * adv + weights["adv_feature"] * feature)
        loss = average(losses)
        loss = loss + self.cfg["kl_weight"] * kl
        finite_or_raise(loss, "VAE loss")
        if training:
            generator_opt.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            # The adversarial terms also give the discriminator gradients; they must not set the generator's clip
            # scale or skip its update.
            valid, _ = safe_gradients(self.generator_modules(), max_abs_norm=None)
            self.log("train/generator_skipped", float(not valid))
            if valid:
                generator_opt.step()
                generator_scheduler.step()
        for name, value in zip(("mel_loss", "adv_gen", "adv_feature"), average(components)):
            self.log(f"{stage}/{name}", value, sync_dist=not training, batch_size=examples)
        self.log(f"{stage}/loss", loss, sync_dist=not training, batch_size=examples)
        self.log(f"{stage}/discriminator", d_loss, sync_dist=not training, batch_size=examples)
        self.log(f"{stage}/kl_loss", kl, sync_dist=not training, batch_size=examples)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, True)

    def validation_step(self, batch, batch_idx):
        # Validation decodes posterior draws as training does, with eps from a generator seeded per (rank, batch),
        # so validation passes stay comparable.
        with self.vae.posterior_draws("val_posterior", self.global_rank, batch_idx):
            return self.step(batch, False)


class DialogueSidon2(LightningModule):
    """Stage 2: the separation model.

    The noisy mixture's input features go through a fully fine-tuned w2v-BERT 2.0 truncated to
    conditioner.layers layers (the conditioner); the readout head maps each frame to both speakers' standardized
    SSL-VAE latents. Training minimizes the permutation-invariant MSE against the latents of the clean speakers
    (posterior draws of the frozen SSL-VAE, standardized with the latent statistics). Inference de-normalizes the
    prediction and decodes it with the SSL-VAE decoder: one forward pass, no sampling.

    cfg (config/model/dialoguesidon2.yaml): vae_checkpoint and statistics_path (stage 1 outputs, read when
    initialize_pretrained), conditioner {name, layers, gradient_checkpointing}, readout {hidden}, sample_posterior,
    efficient_ssl_attention, performance, loss {assignment_weight}, optim. Building with initialize_pretrained
    records the VAE and conditioner configs in cfg, so a checkpoint or release rebuilds without the hub.
    """

    GRADIENT_GROUPS = (("conditioner", "conditioner."), ("readout", "readout."))

    def __init__(self, cfg, initialize_pretrained: bool = True):
        super().__init__()
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

        self.cfg = as_dict(cfg)
        conditioner_cfg = self.cfg.setdefault("conditioner", {})
        if initialize_pretrained:
            for key in ("vae_checkpoint", "statistics_path"):
                if self.cfg.get(key) in UNSET:
                    raise ValueError(f"Set model.cfg.{key} (the SSL-VAE stage outputs) to train the separation model")
            vae_cfg, vae_state = load_vae_checkpoint(self.cfg["vae_checkpoint"])
            self.vae = SSLVAE(vae_cfg, initialize_pretrained=False)
            self.vae.load_weights(vae_state)
            self.vae.load_statistics(self.cfg["statistics_path"])
            self.cfg["vae"] = self.vae.cfg
            layers = conditioner_cfg.get("layers")
            depth = {} if layers is None else {"num_hidden_layers": int(layers)}
            base = Wav2Vec2BertModel.from_pretrained(conditioner_cfg.get("name", "facebook/w2v-bert-2.0"),
                                                     layerdrop=0.0, apply_spec_augment=False, **depth)
            conditioner_cfg["hf_config"] = base.config.to_dict()
        else:
            self.vae = SSLVAE(self.cfg["vae"], initialize_pretrained=False)
            base = Wav2Vec2BertModel(Wav2Vec2BertConfig.from_dict(deepcopy(conditioner_cfg["hf_config"])))
        self.vae.sample_posterior = bool(self.cfg.get("sample_posterior", True))
        self.vae.requires_grad_(False).eval()
        if self.cfg.get("efficient_ssl_attention", True):
            optimize_relative_key_attention(self.vae)
            optimize_relative_key_attention(base)
        self.conditioner = base
        set_conditioner_finetune(self.conditioner)
        if conditioner_cfg.get("gradient_checkpointing", False):
            self.conditioner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Models loaded with from_pretrained start in eval mode.
        self.conditioner.train(self.training)
        self.readout = ReadoutHead(base.config.hidden_size, int(self.cfg.get("readout", {}).get("hidden", 768)),
                                   self.vae.dim)
        self.save_hyperparameters({"cfg": self.cfg}, logger=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    def on_fit_start(self):
        torch.set_float32_matmul_precision("medium")
        compile_training_blocks(self)

    def on_train_start(self):
        # Rank and start step separate the posterior and dropout draws.
        reseed_training(self)

    def encode_condition(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """The conditioner's encoding [B, T, C] of the mixture features and its [B] frame counts."""
        return encode_unpadded(self.conditioner, inputs)

    def losses(self, batch: dict) -> tuple[dict[str, Tensor], int]:
        """The training losses of a batch (input_wav [B, 2, N], noisy_mixture [B, N], input_wav_lens [B], sr) and its
        number of examples."""
        from sidon.dialoguesidon2.audio import prepare_ssl_inputs

        # Features of both clean speakers and the mixture, computed together on the model device.
        batch = prepare_ssl_inputs(batch, self.device)
        target, lengths = encode_speakers(self.vae.encode, batch, self.vae.mean.device)
        condition, condition_lengths = self.encode_condition(batch["noisy_mixture_ssl_inputs"])
        if not torch.equal(lengths, condition_lengths):
            raise ValueError("Mixture and target SSL lengths differ")
        mask = length_mask(lengths, target.size(2))
        losses = self.readout.objective(target, condition, mask, **self.cfg.get("loss", {}))
        finite_or_raise(losses["loss"], "readout loss")
        return losses, target.size(0)

    def step(self, batch: dict, stage: str) -> Tensor:
        losses, count = self.losses(batch)
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, sync_dist=stage != "train", batch_size=count)
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # Posterior draws seeded per (rank, batch) keep validation passes comparable and leave the training stream
        # untouched.
        with self.vae.posterior_draws("val_posterior", self.global_rank, batch_idx):
            return self.step(batch, "val")

    def log_gradient_groups(self):
        groups = {}
        for name, parameter in self.named_parameters():
            if parameter.grad is not None:
                group = next((g for g, prefix in self.GRADIENT_GROUPS if name.startswith(prefix)), "other")
                groups.setdefault(group, []).append(parameter.grad)
        for group, grads in groups.items():
            self.log(f"train/grad_norm_{group}", torch.nn.utils.get_total_norm(grads))

    def on_before_optimizer_step(self, optimizer):
        # Per-module norms only on steps that are written to the loggers.
        connector = getattr(self.trainer, "_logger_connector", None)
        if connector is None or connector.should_update_logs:
            self.log_gradient_groups()
        options = self.cfg["optim"]
        valid, norm = safe_gradients(self, max_norm=float(options.get("max_grad_norm", 1.0)),
                                     max_abs_norm=options.get("skip_grad_norm", 20.0))
        self.log("train/update_skipped", float(not valid))
        self.log("train/grad_norm", torch.nan_to_num(norm, nan=0, posinf=1e20))

    def configure_optimizers(self):
        cfg = self.cfg["optim"]
        parameters = optimizer_groups(self, self.conditioner, cfg.get("conditioner_lr"))
        optimizer = adamw(parameters, fused=cfg.get("fused"), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                          betas=tuple(cfg.get("betas", (0.9, 0.999))))
        warmup, cooldown = int(cfg["warmup_steps"]), cfg.get("cooldown_start")
        max_steps, floor = int(cfg["max_steps"]), float(cfg.get("min_lr_ratio", 0.0))

        def schedule(step):
            return learning_rate_factor(step, warmup, cooldown, max_steps, floor)

        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), "interval": "step",
            "name": "train/lr"}}

    @torch.no_grad()
    def separate_features(self, inputs: dict[str, Tensor], waveform_lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Mixture features -> separated waveforms [B, 2, N] at 24 kHz and their lengths [B]."""
        if self.training:
            raise RuntimeError("Call eval() before inference")
        if (waveform_lengths <= 0).any() or (waveform_lengths > 120 * 24000).any():
            raise ValueError("Input duration must be in (0, 120] seconds")
        condition, lengths = self.encode_condition(inputs)
        mask = length_mask(lengths, condition.size(1))
        latent = split_speakers(self.readout.predict(condition, mask))
        return self.vae.decode(latent, lengths, waveform_lengths), waveform_lengths

    @torch.no_grad()
    def forward(self, mixture: Tensor, lengths: Tensor, sample_rate: int = 24000) -> tuple[Tensor, Tensor]:
        """Mono mixtures [B, N] at sample_rate with [B] lengths -> [B, 2, N24] at 24 kHz and the [B] lengths."""
        from sidon.dialoguesidon2.audio import extract_features

        inputs, output_lengths = extract_features(mixture.to(self.device), lengths.to(self.device), sample_rate)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.separate_features(inputs, output_lengths.to(self.device))
