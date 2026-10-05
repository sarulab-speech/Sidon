"""Standalone DAC decoder finetuning for dialogue Sidon predictors."""

from __future__ import annotations

from typing import Tuple

import audiotools
import dac
import torch
import torch.nn as nn
import torchaudio
import transformers
from lightning import LightningModule
from lightning.pytorch import loggers
from omegaconf import DictConfig

from sidon.model.dialogue_sidion.lightning_module import (
    DialogueSidonDiffusionLightningModule,
    DialogueSidonNoDiffusionHeadLightningModule,
    DialogueSidonNoVaeLatentLayer8LightningModule,
    DialogueSidonNoVaeLatentWithDiffusionHeadLightningModule,
)
from sidon.model.gan_ddp import check_ddp_syncs_manual_backward, ghost_loss
from sidon.model.losses import DACLoss, GANLoss


class DialogueSidonDacDecoderFinetuneLightningModule(LightningModule):
    """Train a fresh DAC decoder (+ discriminator) from predictor outputs."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        predictor_class_name = cfg.predictor_class
        predictor_cls_map = {
            "DialogueSidonDiffusionLightningModule": DialogueSidonDiffusionLightningModule,
            "DialogueSidonNoDiffusionHeadLightningModule": DialogueSidonNoDiffusionHeadLightningModule,
            "DialogueSidonNoVaeLatentLayer8LightningModule": DialogueSidonNoVaeLatentLayer8LightningModule,
            "DialogueSidonNoVaeLatentWithDiffusionHeadLightningModule": DialogueSidonNoVaeLatentWithDiffusionHeadLightningModule,
        }
        if predictor_class_name not in predictor_cls_map:
            raise ValueError(
                "Unsupported predictor_class. "
                f"Got {predictor_class_name}. "
                f"Supported: {sorted(predictor_cls_map.keys())}"
            )

        predictor_cls = predictor_cls_map[predictor_class_name]
        self.predictor = predictor_cls.load_from_checkpoint(
            cfg.predictor_checkpoint_path,
            map_location="cpu",
        ).eval()
        for param in self.predictor.parameters():
            param.requires_grad = False

        self.predictor_representation = str(cfg.get("predictor_representation", "latents")).lower()
        if self.predictor_representation not in ("latents", "features"):
            raise ValueError(
                f"Unsupported predictor_representation={self.predictor_representation}. "
                "Expected 'latents' or 'features'."
            )

        self.use_diffusion_output = bool(cfg.get("use_diffusion_output", False))
        self.diffusion_num_steps = cfg.get("diffusion_num_steps")

        latent_predictor_types = (
            DialogueSidonDiffusionLightningModule,
            DialogueSidonNoDiffusionHeadLightningModule,
        )
        feature_predictor_types = (
            DialogueSidonNoVaeLatentLayer8LightningModule,
            DialogueSidonNoVaeLatentWithDiffusionHeadLightningModule,
        )
        if self.predictor_representation == "latents" and not isinstance(
            self.predictor,
            latent_predictor_types,
        ):
            raise ValueError(
                "predictor_representation='latents' requires a latent predictor class."
            )
        if self.predictor_representation == "features" and not isinstance(
            self.predictor,
            feature_predictor_types,
        ):
            raise ValueError(
                "predictor_representation='features' requires a feature predictor class."
            )

        if self.predictor_representation == "latents":
            source_dim = int(self.predictor.output_linear1.out_features)
        else:
            source_dim = int(self.predictor.student_ssl_model.config.hidden_size)

        decoder_cfg = cfg.get("decoder", {})
        decoder_input_cfg = decoder_cfg.get("input_channel")
        self.decoder_input_dim = source_dim if decoder_input_cfg is None else int(decoder_input_cfg)
        self.decoder = dac.model.dac.Decoder(
            input_channel=self.decoder_input_dim,
            channels=int(decoder_cfg.get("channels", 1536)),
            rates=list(decoder_cfg.get("rates", [8, 5, 4, 3])),
            d_out=1,
        )

        self.input_proj_0: nn.Module | None = None
        self.input_proj_1: nn.Module | None = None
        if source_dim != self.decoder_input_dim:
            self.input_proj_0 = nn.Linear(source_dim, self.decoder_input_dim)
            self.input_proj_1 = nn.Linear(source_dim, self.decoder_input_dim)

        self.discriminator = GANLoss(
            dac.model.discriminator.Discriminator(sample_rate=cfg.sample_rate)
        )
        self.regression_loss = DACLoss(cfg.dac_loss)

        loss_weight_cfg = cfg.get("loss", {}).get("loss_weight", {})
        self.stft_loss_weight = float(loss_weight_cfg.get("stft_loss", 1.0))
        self.mel_loss_weight = float(loss_weight_cfg.get("mel_loss", 15.0))
        self.wav_loss_weight = float(loss_weight_cfg.get("wav_loss", 1.0))
        self.adv_gen_loss_weight = float(loss_weight_cfg.get("adv_gen", 2.0))
        self.adv_feature_loss_weight = float(loss_weight_cfg.get("adv_feature", 1.0))

        self.grad_clip_norm = float(cfg.optim.get("grad_clip_norm", 1.0))
        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        torch.set_float32_matmul_precision("medium")
        self.predictor.eval()
        check_ddp_syncs_manual_backward(self)

    @staticmethod
    def _align_predicted_speakers(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Reorder two-speaker predictions to match target permutation, without logging."""
        n_speakers = predicted.shape[0]
        if n_speakers != 2:
            return predicted

        def _speaker_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            diff = (a - b).pow(2)
            return diff.view(diff.shape[0], diff.shape[1], -1).mean(dim=-1)

        same_cost = _speaker_distance(predicted, target).sum(dim=0)
        swapped = predicted.flip(0)
        swapped_cost = _speaker_distance(swapped, target).sum(dim=0)
        swap_mask = swapped_cost < same_cost
        if not swap_mask.any():
            return predicted

        aligned = predicted.clone()
        aligned[:, swap_mask] = swapped[:, swap_mask]
        return aligned

    @torch.no_grad()
    def _predict_latent_pairs(self, batch, batch_idx: int, stage: str) -> torch.Tensor:
        _, predicted, target, student_features = self.predictor.feature_predictor_step(
            batch,
            batch_idx,
            stage=stage,
            log=False,
        )
        predicted = self._align_predicted_speakers(predicted, target)
        predicted_latents = torch.cat([predicted[0], predicted[1]], dim=-1)
        if not self.use_diffusion_output:
            return predicted_latents

        if getattr(self.predictor, "diffusion_head", None) is None:
            raise ValueError(
                "use_diffusion_output=True requires a predictor with diffusion_head enabled."
            )
        if not hasattr(self.predictor, "sample_latents"):
            raise ValueError(
                "use_diffusion_output=True requires predictor with sample_latents()."
            )

        conditioning = torch.cat(
            [self.predictor._normalize_latents(predicted_latents), student_features],
            dim=-1,
        )
        sampled_latents = self.predictor.sample_latents(
            conditioning,
            conditioning.shape[1],
            num_steps=self.diffusion_num_steps,
        )
        return self.predictor._denormalize_latents(sampled_latents)

    @torch.no_grad()
    def _predict_feature_pairs(self, batch, batch_idx: int, stage: str) -> torch.Tensor:
        _, predicted, target, student_features = self.predictor.feature_predictor_step(
            batch,
            batch_idx,
            stage=stage,
            log=False,
        )
        predicted = self._align_predicted_speakers(predicted, target)
        predicted_features = torch.cat([predicted[0], predicted[1]], dim=-1)
        if not self.use_diffusion_output:
            return predicted_features

        if not hasattr(self.predictor, "sample_ssl_feature_pairs"):
            raise ValueError(
                "use_diffusion_output=True requires predictor with sample_ssl_feature_pairs()."
            )
        if not hasattr(self.predictor, "_normalize_features") or not hasattr(
            self.predictor,
            "_denormalize_features",
        ):
            raise ValueError(
                "use_diffusion_output=True for feature predictors requires feature normalization helpers."
            )

        normalized_features = self.predictor._normalize_features(predicted_features)
        conditioning = torch.cat([normalized_features, student_features], dim=-1)
        sampled_features = self.predictor.sample_ssl_feature_pairs(
            conditioning,
            conditioning.shape[1],
            num_steps=self.diffusion_num_steps,
        )
        return self.predictor._denormalize_features(sampled_features)

    def _decode_pair_tensor(self, pair_tensor: torch.Tensor) -> torch.Tensor:
        source_dim = pair_tensor.shape[-1] // 2
        first = pair_tensor[:, :, :source_dim]
        second = pair_tensor[:, :, source_dim:]

        if self.input_proj_0 is not None and self.input_proj_1 is not None:
            first = self.input_proj_0(first)
            second = self.input_proj_1(second)

        first_wav = self.decoder(first.transpose(1, 2))
        second_wav = self.decoder(second.transpose(1, 2))
        return torch.cat([first_wav, second_wav], dim=1)

    def _resample_targets_if_needed(self, wavs: torch.Tensor, batch) -> torch.Tensor:
        batch_sr = batch.get("sr")
        if batch_sr is None:
            return wavs

        if isinstance(batch_sr, torch.Tensor):
            target_sr = int(batch_sr.view(-1)[0].item())
        else:
            target_sr = int(batch_sr)

        if target_sr == self.cfg.sample_rate:
            return wavs

        batch_size, channels, time = wavs.shape
        wavs_2d = wavs.view(batch_size * channels, time)
        resampled = torchaudio.functional.resample(
            wavs_2d,
            orig_freq=target_sr,
            new_freq=self.cfg.sample_rate,
        )
        return resampled.view(batch_size, channels, -1)

    def step(
        self,
        batch,
        batch_idx: int,
        stage: str = "train",
    ) -> Tuple[torch.Tensor, audiotools.AudioSignal, audiotools.AudioSignal]:
        opt_g, opt_d = self.optimizers()  # type: ignore
        sch_g, sch_d = self.lr_schedulers()  # type: ignore

        with torch.no_grad():
            if self.predictor_representation == "latents":
                pair_tensor = self._predict_latent_pairs(batch, batch_idx, stage)
            else:
                pair_tensor = self._predict_feature_pairs(batch, batch_idx, stage)

        predicted_wavs = self._decode_pair_tensor(pair_tensor)
        target_wavs = self._resample_targets_if_needed(batch["input_wav"].float(), batch)

        if predicted_wavs.ndim != 3 or target_wavs.ndim != 3:
            raise ValueError(
                "Expected predicted and target wavs with shape [B, C, T]. "
                f"Got predicted={tuple(predicted_wavs.shape)}, target={tuple(target_wavs.shape)}"
            )

        min_length = min(predicted_wavs.shape[-1], target_wavs.shape[-1])
        predicted_wavs = predicted_wavs[:, :, :min_length]
        target_wavs = target_wavs[:, :, :min_length]

        batch_size, channels, time = predicted_wavs.shape
        pred_flat = predicted_wavs.view(batch_size * channels, 1, time)
        target_flat = target_wavs.view(batch_size * channels, 1, time)

        predicted_audio = audiotools.AudioSignal(
            pred_flat,
            sample_rate=self.cfg.sample_rate,
        )
        target_audio = audiotools.AudioSignal(
            target_flat,
            sample_rate=self.cfg.sample_rate,
        )

        reg_losses = self.regression_loss(target_audio, predicted_audio)
        stft_loss = reg_losses["stft_loss"]
        mel_loss = reg_losses["mel_loss"]
        wav_loss = reg_losses["wav_loss"]
        recon_loss = (
            self.stft_loss_weight * stft_loss
            + self.mel_loss_weight * mel_loss
            + self.wav_loss_weight * wav_loss
        )

        discriminator_loss = self.discriminator.discriminator_loss(
            predicted_audio.audio_data.detach(),
            target_audio.audio_data,
        )
        if stage == "train":
            opt_d.zero_grad()
            # The ghost term lets DDP see the generator in this backward.
            self.manual_backward(
                discriminator_loss + ghost_loss(self._generator_parameters())
            )
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_norm)
            opt_d.step()
            sch_d.step()

        adv_gen, adv_feature = self.discriminator.generator_loss(
            predicted_audio.audio_data,
            target_audio.audio_data,
        )
        total_loss = (
            recon_loss
            + self.adv_gen_loss_weight * adv_gen
            + self.adv_feature_loss_weight * adv_feature
        )

        if stage == "train":
            opt_g.zero_grad()
            self.manual_backward(total_loss)
            torch.nn.utils.clip_grad_norm_(self._generator_parameters(), self.grad_clip_norm)
            opt_g.step()
            sch_g.step()

        self.log(
            f"{stage}/stft_loss",
            stft_loss,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=stage == "val",
        )
        self.log(
            f"{stage}/mel_loss",
            mel_loss,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=stage == "val",
        )
        self.log(
            f"{stage}/wav_loss",
            wav_loss,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=stage == "val",
        )
        self.log(
            f"{stage}/recon_loss",
            recon_loss,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=stage == "val",
        )
        self.log(
            f"{stage}/discriminator_loss",
            discriminator_loss,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=stage == "val",
        )
        self.log(
            f"{stage}/adv_gen",
            adv_gen,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=stage == "val",
        )
        self.log(
            f"{stage}/adv_feature",
            adv_feature,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=stage == "val",
        )
        self.log(
            f"{stage}/total_loss",
            total_loss,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=stage == "val",
        )

        pred_audio_multi = audiotools.AudioSignal(
            predicted_wavs,
            sample_rate=self.cfg.sample_rate,
        )
        target_audio_multi = audiotools.AudioSignal(
            target_wavs,
            sample_rate=self.cfg.sample_rate,
        )
        return total_loss, pred_audio_multi, target_audio_multi

    def _generator_parameters(self):
        params = list(self.decoder.parameters())
        if self.input_proj_0 is not None and self.input_proj_1 is not None:
            params.extend(self.input_proj_0.parameters())
            params.extend(self.input_proj_1.parameters())
        return params

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predicted_audio, target_audio = self.step(batch, batch_idx, stage="val")
        if self.global_rank == 0 and batch_idx < 3:
            self.log_audio(
                predicted_audio.audio_data[0].detach(),
                f"val/synthesized_{batch_idx}",
                self.cfg.sample_rate,
            )
            self.log_audio(
                batch["noisy_16k_mixture"][0].detach(),
                f"val/noisy_input_{batch_idx}",
                16_000,
            )
            self.log_audio(
                target_audio.audio_data[0].detach(),
                f"val/original_{batch_idx}",
                self.cfg.sample_rate,
            )
        return loss

    def configure_optimizers(self):  # type: ignore
        opt_g = torch.optim.AdamW(
            self._generator_parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
            betas=(0.8, 0.98),
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.cfg.optim.get("disc_lr", self.cfg.optim.lr),
            weight_decay=self.cfg.optim.weight_decay,
            betas=(0.8, 0.98),
        )

        warmup_steps = int(self.cfg.optim.get("warmup_steps", 2_000))
        sch_g = transformers.get_constant_schedule_with_warmup(
            opt_g,
            num_warmup_steps=warmup_steps,
        )
        sch_d = transformers.get_constant_schedule_with_warmup(
            opt_d,
            num_warmup_steps=warmup_steps,
        )
        return (
            {
                "optimizer": opt_g,
                "lr_scheduler": sch_g,
            },
            {
                "optimizer": opt_d,
                "lr_scheduler": sch_d,
            },
        )

    def log_audio(self, audio: torch.Tensor, name: str, sampling_rate: int) -> None:
        audio = audio.float().cpu().numpy().T
        for logger in self.loggers:
            if isinstance(logger, loggers.WandbLogger):
                import wandb

                wandb.log(
                    {name: wandb.Audio(audio, sample_rate=sampling_rate)},
                    step=self.global_step,
                )
            elif isinstance(logger, loggers.TensorBoardLogger):
                logger.experiment.add_audio(
                    name,
                    audio,
                    self.global_step,
                    sampling_rate,
                )


class DialogueSidonVaeDecoderFinetuneLightningModule(DialogueSidonDacDecoderFinetuneLightningModule):
    """Backward-compatible alias to the DAC decoder finetune module."""

