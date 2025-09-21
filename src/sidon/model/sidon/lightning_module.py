"""Lightning modules for Sidon models."""

from __future__ import annotations

from typing import Tuple

import audiotools
import dac
import hydra
import torch
import transformers
from lightning import LightningModule
from lightning.pytorch import loggers
from omegaconf import DictConfig
from peft import LoraConfig, inject_adapter_in_model

from sidon.model.losses import DACLoss, GANLoss


class FeaturePredictorLightningModule(LightningModule):
    """Pre-trains a LoRA-adapted SSL model to mimic a frozen teacher."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.student_ssl_model = transformers.Wav2Vec2BertModel.from_pretrained(
            cfg.ssl_model_name, num_hidden_layers=8, layerdrop=0.0
        ).train()
        self.teacher_ssl_model = transformers.Wav2Vec2BertModel.from_pretrained(
            cfg.ssl_model_name, num_hidden_layers=8
        ).eval()
        adapter_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="lora_only",
            target_modules=["output_dense"],
        )
        self.student_ssl_model = inject_adapter_in_model(
            adapter_config,
            self.student_ssl_model,
        )

        self.ssl_model_criterion = torch.nn.MSELoss()
        for param in self.teacher_ssl_model.parameters():
            param.requires_grad = False

    def on_fit_start(self) -> None:
        torch.set_float32_matmul_precision("medium")

    def step(self, batch, batch_idx: int, stage: str = "train") -> torch.Tensor:
        noisy_ssl_inputs = batch["noisy_ssl_inputs"]
        clean_ssl_inputs = batch["ssl_inputs"]

        with torch.inference_mode():
            teacher_features = self.teacher_ssl_model(
                **clean_ssl_inputs
            ).last_hidden_state
        student_features = self.student_ssl_model(**noisy_ssl_inputs).last_hidden_state

        ssl_loss = self.ssl_model_criterion(student_features, teacher_features.clone())
        self.log(
            f"{stage}/ssl_loss",
            ssl_loss,
            on_step=stage == "train",
            on_epoch=stage == "val",
            sync_dist=stage == "val",
        )

        return ssl_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="val")

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(
            self.student_ssl_model.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2_000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class SidonLightningModule(LightningModule):
    """Sidon decoder/discriminator training using a frozen SSL encoder."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        if cfg.pretraining:
            self.student_ssl_model = transformers.Wav2Vec2BertModel.from_pretrained(
                cfg.ssl_model_name, num_hidden_layers=8
            ).eval()
        else:
            self.student_ssl_model = (
                FeaturePredictorLightningModule.load_from_checkpoint(
                    cfg.ssl_model_name,
                    map_location=self.device,
                ).student_ssl_model.eval()
            )

        if not cfg.pretraining:
            pretrained = SidonLightningModule.load_from_checkpoint(
                cfg.pretrain_path,
                map_location=self.device,
            )
            self.decoder = pretrained.decoder
            self.discriminator = pretrained.discriminator
        else:
            self.decoder = dac.model.dac.Decoder(
                input_channel=self.student_ssl_model.config.hidden_size,  # type: ignore
                channels=1536,
                rates=[8, 5, 4, 3, 2],
            )
            self.discriminator = GANLoss(
                dac.model.discriminator.Discriminator(sample_rate=cfg.sample_rate)
            )
        self.regression_loss = DACLoss(cfg.dac_loss)

        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        torch.set_float32_matmul_precision("medium")

    def step(
        self, batch, batch_idx: int, stage: str = "train"
    ) -> Tuple[torch.Tensor, audiotools.AudioSignal]:
        wavs = batch["input_wav"]
        batch_size = wavs.shape[0]
        opt_g, opt_d = self.optimizers()  # type: ignore
        sch_g, sch_d = self.lr_schedulers()  # type: ignore

        with torch.no_grad():
            if self.cfg.pretraining:
                clean_ssl_inputs = batch["ssl_inputs"]
                student_features = self.student_ssl_model(
                    **clean_ssl_inputs
                ).last_hidden_state
            else:
                noisy_ssl_inputs = batch["noisy_ssl_inputs"]
                student_features = self.student_ssl_model(
                    **noisy_ssl_inputs
                ).last_hidden_state
        input_features = student_features.detach().transpose(1, 2)
        predicted_clean_wavs = self.decoder.forward(input_features)
        if abs(predicted_clean_wavs.shape[-1] - wavs.shape[-1]) > 480:
            raise ValueError("Predicted waveform length deviates too much from target")
        min_length = min(predicted_clean_wavs.shape[-1], wavs.shape[-1])
        predicted_clean_wavs = predicted_clean_wavs[:, :, :min_length]
        wavs = wavs.unsqueeze(1)[:, :, :min_length]

        predicted_clean_wavs = audiotools.AudioSignal(
            predicted_clean_wavs.view(batch_size, 1, -1),
            sample_rate=self.cfg.sample_rate,
        )
        wavs = audiotools.AudioSignal(
            wavs.view(batch_size, 1, -1), sample_rate=self.cfg.sample_rate
        )
        regression_loss = self.regression_loss(wavs, predicted_clean_wavs)["mel_loss"]

        discriminator_loss = self.discriminator.discriminator_loss(
            predicted_clean_wavs.audio_data.detach(),
            wavs.audio_data,
        )
        if stage == "train":
            opt_d.zero_grad()
            self.manual_backward(discriminator_loss)  # type: ignore
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt_d.step()
            sch_d.step()  # type: ignore
        adv_gen, adv_feature = self.discriminator.generator_loss(
            predicted_clean_wavs.audio_data,
            wavs.audio_data,
        )

        self.log(
            f"{stage}/regression_loss", regression_loss, on_step=True, on_epoch=True
        )
        self.log(
            f"{stage}/discriminator_loss",
            discriminator_loss,
            on_step=True,
            on_epoch=True,
        )
        self.log(f"{stage}/adv_gen", adv_gen, on_step=stage == "train", on_epoch=True)
        self.log(
            f"{stage}/adv_feature", adv_feature, on_step=stage == "train", on_epoch=True
        )
        total_loss = (
            self.cfg.loss.loss_weight["regression_loss"] * regression_loss
            + self.cfg.loss.loss_weight["adv_gen"] * adv_gen
            + self.cfg.loss.loss_weight["adv_feature"] * adv_feature
        )
        self.log(f"{stage}/total_loss", total_loss)
        if stage == "train":
            opt_g.zero_grad()
            self.manual_backward(total_loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt_g.step()
            sch_g.step()  # type: ignore

        return total_loss, predicted_clean_wavs

    def on_exception(self, exception: Exception) -> None:
        raise exception

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predicted_clean_wavs = self.step(batch, batch_idx, stage="val")
        if self.global_rank == 0 and batch_idx == 0:
            self.log_audio(
                predicted_clean_wavs.audio_data[0].view(-1).detach(),
                "val/synthesized",
                self.cfg.sample_rate,
            )
            if not self.cfg.pretraining:
                self.log_audio(
                    batch["noisy_input_wav"][0].view(-1).detach(),
                    "val/noisy_input",
                    self.cfg.sample_rate,
                )
            self.log_audio(
                batch["input_wav"][0].view(-1).detach(),
                "val/original",
                self.cfg.sample_rate,
            )
        return loss

    def configure_optimizers(self):  # type: ignore
        opt_g = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
            betas=(0.8, 0.98),
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
            betas=(0.8, 0.98),
        )
        sch_g = hydra.utils.instantiate(
            self.cfg.scheduler.generator,  # type: ignore
            optimizer=opt_g,
        )
        sch_d = hydra.utils.instantiate(
            self.cfg.scheduler.discriminator,  # type: ignore
            optimizer=opt_d,
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
        audio = audio.float().cpu()
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

