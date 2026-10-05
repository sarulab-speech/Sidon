from lightning import LightningModule
from typing import Optional, Union
import itertools

import hydra
from lightning.pytorch import loggers
from omegaconf import DictConfig
import torchaudio
import torch
import transformers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTModel
from torch import nn

import audiotools
import dac
from sidon.model.gan_ddp import check_ddp_syncs_manual_backward, ghost_loss
from sidon.model.losses import DACLoss, GANLoss
from sidon.model.dialogue_sidion.audio import extract_seamless_m4t_features


class VAEBottleneck(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim * 2)  # for mean and logvar
        )
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick

        return z, mu, logvar, self.kl_loss(mu, logvar)
    def encode(self, x):
        return self.forward(x)
    def kl_loss(self, mu, logvar):
        # KL divergence loss
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=-1).mean()



class SSLVAE(LightningModule):
    """VAE model which compresses ssl features and reconstructs waveforms."""
    def __init__(self,cfg:DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.ssl_model = transformers.Wav2Vec2BertModel.from_pretrained(
            cfg.ssl_model_name_or_path,
            num_hidden_layers=cfg.ssl_num_hidden_layers,
            layerdrop=0.0
        ).eval()
        # Frozen encoder: it runs under inference_mode and no optimizer updates
        # it. Under DDP every parameter that requires grad must get a gradient.
        self.ssl_model.requires_grad_(False)
        self.bottleneck = VAEBottleneck(cfg.vae)

        self.decoder = dac.model.dac.Decoder(
            input_channel=cfg.vae.latent_dim,
            channels=1536,
            rates=[8,5,4,3],
            d_out=1,
        )
        self.discriminator = GANLoss(
            dac.model.discriminator.Discriminator(sample_rate=cfg.sample_rate)
        )
        self.regression_loss = DACLoss(cfg.dac_loss)
        self.automatic_optimization = False
        self.cfg = cfg
    def on_fit_start(self):
        torch.set_float32_matmul_precision('medium')
        check_ddp_syncs_manual_backward(self)
    
    def encode(self, ssl_inputs):
        with torch.inference_mode():
            ssl_features = self.ssl_model(**ssl_inputs).last_hidden_state
        return self.bottleneck.encode(ssl_features.clone())

    
    def step(self, batch, batch_idx:int,stage:str = "train"):
        random_idx = torch.randint(high=2,size=(1,))
        wavs = batch["input_wav"][:,random_idx[0],None,:] # only use first channel

        batch_size = wavs.shape[0]
        opt_g, opt_d = self.optimizers()  # type: ignore
        sch_g, sch_d = self.lr_schedulers()  # type: ignore
        with torch.inference_mode():
            wavs_16k = torchaudio.functional.resample(
                wavs.view(batch_size,-1),
                orig_freq=self.cfg.sample_rate,
                new_freq=16000,
            ).view(batch_size, -1)
            input_features = extract_seamless_m4t_features(
                [torch.nn.functional.pad(0.9 * clean_input_wav.view(-1) / clean_input_wav.abs().max(),(160,160)) for clean_input_wav in wavs_16k],
                device=str(self.device)
            )
        z, mu, logvar,kl_loss = self.encode(input_features)

        predicted_clean_wavs = self.decoder.forward(z.transpose(1,2))
        if abs(predicted_clean_wavs.shape[-1] - wavs.shape[-1]) > 480:
            raise ValueError("Predicted waveform length deviates too much from target")
        min_length = min(predicted_clean_wavs.shape[-1], wavs.shape[-1])
        predicted_clean_wavs = predicted_clean_wavs[:, :, :min_length]
        wavs = wavs[:, :, :min_length]

        predicted_clean_wavs = audiotools.AudioSignal(
            predicted_clean_wavs.view(batch_size, 1, -1),
            sample_rate=self.cfg.sample_rate,
        )
        wavs = audiotools.AudioSignal(
            wavs.view(batch_size, 1, -1), sample_rate=self.cfg.sample_rate
        )
        regression_loss = self.regression_loss(wavs, predicted_clean_wavs)["mel_loss"]

        discriminator_loss = self.discriminator.discriminator_loss(
            predicted_clean_wavs.audio_data.detach()[:,None,0],
            wavs.audio_data[:,None,0],
        )
        if stage == "train":
            opt_d.zero_grad()
            # The ghost term lets DDP see the generator in this backward.
            self.manual_backward(
                discriminator_loss
                + ghost_loss(
                    itertools.chain(
                        self.bottleneck.parameters(), self.decoder.parameters()
                    )
                )
            )
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt_d.step()
            sch_d.step()  # type: ignore
        adv_gen, adv_feature = self.discriminator.generator_loss(
            predicted_clean_wavs.audio_data[:,None,0],
            wavs.audio_data[:,None,0],
        )

        self.log(
            f"{stage}/regression_loss", regression_loss, on_step=True, on_epoch=True
        )
        self.log(f"{stage}/kl_loss", kl_loss, on_step=True, on_epoch=True)
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
            + self.cfg.loss.loss_weight["kl_loss"] * kl_loss
        )
        self.log(f"{stage}/total_loss", total_loss)
        if stage == "train":
            opt_g.zero_grad()
            self.manual_backward(total_loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt_g.step()
            sch_g.step()  # type: ignore

        return total_loss, predicted_clean_wavs
    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predicted_clean_wavs = self.step(batch, batch_idx, stage="val")
        if self.global_rank == 0 and batch_idx < 4:
            self.log_audio(
                predicted_clean_wavs.audio_data[0].detach(),
                "val/synthesized",
                self.cfg.sample_rate,
            )
            self.log_audio(
                batch["noisy_input_wav"][0].detach(),
                "val/noisy_input",
                self.cfg.sample_rate,
            )
            self.log_audio(
                batch["input_wav"][0].detach(),
                "val/original",
                self.cfg.sample_rate,
            )
        return loss

    def configure_optimizers(self):  # type: ignore
        opt_g = torch.optim.AdamW(
            itertools.chain(
                self.bottleneck.parameters(),
                self.decoder.parameters(),
            ),
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

class DitModel(StableAudioDiTModel):
    def __init__(self, sample_size = 1024, in_channels = 64, num_layers = 24, attention_head_dim = 64, num_attention_heads = 24, num_key_value_attention_heads = 12, out_channels = 64, cross_attention_dim = 768, time_proj_dim = 256, global_states_input_dim = 1536, cross_attention_input_dim = 768):
        super().__init__(sample_size, in_channels, num_layers, attention_head_dim, num_attention_heads, num_key_value_attention_heads, out_channels, cross_attention_dim, time_proj_dim, global_states_input_dim, cross_attention_input_dim)
        del self.global_proj
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        global_hidden_states: torch.FloatTensor = None,
        rotary_embedding: torch.FloatTensor = None,
        return_dict: bool = True,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`StableAudioDiTModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, in_channels, sequence_len)`):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, encoder_sequence_len, cross_attention_input_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            global_hidden_states (`torch.FloatTensor` of shape `(batch size, global_sequence_len, global_states_input_dim)`):
               Global embeddings that will be prepended to the hidden states.
            rotary_embedding (`torch.Tensor`):
                The rotary embeddings to apply on query and key tensors during attention calculation.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token indices, formed by concatenating the attention
                masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token cross-attention indices, formed by concatenating
                the attention masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        cross_attention_hidden_states = self.cross_attention_proj(encoder_hidden_states)
        time_hidden_states = self.timestep_proj(self.time_proj(timestep.to(self.dtype)))

        global_hidden_states = time_hidden_states.unsqueeze(1)

        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        # (batch_size, dim, sequence_length) -> (batch_size, sequence_length, dim)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.proj_in(hidden_states)

        # prepend global states to hidden states
        hidden_states = torch.cat([global_hidden_states, hidden_states], dim=-2)
        if attention_mask is not None:
            prepend_mask = torch.ones((hidden_states.shape[0], 1), device=hidden_states.device, dtype=torch.bool)
            attention_mask = torch.cat([prepend_mask, attention_mask], dim=-1)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    cross_attention_hidden_states,
                    encoder_attention_mask,
                    rotary_embedding,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=cross_attention_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    rotary_embedding=rotary_embedding,
                )

        hidden_states = self.proj_out(hidden_states)

        # (batch_size, sequence_length, dim) -> (batch_size, dim, sequence_length)
        # remove prepend length that has been added by global hidden states
        hidden_states = hidden_states.transpose(1, 2)[:, :, 1:]
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
