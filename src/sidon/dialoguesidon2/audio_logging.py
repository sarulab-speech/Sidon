"""Validation audio: the first examples of each validation pass, separated (separation model) or reconstructed
(SSL-VAE stage), logged to W&B or TensorBoard when configured and written as WAV files under the log directory."""

import json
import threading
from pathlib import Path

import torch
import torchaudio
from lightning import Callback
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


def write_files(root: Path, waves: dict, metadata: dict, batch_idx: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, wave in waves.items():
        torchaudio.save(str(root / f"{name}_{batch_idx}.wav"), wave.unsqueeze(0), 24000, encoding="PCM_F",
                        bits_per_sample=32)
    (root / f"metadata_{batch_idx}.json").write_text(json.dumps(metadata, indent=2) + "\n")


class ValidationAudio(Callback):
    def __init__(self, max_batches: int = 1):
        self.max_batches = max_batches
        self.output_dir = None
        # Files are written by a background thread, so a slow filesystem never holds rank zero inside validation
        # while the other ranks wait.
        self._writer: threading.Thread | None = None

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for the writer thread; True when no write is pending."""
        if self._writer is not None:
            self._writer.join(timeout)
        return self._writer is None or not self._writer.is_alive()

    def on_fit_start(self, trainer, pl_module):
        # Trainer.log_dir broadcasts, so every DDP rank must enter this hook.
        self.output_dir = Path(trainer.log_dir or trainer.default_root_dir)

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        from sidon.dialoguesidon2.audio import prepare_ssl_inputs
        from sidon.model.dialoguesidon2.lightning_module import SSLVAEModule, encode_speakers

        if not trainer.is_global_zero or trainer.sanity_checking or batch_idx >= self.max_batches:
            return

        def first(value):
            if isinstance(value, torch.Tensor):
                return value[:1] if value.ndim else value
            if isinstance(value, dict):
                return {key: first(item) for key, item in value.items()}
            return value

        batch = prepare_ssl_inputs(first(batch), pl_module.device)
        length = int(batch["input_wav_lens"][0])
        original = batch["input_wav"][0, :, :length]
        audio = {f"original_s{i}": original[i] for i in range(2)}
        if "noisy_mixture" in batch:
            audio["noisy_input"] = batch["noisy_mixture"][0, :length]
        if isinstance(pl_module, SSLVAEModule):
            vae = pl_module.vae
            latent, frames = encode_speakers(vae.posterior_mean, batch, vae.mean.device)
            generated = vae.decode(latent, frames, batch["input_wav_lens"].to(latent.device), normalized=False)[0]
        else:
            generated = pl_module.separate_features(batch["noisy_mixture_ssl_inputs"],
                                                    batch["input_wav_lens"].to(pl_module.device))[0][0]
        audio.update({f"synthesized_s{i}": generated[i, :length] for i in range(generated.size(0))})
        waves = {name: wave.detach().float().cpu() for name, wave in audio.items()}
        for name, wave in waves.items():
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    import wandb

                    logger.experiment.log({f"val/{name}_{batch_idx}": wandb.Audio(wave.numpy(), sample_rate=24000),
                                           "trainer/global_step": trainer.global_step})
                elif isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_audio(f"val/{name}_{batch_idx}", wave.unsqueeze(0),
                                                trainer.global_step, sample_rate=24000)
        if self._writer is not None and self._writer.is_alive():
            print(f"Validation audio of step {trainer.global_step} skipped: the previous write has not finished",
                  flush=True)
            return
        root = self.output_dir / "audio" / f"step_{trainer.global_step:09d}"
        metadata = {"step": trainer.global_step, "sample_rate": 24000, "samples": length}
        self._writer = threading.Thread(target=write_files, args=(root, waves, metadata, batch_idx),
                                        name="validation-audio", daemon=True)
        self._writer.start()
