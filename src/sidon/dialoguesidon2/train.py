"""Hydra training entry point of both DialogueSidon2 stages.

  # Stage 1: SSL-VAE
  python -m sidon.dialoguesidon2.train model=dialoguesidon2_ssl_vae data=dialoguesidon2_ssl_vae \
      train=dialoguesidon2_ssl_vae 'data.datamodule.train_urls=[...]' 'data.datamodule.val_urls=[...]'
  # Stage 2: separation (after sidon.dialoguesidon2.statistics)
  python -m sidon.dialoguesidon2.train model.cfg.vae_checkpoint=<stage 1 ckpt> \
      model.cfg.statistics_path=<statistics> 'data.datamodule.train_urls=[...]' 'data.datamodule.val_urls=[...]'

train.ckpt_path resumes a run: the model is rebuilt from the checkpoint (no hub download), and only execution
options and the learning-rate tail come from the current config (performance.apply_resume_options).
scripts/dialoguesidon2/ has multi-GPU launchers.
"""

from pathlib import Path

import hydra
from lightning import LightningModule, Trainer, seed_everything
from omegaconf import DictConfig

from sidon.dialoguesidon2.loggers import training_loggers
from sidon.dialoguesidon2.timing import AllocationTimer
from sidon.model.dialoguesidon2.lightning_module import as_dict
from sidon.model.dialoguesidon2.performance import apply_resume_options


def build_model(cfg: DictConfig) -> LightningModule:
    """Resume (train.ckpt_path) or build a fresh model from the config."""
    if cfg.train.get("ckpt_path"):
        model_class = hydra.utils.get_class(cfg.model.lightning_module._target_)
        model = model_class.load_from_checkpoint(cfg.train.ckpt_path, map_location="cpu", initialize_pretrained=False)
        apply_resume_options(model, as_dict(cfg.model.cfg))
        return model
    return hydra.utils.instantiate(cfg.model.lightning_module, cfg=cfg.model.cfg)


@hydra.main(config_path="../../../config", config_name="dialoguesidon2", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    data = hydra.utils.instantiate(cfg.data.datamodule)
    model = build_model(cfg)
    trainer_options = dict(hydra.utils.instantiate(cfg.train.trainer))
    callbacks = [hydra.utils.instantiate(c) for c in cfg.train.callbacks]
    if cfg.train.get("reset_time_on_resume", False):
        callbacks.append(AllocationTimer(duration=trainer_options.pop("max_time")))
    trainer = Trainer(**trainer_options, logger=training_loggers(cfg), callbacks=callbacks)
    trainer.fit(model, datamodule=data, ckpt_path=cfg.train.get("ckpt_path"))
    # A time-limited run can stop between checkpoints: keep its final state.
    checkpoint_dir = trainer.checkpoint_callback.dirpath if trainer.checkpoint_callback else trainer.default_root_dir
    trainer.save_checkpoint(str(Path(checkpoint_dir) / "final.ckpt"))


if __name__ == "__main__":
    main()
