"""Hydra entrypoint for training Sidon model variants."""

from __future__ import annotations

from typing import Any, Dict

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")
    """Instantiate datamodule and model from config and launch training."""
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    lightning_module = hydra.utils.instantiate(
        cfg.model.lightning_module,
        cfg=cfg.model.cfg,
    )

    if cfg.train.get("compile", False):
        lightning_module = torch.compile(lightning_module)

    callbacks = [
        hydra.utils.instantiate(callback)
        for callback in cfg.train.get("callbacks", [])
    ]
    logger = hydra.utils.instantiate(cfg.train.logger)

    trainer_kwargs: Dict[str, Any] = OmegaConf.to_container(
        cfg.train.get("trainer", {}), resolve=True
    ) or {}
    trainer = Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)

    ckpt_path = cfg.train.get("ckpt_path")
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
