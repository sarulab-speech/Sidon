"""Training loggers: the configured logger (CSV by default) and, when train.wandb.enabled, Weights & Biases."""

import hydra
from lightning.pytorch.loggers import WandbLogger


def training_loggers(cfg):
    loggers = [hydra.utils.instantiate(cfg.train.logger)]
    options = cfg.train.get("wandb") or {}
    if not options.get("enabled", False):
        return loggers
    loggers.append(WandbLogger(entity=options.get("entity"), project=options.get("project", "dialoguesidon2"),
                               name=options.get("name"), save_dir=cfg.train.logger.save_dir, log_model=False))
    return loggers
