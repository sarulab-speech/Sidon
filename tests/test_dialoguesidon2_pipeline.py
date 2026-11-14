"""Both training stages end to end on CPU with tiny models and shards, through the Hydra configs: SSL-VAE stage ->
latent statistics -> separation training -> resume -> release. The hub is replaced by a tiny w2v-BERT and the DAC
discriminator by a one-scale stand-in."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dialoguesidon2_fixtures import (fake_discriminator, tiny_hub, write_dialogue_shards,  # noqa: E402
                                     write_sidon_shards)

REPO = Path(__file__).resolve().parents[1]
CPU_TRAINER = ["train.trainer.accelerator=cpu", "train.trainer.devices=1", "train.trainer.num_nodes=1",
               "~train.trainer.strategy", "train.trainer.precision=32-true", "train.trainer.max_steps=2",
               "train.trainer.val_check_interval=1", "train.trainer.limit_val_batches=1",
               "train.trainer.log_every_n_steps=1", "+train.trainer.enable_progress_bar=false",
               "+train.trainer.enable_model_summary=false"]
TINY_VAE = ["model.cfg.vae.target_layer=2", "model.cfg.vae.input_dim=16", "model.cfg.vae.hidden_dim=8",
            "model.cfg.vae.latent_dim=4", "model.cfg.vae.decoder_channels=16", "model.cfg.decoder_crop_frames=40"]
NO_COMPILE = ["model.cfg.performance.compile_ssl=false", "model.cfg.performance.flex_ssl_attention=false"]


def compose(overrides):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        return hydra_compose("dialoguesidon2", overrides=overrides)


def fit(cfg):
    """What sidon.dialoguesidon2.train.main does after composing the config."""
    import hydra
    from lightning import Trainer, seed_everything

    from sidon.dialoguesidon2.loggers import training_loggers
    from sidon.dialoguesidon2.train import build_model

    seed_everything(cfg.seed, workers=True)
    data = hydra.utils.instantiate(cfg.data.datamodule)
    model = build_model(cfg)
    callbacks = [hydra.utils.instantiate(c) for c in cfg.train.callbacks]
    trainer = Trainer(**dict(hydra.utils.instantiate(cfg.train.trainer)), logger=training_loggers(cfg),
                      callbacks=callbacks)
    trainer.fit(model, datamodule=data, ckpt_path=cfg.train.get("ckpt_path"))
    for callback in callbacks:
        if hasattr(callback, "flush"):
            callback.flush(30)
    path = Path(trainer.checkpoint_callback.dirpath) / "final.ckpt"
    trainer.save_checkpoint(str(path))
    return model, trainer, path


class PipelineTests(unittest.TestCase):
    def test_two_stages_statistics_resume_and_release(self):
        import warnings

        from sidon.dialoguesidon2.release import load_release, main as release_main
        from sidon.dialoguesidon2.statistics import compute_statistics

        warnings.filterwarnings("ignore")
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_shards(root / "dialogue")
            write_sidon_shards(root / "sidon")
            dialogue = [f"data.datamodule.train_urls=[{root / 'dialogue' / 'train'}]",
                        f"data.datamodule.val_urls=[{root / 'dialogue' / 'valid'}]"]

            # Stage 1: the SSL-VAE.
            stage1 = compose(["model=dialoguesidon2_ssl_vae", "data=dialoguesidon2_ssl_vae",
                              "train=dialoguesidon2_ssl_vae", *TINY_VAE, *NO_COMPILE, *CPU_TRAINER,
                              f"data.datamodule.train_urls=[{root / 'sidon' / 'train'}]",
                              f"data.datamodule.val_urls=[{root / 'sidon' / 'valid'}]",
                              "data.datamodule.batch_size=2", "data.datamodule.num_workers=0",
                              "data.datamodule.segment_seconds=1.0", f"train.logger.save_dir={root / 'runs'}",
                              "train.logger.name=stage1"])
            with tiny_hub(), fake_discriminator():
                stage, trainer, vae_checkpoint = fit(stage1)
            self.assertEqual(trainer.global_step, 2)  # one batch: a discriminator and a generator step
            self.assertTrue(list((root / "runs" / "stage1").rglob("synthesized_s0_0.wav")))

            # Latent statistics of the stage 1 checkpoint.
            statistics = compose([f"statistics.vae_checkpoint={vae_checkpoint}",
                                  f"statistics.train_urls=[{root / 'dialogue' / 'train'}]",
                                  f"statistics.output={root / 'statistics.pt'}", "statistics.segment_seconds=1.0",
                                  "statistics.batch_size=2", "statistics.max_batches=3", "statistics.device=cpu",
                                  "statistics.precision=32-true"])
            result = compute_statistics(statistics)
            self.assertEqual(result["batches"], 3)
            self.assertEqual(tuple(result["mean"].shape), (4,))
            torch.save(result, root / "statistics.pt")

            # Stage 2: the separation model, from the stage 1 checkpoint and the statistics.
            separation = ["model.cfg.conditioner.layers=3", "model.cfg.readout.hidden=12", *NO_COMPILE, *CPU_TRAINER,
                          *dialogue, "data.datamodule.batch_size=2", "data.datamodule.num_workers=0",
                          "data.datamodule.segment_seconds=1.0", f"train.logger.save_dir={root / 'runs'}",
                          "train.logger.name=stage2", f"model.cfg.vae_checkpoint={vae_checkpoint}",
                          f"model.cfg.statistics_path={root / 'statistics.pt'}"]
            with tiny_hub():
                model, trainer, checkpoint = fit(compose(separation))
            self.assertEqual(trainer.global_step, 2)
            frozen = {k: v for k, v in model.vae.state_dict().items() if k not in ("mean", "var", "stats_ready")}
            stage_state = stage.vae.state_dict()
            self.assertTrue(all(torch.equal(v.cpu(), stage_state[k].cpu()) for k, v in frozen.items()))
            torch.testing.assert_close(model.vae.mean.cpu(), result["mean"])
            self.assertTrue(list((root / "runs" / "stage2").rglob("synthesized_s1_0.wav")))

            # Resume: rebuilt from the checkpoint alone (no hub), the step count continues.
            with patch("transformers.Wav2Vec2BertModel.from_pretrained", side_effect=AssertionError("hub used")):
                resumed, trainer, _ = fit(compose([*separation, "train.trainer.max_steps=4",
                                                   f"train.ckpt_path={checkpoint}", "train.logger.name=stage2b"]))
            self.assertEqual(trainer.global_step, 4)
            self.assertEqual(resumed.cfg["optim"]["max_steps"], 4)

            # Release: model.safetensors + config.json without local paths, strict reload.
            release_main(["--checkpoint", str(checkpoint), "--out", str(root / "release")])
            text = (root / "release" / "config.json").read_text()
            self.assertNotIn(str(root), text)
            reloaded = load_release(root / "release")
            expected = model.state_dict()
            self.assertTrue(all(torch.equal(v, expected[k].cpu()) for k, v in reloaded.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
