"""Two-process CPU DDP check that the manual-optimization GAN stages keep
their replicas in sync.

Each batch runs a discriminator and a generator ``manual_backward`` inside one
DDP forward. With ``static_graph=True`` DDP never all-reduces those gradients,
so every rank trained its own copy. ``train=gan`` (``static_graph: false``),
the ghost loss and the frozen SSL encoders make every optimizer step see the
same averaged gradients on all ranks.
"""

import importlib.util
import json
import logging
import os
import re
import socket
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO = Path(__file__).resolve().parents[1]
WORLD_SIZE = 2
BATCHES = 2
GAN_MODELS = (
    "sidon_vocoder_pretrain",
    "sidon_vocoder_finetune",
    "ssl_vae",
    "diffusion_dialogue_sidon_decoder_finetune",
)
# SSLVAE's module imports diffusers.
MODELS = ("sidon_vocoder_pretrain",) + (
    ("ssl_vae",) if importlib.util.find_spec("diffusers") else ()
)


def tiny_fakes():
    """Tiny stand-ins for the pretrained encoder and the DAC networks."""
    import dac
    import transformers

    def tiny_w2v(cls, *args, **kwargs):
        config = transformers.Wav2Vec2BertConfig(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, feature_projection_input_dim=160,
            conv_depthwise_kernel_size=3, output_hidden_size=16, layerdrop=0.0,
        )
        return transformers.Wav2Vec2BertModel(config)

    decoder, discriminator = dac.model.dac.Decoder, dac.model.discriminator.Discriminator

    class TinyDecoder(decoder):
        def __init__(self, input_channel, channels, rates, d_out=1):
            super().__init__(input_channel, 64, rates, d_out)

    class TinyDiscriminator(discriminator):
        def __init__(self, sample_rate=44100, **kwargs):
            super().__init__(rates=[], periods=[2], fft_sizes=[256], sample_rate=sample_rate)

    return [
        patch.object(transformers.Wav2Vec2BertModel, "from_pretrained", classmethod(tiny_w2v)),
        patch.object(dac.model.dac, "Decoder", TinyDecoder),
        patch.object(dac.model.discriminator, "Discriminator", TinyDiscriminator),
    ]


def compose(model: str, train: str):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    overrides = [
        f"model={model}", f"train={train}", "train.trainer.precision=32",
        "+train.trainer.accelerator=cpu", f"+train.trainer.devices={WORLD_SIZE}",
        "+train.trainer.num_nodes=1", f"train.trainer.max_steps={2 * BATCHES}",
        "train.trainer.limit_val_batches=0", "+train.trainer.num_sanity_val_steps=0",
        "+train.trainer.enable_checkpointing=false", "+train.trainer.enable_progress_bar=false",
        "+train.trainer.enable_model_summary=false", "+train.trainer.use_distributed_sampler=false",
        "+train.trainer.logger=false", "+train.trainer.strategy.process_group_backend=gloo",
        "+train.trainer.strategy.timeout={_target_:datetime.timedelta,seconds:120}",
    ]
    if train == "default":
        overrides.append("train.trainer.gradient_clip_val=null")  # as the launchers did
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        return hydra_compose(config_name="config", overrides=overrides)


def generator_and_discriminator(module):
    generator = [*module.decoder.parameters()]
    if hasattr(module, "bottleneck"):
        generator += [*module.bottleneck.parameters()]
    return generator, [*module.discriminator.parameters()]


class Batches(torch.utils.data.IterableDataset):
    """Different audio on every rank, so only all-reduced gradients can agree."""

    def __init__(self, model: str, rank: int):
        self.model, self.rank = model, rank

    def __iter__(self):
        generator = torch.Generator().manual_seed(1000 + self.rank)
        for _ in range(BATCHES):
            if self.model == "ssl_vae":
                yield {"input_wav": 0.3 * torch.randn(2, 2, 4800, generator=generator)}
            else:
                ssl = {"input_features": torch.randn(2, 4, 160, generator=generator),
                       "attention_mask": torch.ones(2, 4, dtype=torch.long)}
                yield {"input_wav": 0.3 * torch.randn(2, 4 * 960, generator=generator),
                       "ssl_inputs": ssl}


def equal_across_ranks(tensors) -> bool:
    flat = torch.cat([t.detach().flatten() for t in tensors])
    gathered = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flat)
    return all(torch.equal(gathered[0], other) for other in gathered[1:])


def worker(rank: int, port: int, root: str) -> None:
    import hydra
    from lightning import Callback, Trainer

    os.environ.update(
        OMPI_COMM_WORLD_SIZE=str(WORLD_SIZE), OMPI_COMM_WORLD_RANK=str(rank),
        OMPI_COMM_WORLD_LOCAL_RANK=str(rank), OMPI_COMM_WORLD_LOCAL_SIZE=str(WORLD_SIZE),
        MAIN_ADDR="127.0.0.1", MAIN_PORT=str(port),
    )
    torch.set_num_threads(1)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.simplefilter("ignore")

    class RecordSync(Callback):
        def __init__(self):
            self.steps, self.batches = [], []

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            grads = [p.grad for group in optimizer.param_groups for p in group["params"]]
            self.steps.append({
                "grads_equal": equal_across_ranks(grads),
                "nonzero": bool(any(g.abs().sum() > 0 for g in grads)),
            })

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.batches.append([equal_across_ranks(group) for group in generator_and_discriminator(pl_module)])

    results = {}
    fakes = tiny_fakes()
    for fake in fakes:
        fake.start()
    try:
        for train in ("gan", "default"):
            for model in MODELS:
                record = RecordSync()
                result = {}
                try:
                    cfg = compose(model, train)
                    if model == "ssl_vae":
                        cfg.model.cfg.vae.input_dim = 16
                    torch.manual_seed(0)
                    module = hydra.utils.instantiate(cfg.model.lightning_module, cfg=cfg.model.cfg)
                    trainer = Trainer(callbacks=[record], **hydra.utils.instantiate(cfg.train.trainer))
                    trainer.fit(module, train_dataloaders=torch.utils.data.DataLoader(
                        Batches(model, rank), batch_size=None))
                except Exception as error:  # reported by the test on rank 0
                    result["error"] = f"{type(error).__name__}: {error}"
                result.update(steps=record.steps, batches=record.batches)
                results[f"{train}/{model}"] = result
    finally:
        for fake in fakes:
            fake.stop()
    if rank == 0:
        (Path(root) / "sync.json").write_text(json.dumps(results))


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "needs torch.distributed with gloo")
class GanDDPSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        with socket.socket() as stream:
            stream.bind(("127.0.0.1", 0))
            port = stream.getsockname()[1]
        mp.spawn(worker, args=(port, directory.name), nprocs=WORLD_SIZE, join=True)
        cls.results = json.loads((Path(directory.name) / "sync.json").read_text())

    def test_gan_config_all_reduces_both_updates(self):
        for model in MODELS:
            with self.subTest(model=model):
                result = self.results[f"gan/{model}"]
                self.assertIsNone(result.get("error"))
                self.assertEqual(len(result["steps"]), 2 * BATCHES)
                for index, step in enumerate(result["steps"]):
                    self.assertTrue(step["nonzero"], f"step {index}: all gradients are zero")
                    self.assertTrue(step["grads_equal"], f"step {index}: gradients differ across ranks")
                # [generator equal, discriminator equal] after every batch
                self.assertEqual(result["batches"], [[True, True]] * BATCHES)

    def test_default_config_never_trains_diverging_replicas(self):
        # static_graph=True must either be rejected up front or keep the
        # ranks in sync; it must not silently train one replica per rank.
        for model in MODELS:
            with self.subTest(model=model):
                result = self.results[f"default/{model}"]
                if result.get("error") is None:
                    self.assertEqual(len(result["steps"]), 2 * BATCHES)
                    for index, step in enumerate(result["steps"]):
                        self.assertTrue(step["grads_equal"], f"step {index}: gradients differ across ranks")
                    self.assertEqual(result["batches"], [[True, True]] * BATCHES)
                else:
                    self.assertIn("static_graph=True", result["error"])
                    self.assertEqual(result["steps"], [])


class GhostLossTests(unittest.TestCase):
    def test_zero_value_and_zero_gradients(self):
        from sidon.model.gan_ddp import ghost_loss

        layer = torch.nn.Linear(3, 2)
        frozen = torch.nn.Linear(3, 2).requires_grad_(False)
        loss = ghost_loss([*layer.parameters(), *frozen.parameters()])
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        for parameter in layer.parameters():
            self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter)))
        for parameter in frozen.parameters():
            self.assertIsNone(parameter.grad)


class LaunchSettingsTests(unittest.TestCase):
    def test_gan_launchers_use_gan_train_config(self):
        model = re.compile(r"^\s*model=(\S+)", re.MULTILINE)
        checked = 0
        for script in sorted((REPO / "scripts/pbs").rglob("*.sh")):
            text = script.read_text()
            if any(name in GAN_MODELS for name in model.findall(text)):
                checked += 1
                self.assertRegex(text, r"(?m)^\s*train=gan\b", script.name)
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
