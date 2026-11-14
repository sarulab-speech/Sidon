"""The SSL-VAE stage on CPU with a tiny w2v-BERT, a 16-channel decoder and a one-scale discriminator."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dialoguesidon2_fixtures import dialogue_batch, tiny_stage  # noqa: E402

from sidon.model.dialoguesidon2.lightning_module import kl_divergence, posterior_speakers  # noqa: E402


def vae_batch(seconds: float = 2.0) -> dict:
    batch = dialogue_batch(seconds=(seconds, seconds), seed=4)
    batch["noisy_mixture"] = batch["input_wav"].mean(1)
    return batch


def run_step(module, batch, training=True):
    """One step with the module's own optimizers; returns the loss, the logged values and the optimizers."""
    configured = module.configure_optimizers()
    optimizers = [entry["optimizer"] for entry in configured]
    schedulers = [entry["lr_scheduler"]["scheduler"] for entry in configured]
    logged = {}
    with patch.object(module, "optimizers", return_value=tuple(optimizers)), \
            patch.object(module, "lr_schedulers", return_value=tuple(schedulers)), \
            patch.object(module, "manual_backward", side_effect=lambda loss, **kw: loss.backward(**kw)), \
            patch.object(module, "log", side_effect=lambda name, value, **kw: logged.__setitem__(
                name, torch.as_tensor(value).detach().clone())):
        loss = module.step(batch, training)
    return loss, logged, optimizers


class VAEStageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_training_step_updates_bottleneck_decoder_and_discriminator_only(self):
        module = tiny_stage().train()
        before = {name: value.detach().clone() for name, value in module.state_dict().items()}
        torch.manual_seed(0)
        loss, logged, optimizers = run_step(module, vae_batch())
        self.assertTrue(torch.isfinite(loss))
        for key in ("train/loss", "train/mel_loss", "train/adv_gen", "train/adv_feature", "train/kl_loss",
                    "train/discriminator"):
            self.assertIn(key, logged)
        self.assertEqual(float(logged["train/generator_skipped"]), 0.0)
        after = module.state_dict()
        changed = {name for name in before if not torch.equal(before[name], after[name])}
        self.assertTrue(any(name.startswith("vae.bottleneck.") for name in changed))
        self.assertTrue(any(name.startswith("vae.decoder.") for name in changed))
        self.assertTrue(any(name.startswith("discriminator.") for name in changed))
        self.assertFalse(any(name.startswith("vae.ssl_model.") for name in changed))
        generator, discriminator = optimizers
        self.assertEqual({id(p) for g in generator.param_groups for p in g["params"]},
                         {id(p) for p in [*module.vae.bottleneck.parameters(), *module.vae.decoder.parameters()]})
        self.assertEqual(generator.param_groups[0]["betas"], (0.8, 0.98))
        self.assertTrue(all(not p.requires_grad for p in module.vae.ssl_model.parameters()))

    def test_loss_is_the_weighted_sum_with_the_kl_term(self):
        module = tiny_stage().eval()
        with torch.no_grad():
            torch.manual_seed(3)
            loss, logged, _ = run_step(module, vae_batch(), training=False)
        expected = (15.0 * logged["val/mel_loss"] + 2.0 * logged["val/adv_gen"] + 1.0 * logged["val/adv_feature"]
                    + 1e-5 * logged["val/kl_loss"])
        torch.testing.assert_close(loss, expected)

    def test_kl_matches_the_closed_form_over_valid_frames(self):
        mu, logvar = torch.randn(2, 2, 5, 3), torch.randn(2, 2, 5, 3)
        lengths = torch.tensor([5, 3])
        expected = torch.cat([(-0.5 * (1 + logvar[i, :, :n] - mu[i, :, :n] ** 2 - logvar[i, :, :n].exp()).sum(-1))
                              .reshape(-1) for i, n in enumerate(lengths.tolist())]).mean()
        torch.testing.assert_close(kl_divergence(mu, logvar, lengths), expected)

    def test_posterior_draws_have_gradients_through_the_bottleneck_only(self):
        module = tiny_stage().train()
        z, mu, logvar, lengths = posterior_speakers(module.vae, vae_batch(1.0))
        self.assertEqual(tuple(z.shape[:2]), (2, 2))
        z.sum().backward()
        self.assertTrue(all(p.grad is not None for p in module.vae.bottleneck.parameters()))
        self.assertTrue(all(p.grad is None for p in module.vae.ssl_model.parameters()))

    def test_crops_respect_the_frame_limit_and_alignment(self):
        module = tiny_stage().train()
        batch = vae_batch(2.0)  # 100 frames, decoder_crop_frames 60
        torch.manual_seed(5)
        pairs, _ = module.reconstruction(batch, training=True)
        self.assertEqual(sum(count for _, _, count in pairs), 2)
        for prediction, target, count in pairs:
            self.assertEqual(prediction.shape, target.shape)
            self.assertLessEqual(prediction.size(-1), 60 * 480)
            self.assertEqual(prediction.size(0), 2 * count)

    def test_validation_draws_are_seeded(self):
        module = tiny_stage().eval()
        logged = []
        with patch.object(module, "log", side_effect=lambda name, value, **kw: logged.append((name, float(value)))):
            with torch.no_grad():
                torch.manual_seed(1)
                first = module.validation_step(vae_batch(), 0)
                torch.manual_seed(2)
                second = module.validation_step(vae_batch(), 0)
        self.assertEqual(float(first), float(second))


if __name__ == "__main__":
    unittest.main()
