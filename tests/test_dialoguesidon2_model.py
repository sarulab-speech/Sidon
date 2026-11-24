"""The separation model on CPU with tiny random weights: PIT, gradient flow, optimizer and schedule, posterior
draws, efficient attention, inference masking and the release format."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dialoguesidon2_fixtures import LATENT, dialogue_batch, tiny_config, tiny_separator  # noqa: E402

from sidon.model.dialoguesidon2.functional import length_mask, masked_mse, pit_assignment  # noqa: E402
from sidon.model.dialoguesidon2.lightning_module import learning_rate_factor, safe_gradients  # noqa: E402
from sidon.model.dialoguesidon2.readout import split_speakers  # noqa: E402


class PITTests(unittest.TestCase):
    def test_assignment_picks_the_better_permutation_per_example(self):
        torch.manual_seed(0)
        target = torch.randn(3, 2, 7, 4)
        prediction = target.clone()
        prediction[1] = target[1].flip(0)  # example 1 predicted in the swapped order
        prediction += 0.01 * torch.randn_like(prediction)
        mask = length_mask(torch.tensor([7, 5, 6]), 7)
        ordered, loss = pit_assignment(prediction, target, mask)
        self.assertTrue(torch.equal(ordered[0], target[0]) and torch.equal(ordered[2], target[2]))
        self.assertTrue(torch.equal(ordered[1], target[1].flip(0)))
        self.assertAlmostEqual(float(loss), float(masked_mse(prediction, ordered, mask).mean()), places=6)
        self.assertLess(float(loss), 1e-3)

    def test_padding_does_not_enter_the_loss(self):
        target = torch.zeros(1, 2, 5, 3)
        prediction = torch.zeros(1, 2, 5, 3)
        prediction[..., 3:, :] = 100.0  # only padded frames differ
        _, loss = pit_assignment(prediction, target, length_mask(torch.tensor([3]), 5))
        self.assertEqual(float(loss), 0.0)

    def test_split_speakers_layout(self):
        joint = torch.arange(2 * 3 * 8).reshape(2, 3, 8).float()
        speakers = split_speakers(joint)
        self.assertEqual(tuple(speakers.shape), (2, 2, 3, 4))
        self.assertTrue(torch.equal(speakers[:, 1], joint[..., 4:]))


class SeparationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_gradients_reach_only_the_conditioner_and_the_readout(self):
        model = tiny_separator().train()
        torch.manual_seed(1)
        losses, count = model.losses(dialogue_batch())
        self.assertEqual(count, 2)
        losses["loss"].backward()
        for name, parameter in model.named_parameters():
            if name.startswith("vae.") or "masked_spec_embed" in name:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertTrue(parameter.requires_grad, name)
                self.assertIsNotNone(parameter.grad, name)
        self.assertFalse(model.vae.training)  # the frozen VAE stays in eval mode
        self.assertTrue(model.conditioner.training)

    def test_optimizer_groups_and_learning_rate_schedule(self):
        model = tiny_separator()
        configured = model.configure_optimizers()
        optimizer = configured["optimizer"]
        head, conditioner = optimizer.param_groups
        self.assertEqual({id(p) for p in head["params"]}, {id(p) for p in model.readout.parameters()})
        self.assertEqual({id(p) for p in conditioner["params"]},
                         {id(p) for n, p in model.conditioner.named_parameters() if "masked_spec_embed" not in n})
        self.assertEqual(head["betas"], (0.9, 0.95))
        scheduler = configured["lr_scheduler"]["scheduler"]
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.0, 0.0])  # warmup starts at 0
        factor = scheduler.lr_lambdas[0]
        self.assertEqual(factor(1000), 0.5)
        self.assertEqual(factor(2000), 1.0)
        self.assertEqual(factor(159999), 1.0)
        self.assertEqual(factor(180000), 0.5)
        self.assertEqual(factor(200000), 0.0)
        self.assertEqual(learning_rate_factor(10, 0, None, 100), 1.0)
        for _ in range(1000):
            scheduler.step()
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.5e-4, 1e-5])

    def test_safe_gradients_clips_like_clip_grad_norm_and_skips_non_finite(self):
        model, reference = nn.Linear(4, 3), nn.Linear(4, 3)
        reference.load_state_dict(model.state_dict())
        x = torch.randn(5, 4) * 10
        model(x).square().sum().backward()
        reference(x).square().sum().backward()
        valid, norm = safe_gradients(model, max_abs_norm=None)
        expected = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        self.assertTrue(valid)
        torch.testing.assert_close(norm, expected)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        model.weight.grad[0, 0] = float("nan")
        valid, _ = safe_gradients(model, max_abs_norm=None)
        self.assertFalse(valid)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_posterior_draws_are_repeatable_and_keyed(self):
        model = tiny_separator().eval()
        from sidon.dialoguesidon2.audio import prepare_ssl_inputs
        from sidon.model.dialoguesidon2.lightning_module import encode_speakers

        batch = prepare_ssl_inputs(dialogue_batch(), "cpu")

        def targets(*parts):
            with model.vae.posterior_draws(*parts):
                return encode_speakers(model.vae.encode, batch, "cpu")[0]

        first, again = targets("val_posterior", 0, 0), targets("val_posterior", 0, 0)
        other = targets("val_posterior", 0, 1)
        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(first, other))
        frames = encode_speakers(model.vae.encode, batch, "cpu")[1]
        self.assertTrue(torch.all(first[1, :, int(frames[1]):] == 0))  # padding stays zero
        model.vae.sample_posterior = False
        mean_a, mean_b = targets("a"), targets("b")
        self.assertTrue(torch.equal(mean_a, mean_b))  # the posterior mean draws nothing

    def test_efficient_attention_matches_the_transformers_implementation(self):
        from transformers import Wav2Vec2BertModel

        from sidon.model.dialoguesidon2.ssl_attention import optimize_relative_key_attention

        torch.manual_seed(0)
        reference = Wav2Vec2BertModel(tiny_config(num_hidden_layers=2)).eval()
        optimized = Wav2Vec2BertModel(tiny_config(num_hidden_layers=2)).eval()
        optimized.load_state_dict(reference.state_dict())
        self.assertEqual(optimize_relative_key_attention(optimized), 2)
        self.assertEqual(set(optimized.state_dict()), set(reference.state_dict()))
        x = torch.randn(2, 90, 160, requires_grad=True)
        expected = reference(input_features=x).last_hidden_state
        actual = optimized(input_features=x).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        expected.square().sum().backward()
        grads = [p.grad.clone() for p in reference.parameters() if p.grad is not None]
        actual.square().sum().backward()
        for got, want in zip([p.grad for p in optimized.parameters() if p.grad is not None], grads):
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

    def test_inference_shapes_and_padding_independence(self):
        model = tiny_separator().eval()
        batch = dialogue_batch(seconds=(2.0, 1.2))
        waves, lengths = model(batch["noisy_mixture"], batch["input_wav_lens"], 24000)
        self.assertEqual(tuple(waves.shape), (2, 2, batch["noisy_mixture"].size(1)))
        self.assertTrue(torch.equal(lengths, batch["input_wav_lens"]))
        n = int(lengths[1])
        self.assertTrue(torch.all(waves[1, :, n:] == 0))
        alone, _ = model(batch["noisy_mixture"][1:, :n], lengths[1:], 24000)
        torch.testing.assert_close(waves[1, :, :n], alone[0], atol=1e-5, rtol=1e-5)
        with self.assertRaises(RuntimeError):
            model.train()(batch["noisy_mixture"], batch["input_wav_lens"], 24000)

    def test_release_round_trip_is_strict(self):
        import json

        from safetensors.torch import load_file, save_file

        from sidon.dialoguesidon2.release import load_release, save_release

        model = tiny_separator(seed=3)
        model.cfg["vae_checkpoint"] = "/some/local/vae.ckpt"
        with tempfile.TemporaryDirectory() as directory:
            save_release(model, directory, global_step=7)
            config = json.loads((Path(directory) / "config.json").read_text())
            self.assertIsNone(config["cfg"]["vae_checkpoint"])
            self.assertEqual(config["global_step"], 7)
            reloaded = load_release(directory)
            self.assertFalse(reloaded.training)
            expected = model.state_dict()
            self.assertEqual(set(reloaded.state_dict()), set(expected))
            for key, value in reloaded.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)
            state = load_file(str(Path(directory) / "model.safetensors"))
            state.pop("readout.head.2.bias")
            save_file(state, str(Path(directory) / "model.safetensors"))
            with self.assertRaises(RuntimeError):
                load_release(directory)
            state["readout.head.2.bias"] = torch.zeros(2 * LATENT)
            state["readout.extra"] = torch.zeros(1)
            save_file(state, str(Path(directory) / "model.safetensors"))
            with self.assertRaises(RuntimeError):
                load_release(directory)

    def test_statistics_of_other_encoder_weights_are_refused(self):
        model = tiny_separator()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statistics.pt"
            torch.save({"mean": torch.zeros(LATENT), "var": torch.ones(LATENT), "count": 10,
                        "encoder_identity": "0" * 64}, path)
            with self.assertRaises(ValueError):
                model.vae.load_statistics(str(path))
            torch.save({"mean": torch.zeros(LATENT), "var": torch.ones(LATENT), "count": 10,
                        "encoder_identity": model.vae.identity()}, path)
            model.vae.load_statistics(str(path))


if __name__ == "__main__":
    unittest.main()
