"""The producer (scripts/dialoguesidon2/data/producer): the mixing step, chunking and the legacy degradations.

The mixing tests need only PyTorch. The rest imports the degradations, which need pyroomacoustics; they are skipped
when it cannot be loaded."""

import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dialoguesidon2" / "data"))

from producer import mixing  # noqa: E402

try:
    from producer import datamodule as dm  # noqa: E402
    from producer import degradations as dg  # noqa: E402
    DEGRADATIONS = None
except Exception as error:  # pyroomacoustics or torchaudio extensions unavailable
    DEGRADATIONS = f"{type(error).__name__}: {error}"

SR = 24_000


def sample(key: str, seconds: float = 2.0, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    clean = torch.randn(2, int(seconds * SR), generator=g)
    noisy = clean + 0.1 * torch.randn(clean.shape, generator=g)
    c16 = torch.randn(2, int(seconds * 16_000), generator=g)
    n16 = c16 + 0.1 * torch.randn(c16.shape, generator=g)
    return {"__key__": key, "clean": (clean, SR), "noisy": (noisy, SR), "clean_16k": (c16, 16_000),
            "noisy_16k": (n16, 16_000)}


class MixingTests(unittest.TestCase):
    def test_mix_is_a_weighted_sum_of_the_channels(self):
        wavs = torch.stack([torch.ones(1, 5), 3 * torch.ones(1, 5)], 1)  # (1, 2, 5)
        torch.testing.assert_close(mixing.mix(wavs, 0.25), torch.full((1, 5), 0.25 + 0.75 * 3))

    def test_collate_record_layout(self):
        random.seed(0)
        record = mixing.collate([sample("src-00000_0")], SR, use_noise=True, max_duration=2.0)
        self.assertEqual(list(record), ["input_wav", "input_wav_lens", "sr", "names", "sample_rate", "clean_mixture",
                                        "clean_16k_mixture", "noisy_mixture", "noisy_16k_mixture",
                                        "noisy_input_wav16k", "noisy_input_wav"])
        self.assertEqual(tuple(record["input_wav"].shape), (1, 2, 2 * SR))
        self.assertEqual(tuple(record["noisy_input_wav16k"].shape), (1, 2, 2 * 16_000))
        self.assertEqual(tuple(record["clean_mixture"].shape), (1, 2 * SR))
        self.assertEqual(record["names"], ["src-00000_0"])
        self.assertEqual(record["sr"], SR)
        self.assertEqual(int(record["input_wav_lens"][0]), 2 * SR)

    def test_one_ratio_for_all_mixtures_within_range(self):
        for seed in range(20):
            random.seed(seed)
            expected = random.uniform(*mixing.RATIO_RANGE)
            random.seed(seed)
            s = sample("k", seed=seed)
            record = mixing.collate([s], SR, max_duration=2.0)
            self.assertTrue(0.3 <= expected <= 0.7)
            for target, (wav, _) in (("clean_mixture", s["clean"]), ("noisy_mixture", s["noisy"]),
                                     ("clean_16k_mixture", s["clean_16k"]), ("noisy_16k_mixture", s["noisy_16k"])):
                torch.testing.assert_close(record[target][0], wav[0] * expected + (1 - expected) * wav[1])

    def test_targets_are_padded_to_the_window(self):
        record = mixing.collate([sample("k", seconds=1.5)], SR, max_duration=2.0)
        self.assertEqual(record["input_wav"].shape[-1], 2 * SR)
        self.assertEqual(int(record["input_wav_lens"][0]), int(1.5 * SR))


@unittest.skipIf(DEGRADATIONS is not None, f"producer degradations unavailable: {DEGRADATIONS}")
class ProducerPipelineTests(unittest.TestCase):
    def test_chunk_keys_and_quiet_filter(self):
        g = torch.Generator().manual_seed(1)
        wav = 0.5 * torch.randn(2, int(7.5 * SR), generator=g).clamp(-1, 1)
        wav[:, 6 * SR:] *= 0.01
        chunks = list(dm.chunk_crop([{"__key__": "src", "audio": (wav, SR)}], chunk_seconds=3.0))
        self.assertEqual([c["__key__"] for c in chunks], ["src_0", f"src_{3 * SR}", f"src_{6 * SR}"])
        self.assertEqual([dm.select_non_silent_chunk(c, key="audio") for c in chunks], [True, True, False])
        one = list(dm.chunk_crop([{"__key__": "short", "audio": (wav[:, :SR], SR)}], chunk_seconds=3.0))
        self.assertEqual([c["__key__"] for c in one], ["short_0"])

    def test_legacy_packet_loss_reaches_a_shared_clean_target(self):
        wav = torch.randn(2, 10 * SR)
        s = {"clean": (wav, SR), "noisy": (wav, SR)}
        s = dm.split_by_channel(dm.split_by_channel(s, 0, "noisy"), 0, "clean")
        random.seed(0)
        out = dg.packet_loss(s, input_key="noisy", output_key="noisy")
        zeros = out["noisy"][0] == 0
        self.assertGreater(int(zeros.sum()), 0)
        self.assertTrue(torch.equal(out["clean"][0] == 0, zeros))

    def test_upstream_packet_loss_keeps_the_clean_target(self):
        wav = torch.randn(1, 10 * SR)
        reference = wav.clone()
        random.seed(0)
        out = dg.functions("upstream")["packet_loss"]({"clean": (wav, SR), "noisy": (wav, SR)},
                                                       input_key="noisy", output_key="noisy")
        self.assertGreater(int((out["noisy"][0] == 0).sum()), 0)
        torch.testing.assert_close(out["clean"][0], reference, rtol=0, atol=0)

    def test_legacy_band_limit_may_change_the_length(self):
        lengths = set()
        for seed in range(12):
            random.seed(seed)
            x = torch.randn(1, 3 * SR + 21)
            out = dg.band_limit({"noisy": (x, SR)}, candidate_srs=dm.BAND_LIMIT_RATES, input_key="noisy",
                                output_key="noisy")
            lengths.add(out["noisy"][0].shape[-1] - x.shape[-1])
        self.assertTrue(lengths - {0}, lengths)

    def test_legacy_codec_alignment_keeps_a_shorter_output_shorter(self):
        x = torch.randn(1, 1000)
        self.assertEqual(dg.align_codec_waveform(x, torch.randn(1, 1100)).shape[-1], 1000)
        # a shorter output is sliced from its best offset (10 here), which shortens it further: 950 - 10
        self.assertEqual(dg.align_codec_waveform(x, x[:, 10:960].clone()).shape[-1], 940)

    def test_reverberation_respects_the_rt60_range(self):
        random.seed(3)
        x = torch.randn(1, SR // 2)
        out = dg.convolve_rir_pra({"clean": (x, SR)}, input_key="clean", direct_key="clean", reverb_key="noisy",
                                  rt60_min_max=(0.1, 1.0))
        self.assertEqual(out["noisy"][0].shape, x.shape)
        self.assertEqual(out["clean"][0].shape, x.shape)
        peak = max(out["noisy"][0].abs().max(), out["clean"][0].abs().max())
        torch.testing.assert_close(peak, x.abs().max())

    def test_merge_keeps_channel_order(self):
        a = {"__key__": "k", "clean": (torch.zeros(1, 4), SR), "noisy": (torch.zeros(1, 4), SR)}
        b = {"__key__": "k", "clean": (torch.ones(1, 4), SR), "noisy": (torch.ones(1, 4), SR)}
        merged = dm.merge_dataset(a, ["clean", "noisy"], iter([b]))
        self.assertTrue(torch.equal(merged["clean"][0], torch.cat([torch.zeros(1, 4), torch.ones(1, 4)])))
        with self.assertRaises(ValueError):
            dm.merge_dataset(a, ["clean"], iter([{**b, "__key__": "other"}]))

    def test_unknown_degradation_mode(self):
        with self.assertRaises(ValueError):
            dg.functions("newest")


class ConfigTests(unittest.TestCase):
    def test_producer_config_targets_the_producer_module(self):
        text = (Path(__file__).resolve().parents[1] / "config" / "data" / "dialoguesidon2_producer.yaml").read_text()
        self.assertIn("_target_: producer.datamodule.DialogueProducerDataModule", text)
        for line in ("max_duration: 120.0", "sampling_rate: 24000", "rt60_max: 1.0", "n_shards: 16",
                     "degradations: legacy"):
            self.assertIn(line, text)


if __name__ == "__main__":
    unittest.main()
