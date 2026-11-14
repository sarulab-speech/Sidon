"""The batched feature front end against torchaudio's Kaldi filter bank and the per-waveform reference extractor."""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dialoguesidon2_fixtures import dialogue_batch  # noqa: E402

from sidon.dialoguesidon2.audio import extract_features, kaldi_fbank, prepare_ssl_inputs, stack_features  # noqa: E402
from sidon.model.dialoguesidon2.functional import length_mask  # noqa: E402


class FeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def test_batched_fbank_matches_torchaudio_kaldi(self):
        from torchaudio.compliance import kaldi

        waves = torch.randn(3, 16000) * 0.1
        batched = kaldi_fbank(waves)
        for wave, actual in zip(waves, batched):
            expected = kaldi.fbank(wave[None], num_mel_bins=80, frame_length=25.0, frame_shift=10.0, dither=0.0,
                                   preemphasis_coefficient=0.97, remove_dc_offset=True, window_type="povey",
                                   round_to_power_of_two=True, sample_frequency=16000.0, low_freq=20.0,
                                   high_freq=0.0, use_energy=False, use_log_fbank=True, use_power=True,
                                   snip_edges=True)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_extract_features_matches_per_waveform_reference(self):
        from sidon.model.dialogue_sidion.audio import extract_seamless_m4t_features

        for rate, lengths in ((24000, [24000, 17040, 24000, 300]), (16000, [16000, 9999, 16000])):
            lengths = torch.tensor(lengths)
            mixture = torch.randn(len(lengths), int(lengths.max())) * 0.1
            mixture = mixture.masked_fill(~length_mask(lengths, mixture.size(1)), 0)
            actual, output_lengths = extract_features(mixture, lengths, rate)
            audio = []
            for wave, size in zip(mixture, lengths.tolist()):
                wave = wave[:size]
                if rate != 16000:
                    wave = torchaudio.functional.resample(wave, rate, 16000)
                audio.append(F.pad(wave, (160, 160 + max(0, 320 - wave.numel()))))
            expected = extract_seamless_m4t_features(audio, sampling_rate=16000, device="cpu")
            torch.testing.assert_close(actual["input_features"], expected["input_features"], atol=1e-4, rtol=1e-4)
            self.assertTrue(torch.equal(actual["attention_mask"], expected["attention_mask"]))
            self.assertTrue(torch.equal(output_lengths, (lengths * 24000 + rate - 1) // rate))

    def test_stacking_drops_an_odd_last_frame_from_the_mask(self):
        stacked = stack_features([torch.ones(5, 80), torch.ones(4, 80)])
        self.assertEqual(tuple(stacked["input_features"].shape), (2, 3, 160))
        self.assertEqual(stacked["attention_mask"].sum(1).tolist(), [2, 2])

    def test_prepare_ssl_inputs_computes_all_groups_like_single_calls(self):
        batch = dialogue_batch()
        prepared = prepare_ssl_inputs(batch, "cpu")
        for key, source in (("clean_ssl_inputs_0", batch["input_wav"][:, 0]),
                            ("clean_ssl_inputs_1", batch["input_wav"][:, 1]),
                            ("noisy_mixture_ssl_inputs", batch["noisy_mixture"])):
            single, _ = extract_features(source, batch["input_wav_lens"], 24000)
            self.assertTrue(torch.equal(prepared[key]["attention_mask"], single["attention_mask"]))
            torch.testing.assert_close(prepared[key]["input_features"], single["input_features"])
        # Existing groups are kept, not recomputed.
        again = prepare_ssl_inputs(prepared, "cpu")
        self.assertIs(type(again["clean_ssl_inputs_0"]), dict)
        self.assertTrue(torch.equal(again["noisy_mixture_ssl_inputs"]["input_features"],
                                    prepared["noisy_mixture_ssl_inputs"]["input_features"]))


if __name__ == "__main__":
    unittest.main()
