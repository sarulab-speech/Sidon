"""Degradations must keep "noisy" the same length and timing as "clean".

merge_samples_to_target_length concatenates the "clean" and "noisy" keys
independently, so any per-segment length difference accumulates into a
progressive clean/noisy time drift inside merged samples. band_limit's
down/up resample grows odd-length inputs by a few samples (ceil rounding),
and align_codec_waveform used to keep the head of a longer codec output
(retaining encoder delay) or slice a shorter one in the wrong direction.
"""

import random
import unittest

import torch

from sidon.data.preprocess.degrations import DegrationApply, align_codec_waveform
from sidon.data.preprocess.functional_degrations import band_limit

SR = 48000


class BandLimitLengthTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)

    def test_output_length_matches_input(self):
        # odd lengths used to grow by 1-3 samples through 48k->22050->48k
        for length in [SR * 4, SR * 4 + 1, SR * 4 + 7, 123457, 191999]:
            wav = torch.randn(1, length)
            sample = {"clean": (wav, SR)}
            out = band_limit(
                sample, candidate_srs=[22050], input_key="clean", output_key="noisy"
            )
            self.assertEqual(out["noisy"][0].size(1), length)

    def test_class_based_output_length_matches_input(self):
        apply = DegrationApply.__new__(DegrationApply)
        apply.downsample_sr = [22050]
        wav = torch.randn(1, SR * 4 + 7)
        out = apply.band_limit(wav, SR)
        self.assertEqual(out.size(1), wav.size(1))


class AlignCodecWaveformTests(unittest.TestCase):
    def test_longer_codec_output_is_realigned(self):
        # simulate an encoder delay: codec output is the original shifted
        # right by 100 samples with padding, as mp3 encode/decode produces
        original = torch.randn(1, 1000)
        codec_applied = torch.cat([torch.zeros(1, 100), original], dim=1)
        aligned = align_codec_waveform(original, codec_applied)
        self.assertEqual(aligned.size(1), original.size(1))
        torch.testing.assert_close(aligned, original, rtol=0, atol=0)

    def test_shorter_codec_output_keeps_timing(self):
        # codec output matching original[100:900] must land at offset 100,
        # not shift the content towards the head
        original = torch.randn(1, 1000)
        codec_applied = original[:, 100:900]
        aligned = align_codec_waveform(original, codec_applied)
        self.assertEqual(aligned.size(1), original.size(1))
        torch.testing.assert_close(
            aligned[:, 100:900], codec_applied, rtol=0, atol=0
        )
        self.assertTrue((aligned[:, :100] == 0).all())
        self.assertTrue((aligned[:, 900:] == 0).all())

    def test_equal_length_passthrough(self):
        original = torch.randn(1, 1000)
        codec_applied = torch.randn(1, 1000)
        torch.testing.assert_close(
            align_codec_waveform(original, codec_applied),
            codec_applied,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
