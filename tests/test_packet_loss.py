"""Packet loss must not write into the clean target.

The preprocessing pipelines start "clean" and "noisy" as the same tensor
(rename_audio, then split_by_channel views for dialogues) and degrade only
"noisy". When no earlier degradation replaced "noisy", an in-place packet loss
also zeroed the clean target.
"""

import random
import unittest

import torch

from sidon.data.preprocess.degrations import DegrationApply
from sidon.data.preprocess.functional_degrations import packet_loss

SR = 16000


class PacketLossTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)

    def test_shared_tensor_keeps_clean_target(self):
        wav = torch.randn(1, 10 * SR)
        reference = wav.clone()
        sample = {"clean": (wav, SR), "noisy": (wav, SR)}  # as after rename_audio
        out = packet_loss(sample, input_key="noisy", output_key="noisy")
        torch.testing.assert_close(out["clean"][0], reference, rtol=0, atol=0)
        self.assertGreater(int((out["noisy"][0] == 0).sum()), 0)

    def test_channel_views_keep_clean_target(self):
        stereo = torch.randn(2, 10 * SR)
        reference = stereo.clone()
        # split_by_channel gives clean and noisy views of one storage.
        sample = {"clean": (stereo[0].unsqueeze(0), SR), "noisy": (stereo[0].unsqueeze(0), SR)}
        out = packet_loss(sample, input_key="noisy", output_key="noisy")
        torch.testing.assert_close(stereo, reference, rtol=0, atol=0)
        torch.testing.assert_close(out["clean"][0], reference[:1], rtol=0, atol=0)
        self.assertGreater(int((out["noisy"][0] == 0).sum()), 0)

    def test_class_based_packet_loss_leaves_input(self):
        wav = torch.randn(1, 10 * SR)
        reference = wav.clone()
        out = DegrationApply.packet_loss(None, wav, SR)
        torch.testing.assert_close(wav, reference, rtol=0, atol=0)
        self.assertGreater(int((out == 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
