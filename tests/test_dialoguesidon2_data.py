"""Data modules of both stages on tiny synthetic shards: segmenting, collation, pairing and seeded order."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dialoguesidon2_fixtures import write_dialogue_shards, write_sidon_shards  # noqa: E402

from sidon.dialoguesidon2.data import (DialogueDataModule, collate_dialogues, sample_seed, shard_seed,  # noqa: E402
                                       split_records)
from sidon.dialoguesidon2.sidon_tracks import SidonTrackDataModule, pair_segments  # noqa: E402


def names(loader, batches: int) -> list:
    result = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        result.append((tuple(batch["names"]), tuple(batch["input_wav_lens"].tolist())))
    return result


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.dialogue, cls.sidon = root / "dialogue", root / "sidon"
        write_dialogue_shards(cls.dialogue)
        write_sidon_shards(cls.sidon)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_split_records_cuts_near_equal_frame_aligned_segments(self):
        n = 120 * 24000
        record = {"__key__": "r", "sr.index": 24000, "input_wav.pth": torch.zeros(2, n),
                  "noisy_mixture.pth": torch.zeros(n), "input_wav_lens.pth": torch.tensor([n])}
        segments = list(split_records([record], seconds=30))
        self.assertEqual([s["__key__"] for s in segments], ["r/0/0", "r/0/720000", "r/0/1440000", "r/0/2160000"])
        self.assertTrue(all(int(s["input_wav_lens.pth"]) == 720000 for s in segments))
        odd = dict(record, **{"input_wav.pth": torch.zeros(2, 70000), "noisy_mixture.pth": torch.zeros(70000),
                              "input_wav_lens.pth": torch.tensor([70000])})
        lengths = [int(s["input_wav_lens.pth"]) for s in split_records([odd], seconds=1)]
        self.assertEqual(sum(lengths), 70000)
        self.assertTrue(all(length % 480 == 0 for length in lengths[:-1]))
        self.assertLessEqual(max(lengths), 50 * 480)

    def test_collate_pads_and_keeps_lengths(self):
        records = [{"__key__": f"k{i}", "sr.index": 24000, "input_wav.pth": torch.ones(2, n),
                    "noisy_mixture.pth": torch.ones(n), "input_wav_lens.pth": torch.tensor([n])}
                   for i, n in enumerate((4800, 3360))]
        batch = collate_dialogues(records)
        self.assertEqual(tuple(batch["input_wav"].shape), (2, 2, 4800))
        self.assertEqual(tuple(batch["noisy_mixture"].shape), (2, 4800))
        self.assertEqual(batch["input_wav_lens"].tolist(), [4800, 3360])
        self.assertEqual(float(batch["input_wav"][1, :, 3360:].abs().sum()), 0.0)
        self.assertEqual(batch["names"], ["k0", "k1"])

    def test_training_order_follows_the_data_seed(self):
        def order(seed):
            module = DialogueDataModule([str(self.dialogue / "train")], [str(self.dialogue / "valid")], batch_size=2,
                                        segment_seconds=1.0, data_seed=seed)
            return names(module.train_dataloader(), 5)

        first, again, other = order("a"), order("a"), order("b")
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual(shard_seed(0, "a"), shard_seed(0, "a"))
        self.assertNotEqual(shard_seed(0, "a"), shard_seed(1000, "a"))  # a resumed run reads a new order
        self.assertIsNone(sample_seed(0, "records", None))

    def test_validation_order_is_fixed_and_batches_are_well_formed(self):
        module = DialogueDataModule([str(self.dialogue / "train")], [str(self.dialogue / "valid")], batch_size=2,
                                    segment_seconds=1.0)
        first, again = names(module.val_dataloader(), 3), names(module.val_dataloader(), 3)
        self.assertEqual(first, again)
        batch = next(iter(module.train_dataloader()))
        self.assertEqual(batch["input_wav"].shape[:2], (2, 2))
        self.assertEqual(batch["noisy_mixture"].shape, (2, batch["input_wav"].size(-1)))
        self.assertLessEqual(int(batch["input_wav_lens"].max()), 50 * 480)

    def test_single_speaker_pairs_are_seeded_and_never_one_utterance(self):
        def order(seed):
            module = SidonTrackDataModule([str(self.sidon / "train")], [str(self.sidon / "valid")], batch_size=2,
                                          segment_seconds=0.5, data_seed=seed)
            batches = []
            for index, batch in enumerate(module.train_dataloader()):
                if index >= 4:
                    break
                self.assertEqual(tuple(batch["input_wav"].shape), (2, 2, 12000))
                self.assertEqual(batch["sr"], 24000)
                for name in batch["names"]:
                    first, second = name.split("+")
                    self.assertNotEqual(first.split("#")[0], second.split("#")[0])
                batches.append(tuple(batch["names"]))
            return batches

        self.assertEqual(order("x"), order("x"))
        self.assertNotEqual(order("x"), order("y"))

    def test_pairing_without_seed_keeps_the_reading_order(self):
        items = [("a", 0, torch.zeros(1)), ("a", 1, torch.zeros(1)), ("b", 0, torch.zeros(1)), ("c", 0, torch.zeros(1))]
        pairs = [(x[0], x[1], y[0], y[1]) for x, y in pair_segments(items, None)]
        self.assertEqual(pairs, [("a", 0, "b", 0), ("a", 1, "c", 0)])


if __name__ == "__main__":
    unittest.main()
