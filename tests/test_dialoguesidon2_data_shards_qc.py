"""Checks of the producer's output shards (scripts/dialoguesidon2/data/common/shards_qc.py), on synthetic tars.

The real records are about 100 MB each, so the expected member sizes are patched to small values here."""

import io
import json
import pickle
import stat
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dialoguesidon2" / "data"))

from common import core as cm  # noqa: E402
from common import shards_qc as sq  # noqa: E402

SMALL = {name: (None if size is None else 100 + i) for i, (name, size) in enumerate(sq.MEMBER_SIZES.items())}


def write_shard(path: Path, keys: list[str], sizes: dict = SMALL, drop: str | None = None, wrong: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for key in keys:
            for name, size in sizes.items():
                if name == drop:
                    continue
                if name == "names.pickle":
                    data = pickle.dumps([f"{key}_0"])
                else:
                    data = b"\0" * (size + (1 if name == wrong else 0))
                info = tarfile.TarInfo(f"{key}.{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def write_source(root: Path, train_tars: int, windows_per_tar: int, dev_windows: int) -> None:
    """A source directory as the cut stages write it (empty tars; only names and the manifest matter)."""
    rows = []
    for split, tars, per in (("train", train_tars, windows_per_tar), ("dev", 1, dev_windows)):
        for t in range(tars):
            target = root / split / f"data-{t:05d}.tar"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
            rows += [{"key": f"w-{split}-{t}-{i}", "split": split, "tar": t} for i in range(per)]
    cm.write_tsv(root / "manifest.tsv", rows, ["key", "split", "tar"])


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_complete_record_passes(self):
        path = self.tmp / "a.tar"
        write_shard(path, [f"{1:032x}", f"{2:032x}"])
        with mock.patch.dict(sq.MEMBER_SIZES, SMALL):
            scan = sq.scan_tar(path)
        self.assertEqual(scan["records"], 2)
        self.assertEqual(scan["problems"], [])

    def test_missing_member_and_wrong_size_are_problems(self):
        write_shard(self.tmp / "b.tar", [f"{1:032x}"], drop="noisy_mixture.pth")
        write_shard(self.tmp / "c.tar", [f"{1:032x}"], wrong="input_wav.pth")
        with mock.patch.dict(sq.MEMBER_SIZES, SMALL):
            missing = sq.scan_tar(self.tmp / "b.tar")["problems"]
            wrong = sq.scan_tar(self.tmp / "c.tar")["problems"]
        self.assertEqual(len(missing), 1)
        self.assertIn("members", missing[0])
        self.assertEqual(len(wrong), 1)
        self.assertIn("input_wav.pth", wrong[0])

    def test_real_layout_sizes(self):
        """A 2 x 120 s float32 tensor saved with torch.save is 23,041,577 bytes; the layout derives from that."""
        self.assertEqual(sq.MEMBER_SIZES["input_wav.pth"] - 1577, 2 * 120 * 24_000 * 4)
        self.assertEqual(sq.MEMBER_SIZES["noisy_input_wav16k.pth"] - 1577, 2 * 120 * 16_000 * 4)
        self.assertEqual(sq.MEMBER_SIZES["clean_mixture.pth"] - 1577, 120 * 24_000 * 4)
        self.assertEqual(len(sq.MEMBER_SIZES), 11)


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "source"
        write_source(self.source, train_tars=4, windows_per_tar=3, dev_windows=2)
        self.shards = self.tmp / "out" / "writer"

    def produce(self, tasks: int, shortfall: dict | None = None):
        shortfall = shortfall or {}
        expected = sq.expected_per_task(self.source, tasks)
        for index in range(tasks):
            name = f"t{index:02d}"
            count = expected[name]["records"] - shortfall.get(name, 0)
            write_shard(self.shards / "train" / name / "worker-0-dataset-000000.tar",
                        [f"{name}{i:030x}" for i in range(count)])
        write_shard(self.shards / "valid" / "t00" / "worker-0-dataset-000000.tar",
                    [f"v{i:031x}" for i in range(expected["dev"])])
        return expected

    def test_expected_per_task_follows_the_producer_split(self):
        expected = sq.expected_per_task(self.source, 3)
        # np.array_split of 4 tars into 3 tasks: 2, 1, 1 tars of 3 windows each
        self.assertEqual([expected[f"t{i:02d}"]["records"] for i in range(3)], [6, 3, 3])
        self.assertEqual(expected["dev"], 2)
        listed = [p for part in sq.producer_partition(self.source / "train", 3) for p in part]
        self.assertEqual(len(listed), 4)

    def test_task_check_and_finalize(self):
        expected = self.produce(2)
        with mock.patch.dict(sq.MEMBER_SIZES, SMALL):
            for index in range(2):
                record = sq.task_check(self.shards, f"t{index:02d}", index, expected, None)
                self.assertTrue(record["passed"], record)
            self.assertEqual(sq.finalize(self.shards, self.source, tasks=2), 0)
        self.assertEqual((self.shards / "N_SHARDS.txt").read_text().strip(), "2")
        mode = (self.shards / "train" / "t00").stat().st_mode
        self.assertFalse(mode & stat.S_IWUSR)
        report = json.loads((self.shards / "qc_final.json").read_text())["report"]
        self.assertEqual(report["train_records"], 12)

    def test_shortfall_needs_an_allowance(self):
        expected = self.produce(2, shortfall={"t01": 1})
        with mock.patch.dict(sq.MEMBER_SIZES, SMALL):
            self.assertFalse(sq.task_check(self.shards, "t01", 1, expected, None)["passed"])
            self.assertTrue(sq.task_check(self.shards, "t01", 1, expected, None, max_shortfall=1)["passed"])
            sq.task_check(self.shards, "t00", 0, expected, None)
            strict = sq.headers(self.shards, self.source, tasks=2)
            lenient = sq.headers(self.shards, self.source, tasks=2, max_shortfall=1)
        self.assertFalse(strict["pass"])
        self.assertTrue(lenient["pass"])
        self.assertEqual(lenient["task_shortfalls"], {"t01": {"found": 5, "expected": 6}})

    def test_log_errors_fail_a_task(self):
        expected = self.produce(2)
        log = self.tmp / "t00.log"
        log.write_text("Traceback (most recent call last):\nValueError: boom\n")
        with mock.patch.dict(sq.MEMBER_SIZES, SMALL):
            record = sq.task_check(self.shards, "t00", 0, expected, log)
        self.assertFalse(record["passed"])
        self.assertFalse((self.shards / "_done" / "t00.json").exists())

    def test_duplicate_keys_fail(self):
        self.produce(2)
        write_shard(self.shards / "train" / "t01" / "worker-0-dataset-000001.tar", [f"t00{0:030x}"])
        with mock.patch.dict(sq.MEMBER_SIZES, SMALL):
            report = sq.headers(self.shards, None, tasks=2)
        self.assertEqual(report["duplicate_keys"], 1)
        self.assertFalse(report["pass"])


class AudioStatsTests(unittest.TestCase):
    def test_zero_gaps_and_level(self):
        rng = np.random.default_rng(0)
        x = (0.1 * rng.standard_normal(24_000 * 10)).astype(np.float32)
        for start in (24_000, 72_000, 120_000):
            x[start:start + 2_400] = 0  # 100 ms packets
        x[200_000:200_100] = 0  # too short to count
        stats = sq.channel_stats(x)
        self.assertEqual(stats["zero_gaps_20_200ms"], 3)
        self.assertAlmostEqual(stats["active_db"], 20 * np.log10(0.1), delta=1.5)
        energy = sq.frame_energy(x, 24_000)
        self.assertEqual(len(energy), 1000)


if __name__ == "__main__":
    unittest.main()
