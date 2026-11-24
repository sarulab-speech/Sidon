"""Tests of the simulated-conversation pipeline (scripts/dialoguesidon2/data/sim) on synthetic utterances only.

A small synthetic index (index.tsv and its PCM cache, built with index_libritts.analyse) stands in for LibriTTS-R;
plans use the shipped turn-taking table sim/turn_fit.json with fewer records and tars than the real run.
"""

from __future__ import annotations

import copy
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "scripts" / "dialoguesidon2" / "data"
sys.path.insert(0, str(DATA))

from sim import base as sc  # noqa: E402
from sim import cut, fit_turns, index_libritts as il, qc, simulate as sm  # noqa: E402

SPEAKERS = {"train-clean-100": [11, 12, 13, 14], "train-clean-360": [21, 22], "train-other-500": [31, 32]}
SPEECH_PER_SPEAKER_S = 135.0


def synthetic_utterance(rng: np.random.Generator, seconds: float) -> np.ndarray:
    """Band-passed noise bursts (syllable-like) with pauses, at a random level, 24 kHz int16."""
    from scipy.signal import butter, sosfilt
    n = int(seconds * sc.SR)
    noise = sosfilt(butter(4, [200, 4000], btype="bandpass", fs=sc.SR, output="sos"), rng.standard_normal(n))
    envelope = np.zeros(n)
    t = int(rng.uniform(0.05, 0.3) * sc.SR)
    while t < n - int(0.1 * sc.SR):
        d = int(rng.uniform(0.15, 1.2) * sc.SR)
        envelope[t:t + d] = rng.uniform(0.5, 1.0)
        t += d + int(rng.uniform(0.05, 0.7) * sc.SR)
    level = 10 ** (rng.uniform(-30, -12) / 20)
    x = level * noise / (noise.std() + 1e-9) * envelope + 1e-4 * rng.standard_normal(n)
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)


def build_index(root: Path, seed: int = 1) -> Path:
    """index.tsv and pcm/<subset>/data-00000.npy as index_libritts writes them."""
    rng = np.random.default_rng(seed)
    rows = []
    for subset, speakers in SPEAKERS.items():
        chunks, offset = [], 0
        for spk in speakers:
            total, k = 0.0, 0
            while total < SPEECH_PER_SPEAKER_S:
                chapter = spk * 10 + 1 + k % 2
                seconds = rng.uniform(0.5, 1.3) if rng.random() < 0.2 else rng.uniform(2.0, 9.0)
                x = synthetic_utterance(rng, seconds)
                result = il.analyse(x)
                utt = f"{spk}_{chapter}_{k // 3:06d}_{k % 3:06d}"
                row = {"utt": utt, "speaker": str(spk), "chapter": str(chapter), "para": k // 3, "sent": k % 3,
                       "subset": subset, "tar": 0, "member": f"{spk}/{chapter}/{utt}.wav.audio.wav",
                       "samples": len(x), "usable": result["usable"]}
                if result["usable"]:
                    row.update({key: result[key] for key in ("trim_start", "length", "active_db", "peak", "runs")},
                               pcm_offset=offset)
                    chunks.append(result["samples"])
                    offset += result["length"]
                rows.append(row)
                total += seconds
                k += 1
        (root / "pcm" / subset).mkdir(parents=True, exist_ok=True)
        np.save(root / "pcm" / subset / "data-00000.npy", np.concatenate(chunks).astype(np.int16))
    rows.sort(key=il.sort_key)
    sc.cm.write_tsv(root / "index.tsv", rows, il.COLUMNS)
    return root


class SimPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = copy.deepcopy(sc.PARAMS)
        sc.PARAMS["split"].update(train_tars=2, train_records=4, dev_records=2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.tmp.name)
        cls.index = build_index(cls.work / "index")
        cls.fit = fit_turns.load_fit()
        cls.utts = sm.load_utterances(cls.index / "index.tsv")
        cls.rows, cls.info = sm.plan(cls.utts, cls.fit)

    @classmethod
    def tearDownClass(cls):
        sc.PARAMS.clear()
        sc.PARAMS.update(cls.saved)
        cls.tmp.cleanup()

    def test_shipped_table(self):
        for name in ("turn_s", "pause_s", "fto_s", "interjection_s", "activity_union", "overlap_share"):
            table = self.fit["tables"][name]
            self.assertEqual(len(table["q"]), sc.QUANTILE_POINTS)
            self.assertEqual(table["q"], sorted(table["q"]))
            self.assertEqual(self.fit["summary"][name]["p50"], table["p50"])
        self.assertGreater(self.fit["rates"]["interjections_per_turn_s"], 0)

    def test_plan_is_deterministic_and_complete(self):
        again, _ = sm.plan(sm.load_utterances(self.index / "index.tsv"), fit_turns.load_fit())
        self.assertEqual(sm.plan_text(again), sm.plan_text(self.rows))
        counts = {split: sum(r["split"] == split for r in self.rows) for split in ("dev", "train")}
        self.assertEqual(counts, {"dev": 2, "train": 4})

    def test_styles_alternate_and_tars_balance(self):
        for r in self.rows:
            self.assertEqual(r["style"], sc.STYLES[r["n"] % 2])
            self.assertEqual(sc.parse_key(r["key"]), r["n"])
            if r["split"] == "train":
                self.assertEqual(r["tar"], (r["n"] // 2) % 2)

    def test_reuse_cap_and_disjoint_speakers(self):
        self.assertEqual(qc.reuse_problems(self.rows, sc.PARAMS["reuse"]["max_uses"]), [])
        speakers = {split: {r[f"speaker_{c}"] for r in self.rows if r["split"] == split for c in "ab"}
                    for split in ("dev", "train")}
        self.assertFalse(speakers["dev"] & speakers["train"])
        for r in self.rows:
            self.assertNotEqual(r["speaker_a"], r["speaker_b"])

    def test_levels_planned_at_target(self):
        level = sc.PARAMS["level"]
        active = {u.utt: u.active_db for u in self.utts}
        for r in self.rows:
            for c in "ab":
                for utt, _, gain, _ in sm.channel(r, c):
                    planned = gain + active[utt]
                    self.assertLessEqual(abs(planned - level["target_dbfs"]), level["jitter_db"] + 0.001)

    def test_render_cut_and_checks(self):
        import soundfile as sf
        work = self.work / "run"
        self.assertEqual(sm.plan_command(self.index / "index.tsv", fit_turns.SHIPPED, work / "clean", False, 4), 0)
        self.assertEqual(sm.render_command(self.index, work / "clean", 2), 0)
        rows = sm.read_plan(work / "clean" / "plan.tsv")
        for r in rows:
            for c in "ab":
                path = work / "clean" / sc.track_relpath(r["split"], r["tar"], r["key"], c)
                self.assertIsNone(sc.clean_problem(path))
                x, sr = sf.read(str(path), dtype="int16")
                self.assertEqual((len(x), sr), (sc.TRACK_SAMPLES, sc.SR))
                self.assertLessEqual(np.abs(x.astype(np.int32)).max(), round(0.99 * 32768) + 1)
                self.assertGreater(sc.vad_mask(x).mean(), sc.PARAMS["record"]["min_activity"])
        self.assertEqual(qc.s0(work, 2), 0)
        qc.s1(work, fit_turns.SHIPPED)  # a toy speaker pool may fail the top-10 share: proceed regardless
        self.assertEqual(cut.main(["--work", str(work), "--processes", "2", "--accept-s1-stop"]), 0)
        source = work / "source"
        self.assertEqual(sc.cm.load_json(source / "summary.json")["status"], "complete")
        manifest = sc.cm.read_tsv(source / "manifest.tsv")
        self.assertEqual(len(manifest), len(rows))
        with tarfile.open(source / "train" / "data-00000.tar") as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar}
        keys = sorted({name.split(".", 1)[0] for name in members})
        self.assertTrue(keys)
        for key in keys:
            self.assertEqual(sorted(n.split(".", 1)[1] for n in members if n.startswith(key)),
                             ["flac", "original_path.txt"])
            audio, sr = sf.read(io.BytesIO(members[f"{key}.flac"]), dtype="int16")
            self.assertEqual((audio.shape, sr), ((sc.OUT_FRAMES, 2), sc.OUT_SR))
            clean = [sf.read(str(work / "clean" / sc.track_relpath("train", 0, key, c)), dtype="int16")[0]
                     for c in "ab"]
            for ch in range(2):
                expected, _ = sc.upsample_48k(clean[ch])
                np.testing.assert_array_equal(audio[:, ch], expected)
            meta = json.loads(members[f"{key}.original_path.txt"])
            self.assertIsNone(meta["restored"])


class TurnSegmentationTests(unittest.TestCase):
    def test_turns_fto_and_interjections(self):
        a = np.zeros(1000, bool)
        b = np.zeros(1000, bool)
        a[0:300] = True       # turn 1 (A)
        b[350:600] = True     # turn 2 (B), 0.5 s gap
        a[400:450] = True     # interjection of A inside B's spurt
        a[580:900] = True     # turn 3 (A), 0.2 s overlap
        seg = sc.segment(a, b)
        self.assertEqual(seg["turns"], 3)
        self.assertEqual(seg["interjections"], 1)
        self.assertEqual(seg["fto_s"], [0.5, -0.2])
        self.assertEqual(seg["turn_s"], [2.5])


if __name__ == "__main__":
    unittest.main()
