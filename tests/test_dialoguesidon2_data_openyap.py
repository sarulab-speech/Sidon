"""OpenYap-1k data pipeline on synthetic conversations: the metadata rules, the crosstalk screen and the cut.

No real audio: tracks are band-passed noise inside synthetic word spans (speech-like energy), with an optional
delayed, attenuated copy of the partner (bleed).
"""

import io
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "scripts" / "dialoguesidon2" / "data"
sys.path.insert(0, str(DATA))

from openyap import corpus as oc  # noqa: E402
from openyap import cut_windows as cw  # noqa: E402
from openyap import prescreen as pm  # noqa: E402
from openyap import screen_crosstalk as sc  # noqa: E402

SR = 48_000


def turn_words(seed: int, duration: float) -> dict[str, list[dict]]:
    """Alternating turns of 1.5-5 s with words of 0.2-0.5 s. Every word is distinct, so no word is mistaken for a
    transcribed copy of the partner's (echo words are removed from the masks)."""
    rng = np.random.default_rng(seed)
    words = {"a": [], "b": []}
    t, cur = 0.5, "a"
    while t < duration - 2.0:
        end = min(t + float(rng.uniform(1.5, 5.0)), duration - 1.0)
        while t < end:
            d = float(rng.uniform(0.2, 0.5))
            words[cur].append({"word": f"word{len(words['a']) + len(words['b']):05d}", "start": round(t, 3),
                               "end": round(t + d, 3), "type": "word"})
            t += d + float(rng.uniform(0.05, 0.25))
        t += float(rng.uniform(0.2, 1.0))
        cur = "b" if cur == "a" else "a"
    return words


def speech_tracks(words: dict, samples: int, seed: int, bleed=None) -> dict[str, np.ndarray]:
    """int16 tracks: band-passed noise at about -20 dBFS inside words, -70 dBFS outside; bleed = (x, y, gain_db,
    lag_samples) adds y into x."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, [300, 3000], btype="bandpass", fs=SR, output="sos")
    rng = np.random.default_rng(seed)
    out = {}
    for name, items in words.items():
        envelope = np.full(samples, 10 ** (-70 / 20) / 0.1)
        for w in items:
            envelope[int(w["start"] * SR):int(w["end"] * SR)] = 1.0
        out[name] = 0.1 * sosfilt(sos, rng.standard_normal(samples)) / 0.335 * envelope
    if bleed is not None:
        x, y, gain_db, lag = bleed
        out[x] = out[x].copy()
        out[x][lag:] += 10 ** (gain_db / 20) * out[y][:samples - lag]
    return {k: np.round(np.clip(v, -1, 1) * 32767).astype(np.int16) for k, v in out.items()}


def write_conversation(root: Path, conv: str, duration: float, seed: int, bleed=None) -> dict:
    import soundfile as sf
    words = turn_words(seed, duration)
    tracks = speech_tracks(words, int(duration * SR), seed + 1, bleed)
    (root / conv).mkdir(parents=True)
    for s in "ab":
        sf.write(root / conv / f"speaker_{s}.flac", tracks[s], SR, subtype="PCM_16")
        (root / conv / f"speaker_{s}_transcript.json").write_text(json.dumps({"words": words[s]}))
    return tracks


class PrescreenRules(unittest.TestCase):
    def test_transcript_copies_are_flagged(self):
        words = turn_words(1, 300.0)
        self.assertFalse(pm.transcript_bleed(pm.lag_flag(words)))
        copies = [{**w, "start": w["start"] + 0.5, "end": w["end"] + 0.5} for w in words["b"] if len(w["word"]) >= 4]
        words["a"] = sorted(words["a"] + copies, key=lambda w: w["start"])
        flag = pm.lag_flag(words)
        self.assertTrue(pm.transcript_bleed(flag))
        # The statistic keeps the direction with the larger excess per own word, which need not be the track that
        # holds the copies; either way the peak lies at the copy lag.
        self.assertAlmostEqual(abs(flag["lstar"]), 0.5, delta=0.04)

    def test_conversation_and_window_rules(self):
        base = {"transcript_bleed": False, "silence_profile_a": "continuous", "silence_profile_b": "continuous",
                "dnsmos_ovr_median_a": 3.0, "dnsmos_ovr_median_b": 3.0, "noise_floor_dbfs_a": -60.0,
                "noise_floor_dbfs_b": None}
        self.assertEqual(pm.conversation_reasons(base), [])
        self.assertEqual(pm.conversation_reasons({**base, "silence_profile_b": "dropout"}), ["dropout"])
        self.assertEqual(pm.conversation_reasons({**base, "dnsmos_ovr_median_a": 2.4}), ["ovr_median_a"])
        self.assertEqual(pm.conversation_reasons({**base, "noise_floor_dbfs_b": -35.0}), ["noise_floor_b"])
        window = {"union": 0.8, "sp_a": 0.4, "sp_b": 0.3, "ovr_a": 3.0, "ovr_b": None}
        self.assertTrue(pm.window_ok(window))
        self.assertFalse(pm.window_ok({**window, "union": 0.45}))
        self.assertFalse(pm.window_ok({**window, "sp_b": 0.04}))
        self.assertFalse(pm.window_ok({**window, "ovr_a": 2.2}))

    def test_window_keys(self):
        key = oc.window_key("conv_000000000001", 240)
        self.assertEqual(key, "openyap-000000000001-00240")
        self.assertEqual(oc.parse_window_key(key + "_0"), ("conv_000000000001", 240))


class CrosstalkScreen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "conversations"
        cls.out = cls.tmp / "screen"
        write_conversation(cls.root, "conv_000000000001", 130.0, 10)
        write_conversation(cls.root, "conv_000000000002", 130.0, 20, bleed=("a", "b", -25.0, 4800))
        for conv in oc.conversation_ids(cls.root):
            sc.screen_conversation((cls.root, cls.out, conv, cls.tmp, "test"))
        cls.tracks, cls.windows = sc.collect(cls.out, oc.conversation_ids(cls.root), "test")
        cls.th = sc.load_decisions()["thresholds"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def track(self, conv, s):
        return next(t for t in self.tracks if t["conv"] == conv and t["s"] == s)

    def test_bleed_is_found_at_its_lag(self):
        bleeding = self.track("conv_000000000002", "a")
        self.assertEqual(bleeding["lag"], 10)  # 4,800 samples = 0.1 s = 10 frames
        self.assertGreater(bleeding["prominence"], 0.3)
        self.assertGreater(bleeding["blr"], -35.0)
        self.assertTrue(sc.track_bleeds(bleeding, self.th))

    def test_clean_pair_is_not_flagged(self):
        for s in "ab":
            self.assertFalse(sc.track_bleeds(self.track("conv_000000000001", s), self.th))
        self.assertFalse(sc.track_bleeds(self.track("conv_000000000002", "b"), self.th))

    def test_decisions(self):
        s0_rows = [{"c": c, "silence_profile_a": "continuous", "silence_profile_b": "continuous"}
                   for c in oc.conversation_ids(self.root)]
        s0_windows = [{"conv": c, "offset_s": "0", "keep": "1"} for c in oc.conversation_ids(self.root)]
        rejected, decided = sc.decide(self.tracks, self.windows, s0_rows, s0_windows, self.th)
        keep = {r["conv"]: r["keep"] for r in decided}
        self.assertEqual(keep, {"conv_000000000001": True, "conv_000000000002": False})
        self.assertEqual(rejected["conv_000000000002"], ["bleed_a"])
        _, decided = sc.decide(self.tracks, self.windows, s0_rows, s0_windows, self.th,
                               extra_reject=["conv_000000000001"])
        self.assertFalse(any(r["keep"] for r in decided))


class CutLayout(unittest.TestCase):
    def layout(self, seed=20261012, tars=4):
        """Six conversations; conv 1-2 share a speaker (one component), every window passes S1 and S2."""
        speakers = {1: ("s1", "s2"), 2: ("s2", "s3"), 3: ("s4", "s5"), 4: ("s6", "s7"), 5: ("s8", "s9"),
                    6: ("s10", "s11")}
        lengths = {1: 3, 2: 2, 3: 12, 4: 7, 5: 1, 6: 9}
        s0_rows, s0_windows, s1_tracks, s1_windows, s2 = [], [], [], [], {}
        for n, (a, b) in speakers.items():
            conv = f"conv_{n:012d}"
            s0_rows.append({"c": conv, "speaker_id_a": a, "speaker_id_b": b, "aec_known": False,
                            "transcript_bleed": False})
            for s in "ab":
                s1_tracks.append({"conv": conv, "s": s, "prominence": 0.0, "blr": -60.0, "exc": 0.0})
            for k in range(lengths[n]):
                if n == 3 and k == 6:
                    continue  # a gap splits the run of consecutive windows
                s0_windows.append({"conv": conv, "offset_s": str(120 * k), "keep": "1"})
                row = {"conv": conv, "offset_s": 120 * k, "zr_a": 0, "zr_b": 0, "clip_a": 0, "clip_b": 0}
                s1_windows.append(row)
                s2[(conv, 120 * k)] = row
        params = {**cw.PARAMS, "seed": seed, "train_tars": tars, "dev_min_windows": 4,
                  "s2": {**cw.PARAMS["s2"], "min_control_windows": 0}}
        th = sc.load_decisions()["thresholds"]
        return cw.plan(s0_rows, s0_windows, s1_tracks, s1_windows, s2, th, False, params), s0_rows

    def test_dev_split_is_speaker_disjoint(self):
        layout, s0_rows = self.layout()
        speakers = {r["c"]: {r["speaker_id_a"], r["speaker_id_b"]} for r in s0_rows}
        dev = {c for c, _ in layout["dev"]}
        train = {c for t in layout["tars"] for c, _, _ in t}
        self.assertEqual(dev, {"conv_000000000005", "conv_000000000001", "conv_000000000002"})
        self.assertFalse(set().union(*(speakers[c] for c in dev)) & set().union(*(speakers[c] for c in train)))
        self.assertEqual(layout["counts"]["kept"], 33)
        self.assertEqual(layout["counts"]["dev_windows"], 6)

    def test_blocks_and_seeded_tars(self):
        self.assertEqual(cw.blocks_of([0, 120, 240, 600, 720, 840, 960, 1080, 1200], 5),
                         [[0, 120, 240], [600, 720, 840, 960, 1080], [1200]])
        first, _ = self.layout()
        again, _ = self.layout()
        other, _ = self.layout(seed=1)
        self.assertEqual(first["tars"], again["tars"])
        self.assertNotEqual(first["tars"], other["tars"])
        sizes = [len(t) for t in first["tars"]]
        self.assertEqual(sum(sizes), 27)
        self.assertLessEqual(max(sizes) - min(sizes), 5)
        for tar in first["tars"]:
            blocks = {}
            for conv, offset, block in tar:
                blocks.setdefault(block, []).append((conv, offset))
            for items in blocks.values():
                self.assertLessEqual(len(items), 5)
                self.assertEqual(len({c for c, _ in items}), 1)
                offsets = [o for _, o in items]
                self.assertEqual(offsets, list(range(offsets[0], offsets[0] + 120 * len(offsets), 120)))

    def test_tar_records_hold_the_stereo_window(self):
        import soundfile as sf
        tmp = Path(tempfile.mkdtemp())
        try:
            rng = np.random.default_rng(0)
            restored = tmp / "restored"
            tracks = {}
            for s in "ab":
                tracks[s] = rng.integers(-20000, 20000, 250 * SR).astype(np.int16)
                (restored / "conv_000000000007").mkdir(parents=True, exist_ok=True)
                sf.write(restored / "conv_000000000007" / f"speaker_{s}.flac", tracks[s], SR, subtype="PCM_16")
            entries = [("conv_000000000007", 0), ("conv_000000000007", 120)]
            self.assertEqual(cw.write_tar(("train", 0, entries, restored, tmp / "source", "digest")), "done")
            self.assertEqual(cw.write_tar(("train", 0, entries, restored, tmp / "source", "digest")), "skipped")
            with tarfile.open(tmp / "source" / "train" / "data-00000.tar") as stream:
                members = {m.name: stream.extractfile(m).read() for m in stream}
            self.assertEqual(sorted(members), ["openyap-000000000007-00000.flac",
                                               "openyap-000000000007-00000.original_path.txt",
                                               "openyap-000000000007-00120.flac",
                                               "openyap-000000000007-00120.original_path.txt"])
            for _, offset in entries:
                stereo, sr = sf.read(io.BytesIO(members[f"openyap-000000000007-{offset:05d}.flac"]), dtype="int16")
                self.assertEqual((sr, stereo.shape), (SR, (120 * SR, 2)))
                for c, s in enumerate("ab"):
                    np.testing.assert_array_equal(stereo[:, c], tracks[s][offset * SR:(offset + 120) * SR])
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
