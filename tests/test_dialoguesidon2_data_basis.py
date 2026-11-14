"""Basis pipeline rules on synthetic segment rows (no Basis data, no audio).

Covers the metadata screen (track exclusions, eligibility, pairing and channel order, window rules) and the
decisions of the raw-audio screen under the shipped thresholds.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "scripts" / "dialoguesidon2" / "data"
sys.path.insert(0, str(DATA))

from basis import corpus as bs  # noqa: E402
from basis import plan_pairs as pp  # noqa: E402
from basis import screen as sn  # noqa: E402

SR = 48_000


def conv_id(n: int) -> str:
    return f"TESTCONV{n:012d}"


def words_for(speakers, seconds, seed, start=0.5):
    """Turn-taking word lists with unique tokens per speaker (no echo words by accident)."""
    rng = np.random.default_rng(seed)
    words = {s: [] for s in speakers}
    t = start
    while t < seconds - 3:
        s = speakers[int(rng.integers(len(speakers)))]
        for _ in range(int(rng.integers(4, 14))):
            d = float(rng.uniform(0.15, 0.45))
            if t + d > seconds - 1:
                break
            words[s].append({"word": f"{s.lower()}w{len(words[s]):05d}", "start": round(t, 3),
                             "end": round(t + d, 3)})
            t += d + float(rng.uniform(0.02, 0.15))
        t += float(rng.uniform(0.2, 0.8))
    return words


def segment_row(n, seconds, words, on_call=None, muted=(), bandwidth=None, lang="en"):
    conv = conv_id(n)
    row = {"segment_id": f"{conv}_00000", "conversation_id": conv, "language": lang, "duration": seconds,
           "tracks": {s: f"{lang}/audio/{conv}/{conv}.{s}.flac" for s in words},
           "muted_spans": [{"start": a, "end": b, "reason": "test"} for a, b in muted],
           "speakers": [], "transcript": []}
    for s in words:
        spec = {"speaker": s, "on_call": [{"start": a, "end": b}
                                          for a, b in (on_call or {}).get(s, [(0.0, seconds)])]}
        bw = (bandwidth or {}).get(s, 12000)
        if bw is not None:
            spec["audio"] = {"bandwidth_hz": bw}
        row["speakers"].append(spec)
        for w in words[s]:
            row["transcript"].append({"speaker": s, "start": w["start"], "end": w["end"], "words": [w]})
    row["transcript"].sort(key=lambda r: (r["start"], r["speaker"]))
    return row


class MetadataScreenTests(unittest.TestCase):
    def plan(self, row):
        return pp.plan_segment(row, int(row["duration"] * SR))

    def test_pair_and_channel_order(self):
        row = segment_row(1, 250.0, words_for(["SPKA", "SPKB", "SPKC"], 250.0, 1))
        segment, windows = self.plan(row)
        self.assertEqual(len(windows), 2)  # full 120 s windows only
        for w in windows:
            self.assertTrue(w["keep"], w["reason"])
            self.assertEqual(len(w["eligible"]), 3)
            pair = set(w["eligible"][:2])
            self.assertEqual({w["spk0"], w["spk1"]}, pair)
            order = sorted(pair, key=lambda s: bs.order_hash("order", row["segment_id"], w["offset_s"], s))
            self.assertEqual([w["spk0"], w["spk1"]], order)
            self.assertEqual(w["n_other_on_call"], 1)
        self.assertEqual(segment["controls"], [])

    def test_bandwidth_and_on_call_exclusions(self):
        words = words_for(["SPKA", "SPKB", "SPKC"], 250.0, 2)
        row = segment_row(2, 250.0, words, bandwidth={"SPKB": 2500}, on_call={"SPKC": [(0.0, 100.0)]})
        segment, windows = self.plan(row)
        self.assertEqual(segment["excluded"], {"SPKB": ["T1"]})
        for w in windows:
            self.assertNotIn("SPKB", w["eligible"])
            self.assertNotIn("SPKC", w["eligible"])  # on call for less than 119.5 s of the window
            self.assertFalse(w["keep"])
            self.assertEqual(w["reason"], "eligible<2")

    def test_muted_window_is_dropped(self):
        row = segment_row(3, 250.0, words_for(["SPKA", "SPKB"], 250.0, 3), muted=[(150.0, 160.0)])
        _, windows = self.plan(row)
        self.assertEqual([w["keep"] for w in windows], [True, False])
        self.assertEqual(windows[1]["reason"], "muted")

    def test_transcript_copies_make_a_control(self):
        words = words_for(["SPKA", "SPKB"], 250.0, 4)
        copies = [{"word": w["word"], "start": round(w["start"] + 0.5, 3), "end": round(w["end"] + 0.5, 3)}
                  for w in words["SPKB"] if w["end"] + 0.5 < 249]
        words["SPKA"] = sorted(words["SPKA"] + copies, key=lambda w: w["start"])
        segment, windows = self.plan(segment_row(4, 250.0, words))
        self.assertEqual(segment["excluded"], {"SPKA": ["T0"]})  # copies sit later, in A's track
        self.assertEqual(segment["controls"], ["SPKA"])
        self.assertTrue(all(not w["keep"] for w in windows))
        self.assertTrue(segment["control_windows"])
        self.assertTrue(all(c["x"] == "SPKA" and c["partner"] == "SPKB" for c in segment["control_windows"]))

    def test_window_keys(self):
        sid = f"{conv_id(5)}_00360"
        key = bs.window_key(sid, 240)
        self.assertEqual(key, f"basis-{conv_id(5)}-00360-00240")
        self.assertEqual(bs.parse_window_key(key + "_0"), (sid, 240))


def thresholds() -> dict:
    decisions = json.loads((DATA / "basis" / "thresholds.json").read_text())
    return {**sn.DEFAULT_THRESHOLDS, **decisions["thresholds"]}


class DecisionTests(unittest.TestCase):
    """sn.decide on hand-made screen rows of one segment with tracks a, b (the pair) and c (left out)."""

    sid = f"{conv_id(9)}_00000"

    def segments(self):
        return [{"segment_id": self.sid, "speakers": ["a", "b", "c"], "controls": [], "control_windows": []}]

    def s0_windows(self):
        return [{"segment_id": self.sid, "offset_s": 0, "spk0": "a", "spk1": "b", "keep": True,
                 "n_other_on_call": 1}]

    def screen_rows(self, pair_ab=None, window_ab=None, track_a=None):
        pair = {"seg": self.sid, "x": "a", "y": "b", "is_pair": True, "lag": 20, "prominence": 0.02,
                "blr": -60.0, "exc": 0.0, "blocks": [{"lag": 20, "prominence": 0.02}]}
        pair.update(pair_ab or {})
        quiet = {"seg": self.sid, "is_pair": True, "lag": 0, "prominence": 0.01, "blr": -70.0, "exc": 0.0,
                 "blocks": []}
        pairs = [pair, {**quiet, "x": "a", "y": "c"}, {**quiet, "x": "b", "y": "a"}, {**quiet, "x": "b", "y": "c"}]
        base = {"seg": self.sid, "offset_s": 0, "x": "a", "y": "b", "lag_conv": 20, "blr_conv": -60.0,
                "exc_conv": 0.0, "lag_blk": 20, "blr_blk": -60.0, "exc_blk": 0.0, "zr": 0, "clip": 0}
        row = {**base, **(window_ab or {})}
        windows = [row, {**row, "y": "*", "lag_conv": None, "lag_blk": None, "blr_conv": -60.0, "exc_conv": 0.0},
                   {**base, "x": "b", "y": "a"}]
        track = {"seg": self.sid, "x": "a", "gate_runs_per_hour": 0.0, "floor": -80.0}
        track.update(track_a or {})
        tracks = [track, {"seg": self.sid, "x": "b", "gate_runs_per_hour": 0.0, "floor": -80.0}]
        return {"tracks": tracks, "pairs": pairs, "windows": windows}

    def decide(self, clip_share=0.0, **rows):
        clip = {(self.sid, x, 0): {"clip_share": clip_share} for x in ("a", "b")}
        excluded, decided = sn.decide(self.segments(), self.s0_windows(), self.screen_rows(**rows),
                                      {"tracks": {}, "windows": {}}, thresholds(), False, clip)
        return excluded, decided[0]

    def test_clean_window_is_kept(self):
        _, row = self.decide()
        self.assertTrue(row["keep"], row)

    def test_track_rule_inside_and_outside_the_lag_window(self):
        _, row = self.decide(pair_ab={"prominence": 0.2, "blr": -20.0, "lag": 50})  # +0.5 s: inside
        self.assertFalse(row["keep"])
        self.assertEqual(row["track_reject"], "a:bleed:b")
        _, row = self.decide(pair_ab={"prominence": 0.2, "blr": -20.0, "lag": -300})  # -3 s: outside
        self.assertTrue(row["keep"], row)

    def test_colocated_peak_flags_without_levels(self):
        _, row = self.decide(pair_ab={"prominence": 0.35, "lag": 5, "blr": None, "exc": None})
        self.assertEqual(row["track_reject"], "a:bleed:b")

    def test_window_rule_needs_a_prominent_lag(self):
        loud = {"blr_conv": -20.0, "exc_conv": 10.0, "blr_blk": -20.0, "exc_blk": 10.0}
        _, row = self.decide(pair_ab={"prominence": 0.08, "lag": 20}, window_ab=loud)
        self.assertEqual(row["window_bleed"], "a:b")
        _, row = self.decide(pair_ab={"prominence": 0.03, "lag": 20,
                                      "blocks": [{"lag": 20, "prominence": 0.03}]}, window_ab=loud)
        self.assertEqual(row["window_bleed"], "")  # the lag gate is closed: chance peak

    def test_integrity_rules(self):
        _, row = self.decide(clip_share=0.01)
        self.assertEqual(row["integrity"], "a:clip_share,b:clip_share")
        _, row = self.decide(window_ab={"zr": 2})
        self.assertEqual(row["integrity"], "a:zero_runs")
        _, row = self.decide(window_ab={"zr": 2}, track_a={"gate_runs_per_hour": 500.0})
        self.assertTrue(row["keep"], row)  # a noise-gated track: zero runs are gate closures

    def test_noise_floor_rule(self):
        excluded, row = self.decide(track_a={"floor": -50.0})
        self.assertEqual(excluded[(self.sid, "a")], ["floor"])
        self.assertFalse(row["keep"])


class ThresholdFileTests(unittest.TestCase):
    def test_shipped_values(self):
        decisions = json.loads((DATA / "basis" / "thresholds.json").read_text())
        self.assertEqual(set(decisions), {"thresholds", "fallback_no_excluded_on_call", "s2_controls"})
        th = decisions["thresholds"]
        self.assertIsNone(th["dnsmos_track_min"])
        self.assertIsNone(th["dnsmos_window_min"])
        self.assertEqual(th["track_lag_s"], [-0.1, 2.0])
        self.assertEqual(set(sn.DEFAULT_THRESHOLDS) - set(th), set())


if __name__ == "__main__":
    unittest.main()
