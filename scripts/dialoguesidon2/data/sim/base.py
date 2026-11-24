"""Shared code of the simulated-conversation pipeline: parameters, the record geometry, keys and styles, the energy
VAD and 100 fps activity masks, and the turn segmentation shared by the turn-taking fit (fit_turns.py), the planner
(simulate.py) and the checks of the rendered tracks (qc.py s1), so that all three measure turns the same way.

Masks. Every activity mask is built by common.core.word_mask from spans in seconds (pad 0.1 s, gaps below 0.3 s
filled), the same construction as the word masks of the OpenYap-1k crosstalk screen. On simulated tracks the spans
are the runs of an energy VAD (common.shards_qc.frame_energy at 100 fps above PARAMS["vad"]["dbfs"]), merged across
raw gaps below 0.5 s, which word_mask would merge anyway (0.5 s - 2 x 0.1 s pad < 0.3 s).

Turns (segment): spurts are the runs of one speaker's mask. A spurt of at most 1.5 s lying strictly inside a spurt
of the other speaker is an interjection (counted, then removed). The remaining spurts, in start order, form turns:
maximal runs of one speaker. FTO (floor-transfer offset) = next turn start - this turn end (negative = overlap);
within-turn pauses are the gaps between a turn's spurts. The first and last turn of a window are censored (their
duration is not counted).

Only the standard library, NumPy and soundfile; nothing here reads data by itself.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import core as cm  # noqa: E402
from common import shards_qc as sq  # noqa: E402

SUBSETS = ("train-clean-100", "train-clean-360", "train-other-500")
NAME = "sim-ltr"  # the key prefix of the simulated records
WRITER = "sim_24k_120s"  # the producer's writer name for this corpus
SEED = 20261015
SR = 24_000
TRACK_S = 120
TRACK_SAMPLES = TRACK_S * SR  # 2,880,000
OUT_SR = cm.SR  # 48 kHz: the clean tracks upsampled 2x (the simulated tracks are not restored)
OUT_FRAMES = cm.WINDOW_SAMPLES  # 5,760,000
FPS = cm.FPS
HOP = SR // FPS  # 240 samples per 10 ms frame at 24 kHz
FRAMES = TRACK_S * FPS  # 12,000
KEY_RE = re.compile(r"^sim-ltr-[0-9a-f]{12}-[0-9]{5}$")
# A source key without anchors (the producer appends "_0" to it).
PRODUCER_KEY_PATTERN = r"sim-ltr-[0-9a-f]{12}-[0-9]{5}"
STYLES = ("overlap", "conversation")  # by the parity of n

# Every parameter of the simulation; recorded in every run record.
PARAMS = {
    "version": 2,  # 2: utterances may be reused (version 1 used each utterance once)
    "seed": SEED,
    "subsets": list(SUBSETS),
    "level": {"target_dbfs": -26.0, "jitter_db": 3.0, "peak_max": 0.99},
    "trim": {"rel_peak_db": -40.0, "floor_dbfs": -70.0, "keep_s": 0.05},
    "vad": {"dbfs": -50.0, "rel_level_db": -24.0, "pad_s": 0.1, "gap_s": 0.3, "raw_merge_s": 0.5},
    "record": {"seconds": TRACK_S, "min_activity": 0.05, "quarter_s": 30},
    "overlap": {"pause_s": [0.2, 1.0], "b_start_s": [0.0, 1.0]},
    "conversation": {"interjection_max_s": 1.5, "inside_margin_s": 0.05, "guard_s": 0.35, "fto_margin_s": 0.05,
                     "min_turn_s": 0.3},
    # Reuse cap: each utterance at most max_uses times in all, at most ceil(max_uses / 2) times per style, never
    # twice in one record and never twice with the same partner; max_uses 1 is single use. simulate.py plan
    # --max-uses sets it and plan.json records it. The released data used 4.
    "reuse": {"max_uses": 4},
    # 128 tars x 96 records per use of the pool (12,288 at single use); a short pool yields what it has.
    "split": {"train_tars": 128, "train_records": 49_152, "dev_records": 72,
              "dev_subsets": ["train-clean-100", "train-clean-360"], "dev_reserve_factor": 2.0,
              "speech_per_record_s": {"overlap": 216.0, "conversation": 96.0}},
    "pairing": {"min_remaining_s": {"overlap": 110.0, "conversation": 30.0}, "attempts": 64},
    # report_only: conversation-style measures that the S1 check reports but does not require. Turns are whole
    # LibriTTS-R utterances, so turns come out about twice as long as in the fit, and the overlap share with them.
    "s1": {"tolerance": 0.2, "frame_floor_s": 0.01, "overlap_activity_min": 0.85, "overlap_share_min": 0.7,
           "top10_share_max": 0.073, "report_only": ["turn_s", "overlap_share"]},
}


# ---- keys, styles, layout -------------------------------------------------------------------------------------

def record_key(split: str, n: int, speaker_a: str, speaker_b: str, seed: int = SEED) -> str:
    """sim-ltr-<first 12 hex of sha1("<split>|<n>|<speaker A>|<speaker B>|<seed>")>-<n, 5 digits>."""
    if not 0 <= n < 100_000:
        raise ValueError(f"record number {n} does not fit 5 digits")
    digest = hashlib.sha1(f"{split}|{n}|{speaker_a}|{speaker_b}|{seed}".encode()).hexdigest()[:12]
    return f"sim-ltr-{digest}-{n:05d}"


def parse_key(key: str) -> int:
    """n of a source key, or of the producer's `<key>_0`."""
    match = re.fullmatch(r"sim-ltr-[0-9a-f]{12}-([0-9]{5})(?:_0)?", key)
    if match is None:
        raise ValueError(f"Not a simulated record key: {key}")
    return int(match.group(1))


def style_of(n: int) -> str:
    return STYLES[n % 2]


def tar_of(n: int, tars: int) -> int:
    """Records are dealt to tars in consecutive pairs (2k, 2k + 1), so every tar holds as many overlap-style as
    conversation-style records (+-1). Dealing single records round-robin to an even number of tars would put
    one style only in each tar."""
    return (n // 2) % tars


def track_relpath(split: str, tar: int, key: str, channel: str) -> str:
    """Path of a clean track below clean/ (channel a = ch0, b = ch1)."""
    return f"{split}/t{tar:03d}/{key}_{channel}.flac"


# ---- signal -----------------------------------------------------------------------------------------------------

def frame_db(x: np.ndarray) -> np.ndarray:
    """dBFS per 10 ms frame (common.shards_qc.frame_energy) of int16 or float samples at 24 kHz."""
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    if len(x) < 2 * HOP:
        return np.full(len(x) // HOP, -110.0)
    return sq.frame_energy(x, SR)


def raw_runs(active: np.ndarray, merge_frames: int) -> np.ndarray:
    """Runs [start, end) of True frames, merged across gaps shorter than merge_frames, as an (n, 2) array."""
    runs = runs_of(active)
    if len(runs) < 2:
        return runs
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] < merge_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return np.array(merged, np.int64).reshape(-1, 2)


def runs_of(mask: np.ndarray) -> np.ndarray:
    padded = np.zeros(len(mask) + 2, np.int8)
    padded[1:-1] = np.asarray(mask, bool)
    edges = np.diff(padded)
    return np.stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)], 1).astype(np.int64)


def format_runs(runs: np.ndarray) -> str:
    return ",".join(f"{int(a)}:{int(b)}" for a, b in runs)


def parse_runs(text: str) -> np.ndarray:
    if not text or text == "NA":
        return np.zeros((0, 2), np.int64)
    return np.array([[int(v) for v in item.split(":")] for item in text.split(",")], np.int64).reshape(-1, 2)


def mask_from_spans(spans, frames: int = FRAMES, pad: float | None = None, gap: float | None = None) -> np.ndarray:
    """100 fps mask of [start, end) spans in seconds through word_mask (pad, then gaps below `gap` filled)."""
    pad = PARAMS["vad"]["pad_s"] if pad is None else pad
    gap = PARAMS["vad"]["gap_s"] if gap is None else gap
    words = [{"start": float(a), "end": float(b)} for a, b in spans if b > a]
    return cm.word_mask(words, frames / FPS + 1.0, pad=pad, gap=gap, frames=frames)


def vad_spans(x: np.ndarray, threshold_dbfs: float | None = None) -> list[tuple[float, float]]:
    """Raw active spans (s) of a track: frames above the threshold, merged across gaps below raw_merge_s."""
    threshold = PARAMS["vad"]["dbfs"] if threshold_dbfs is None else threshold_dbfs
    runs = raw_runs(frame_db(x) > threshold, int(round(PARAMS["vad"]["raw_merge_s"] * FPS)))
    return [(a / FPS, b / FPS) for a, b in runs]


def vad_mask(x: np.ndarray) -> np.ndarray:
    """The activity mask of a clean 120 s track (24 kHz) used by the S0 and S1 checks."""
    return mask_from_spans(vad_spans(x), frames=len(x) // HOP)


def activity(a: np.ndarray, b: np.ndarray) -> dict:
    union = a | b
    both = a & b
    u = float(union.mean()) if len(union) else 0.0
    return {"activity_a": float(a.mean()), "activity_b": float(b.mean()), "activity_union": u,
            "overlap_share": float(both.sum() / union.sum()) if union.any() else None}


def quarters_active(union: np.ndarray, quarter_frames: int = PARAMS["record"]["quarter_s"] * FPS) -> list[bool]:
    return [bool(union[i:i + quarter_frames].any()) for i in range(0, len(union), quarter_frames)]


# ---- turns ------------------------------------------------------------------------------------------------------

def segment(a: np.ndarray, b: np.ndarray, max_interjection_s: float | None = None) -> dict:
    """Turn segmentation of two 100 fps masks (module docstring). Durations in seconds."""
    limit = int(round((PARAMS["conversation"]["interjection_max_s"] if max_interjection_s is None
                       else max_interjection_s) * FPS))
    frames = len(a)
    by = {0: runs_of(a), 1: runs_of(b)}
    kept, interjections = [], []
    for c in (0, 1):
        other = by[1 - c]
        for start, end in by[c]:
            inside = other.size and bool(np.any((other[:, 0] < start) & (other[:, 1] > end)))
            if end - start <= limit and inside:
                interjections.append((int(start), int(end), c))
            else:
                kept.append((int(start), int(end), c))
    kept.sort(key=lambda r: (r[0], r[1], r[2]))
    turns: list[list] = []
    for start, end, c in kept:
        if turns and turns[-1][0] == c:
            turns[-1][2] = max(turns[-1][2], end)
            turns[-1][3].append((start, end))
        else:
            turns.append([c, start, end, [(start, end)]])
    fto = [(turns[i + 1][1] - turns[i][2]) / FPS for i in range(len(turns) - 1)]
    pauses = [(t[3][j + 1][0] - t[3][j][1]) / FPS for t in turns for j in range(len(t[3]) - 1)]
    durations = [(t[2] - t[1]) / FPS for t in turns[1:-1]]
    return {"frames": frames, "turns": len(turns), "turn_s": durations, "fto_s": fto, "pause_s": pauses,
            "interjection_s": [(e - s) / FPS for s, e, _ in interjections],
            "turn_seconds": float(sum(t[2] - t[1] for t in turns) / FPS),
            "interjections": len(interjections)}


# ---- tables -----------------------------------------------------------------------------------------------------

QUANTILE_POINTS = 1001


def table(values) -> dict:
    """Empirical quantile table (1,001 points) with the sample size and p10/p50/p90."""
    values = np.asarray([v for v in values if v is not None], np.float64)
    if not len(values):
        return {"n": 0, "q": [], "p10": None, "p50": None, "p90": None}
    q = np.quantile(values, np.linspace(0, 1, QUANTILE_POINTS))
    return {"n": int(len(values)), "q": [round(float(v), 6) for v in q], **pct(values)}


def pct(values) -> dict:
    values = np.asarray([v for v in values if v is not None], np.float64)
    if not len(values):
        return {"p10": None, "p50": None, "p90": None}
    return {f"p{p}": float(np.percentile(values, p)) for p in (10, 50, 90)}


def draw(tab: dict, rng: np.random.Generator) -> float:
    """One draw from an empirical quantile table (inverse CDF with linear interpolation)."""
    q = tab["q"]
    if not q:
        raise ValueError("empty quantile table")
    return float(np.interp(rng.random(), np.linspace(0, 1, len(q)), q))


def within(sim: float | None, ref: float | None, tolerance: float, floor: float = 0.0) -> bool:
    """|sim - ref| <= tolerance * |ref| (+ floor: one 10 ms frame, the masks' resolution)."""
    if sim is None or ref is None:
        return False
    return abs(sim - ref) <= tolerance * abs(ref) + floor


# ---- clean tracks -----------------------------------------------------------------------------------------------

def upsample_48k(x: np.ndarray) -> tuple[np.ndarray, float]:
    """A clean 24 kHz int16 track as 48 kHz int16 (resample_poly(2, 1) in float64), and the scale applied: 1.0
    unless the interpolation overshoots int16, when the whole track is scaled to a peak of 32767. The producer
    peak-normalises each channel, so the scale changes nothing downstream."""
    from scipy.signal import resample_poly
    if x.dtype != np.int16 or x.ndim != 1:
        raise ValueError("expected a mono int16 track")
    y = resample_poly(x.astype(np.float64), 2, 1)
    peak = float(np.abs(y).max()) if len(y) else 0.0
    scale = 32767.0 / peak if peak > 32767.0 else 1.0
    return np.round(y * scale).astype(np.int16), scale


def clean_problem(path: Path) -> str | None:
    """None when `path` is a mono 24 kHz PCM_16 FLAC of exactly 2,880,000 samples (the S0 format check)."""
    import soundfile as sf
    if not Path(path).exists():
        return "missing"
    try:
        info = sf.info(str(path))
    except Exception as error:
        return f"unreadable: {type(error).__name__}: {error}"
    if (info.samplerate, info.channels, info.subtype, info.frames) != (SR, 1, "PCM_16", TRACK_SAMPLES):
        return f"format {info.samplerate} Hz {info.channels} ch {info.subtype} {info.frames} frames"
    return None
