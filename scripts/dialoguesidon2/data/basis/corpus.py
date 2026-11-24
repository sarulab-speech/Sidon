"""Basis Conversations 1500 release files, words, masks and keys.

Reads the release's JSON only: `<lang>/segments.jsonl` (tracks by speaker, per-speaker on-call spans and bandwidth,
muted spans, transcripts), `manifests/lhotse/recordings.jsonl` (num_samples, the expected length of every track of a
segment) and `CHECKSUMS.sha256`. Provides the Unicode word normaliser (NFKC, casefold, letters, marks and numbers
only; long words have >= 4 code points, >= 2 in ja and zh, >= 3 in ko), the echo-word-free masks shared by the
screens, the per-ordered-pair transcript-copy statistic, and the window keys. Nothing here decodes audio.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator

import numpy as np

from common import core as cm

NAME = "basis-1500"
REVISION = "05a19e06b5eed44d701a8860b7e2336a44f762dd"  # Hugging Face revision of the release used
SEED = 20261013
ECHO_WINDOW_S = (-0.3, 3.5)  # echo-word window (s), the same as the OpenYap screen's
COPY_RANGES = {"wide": (-3.5, 3.5), "w10": (-10.0, 10.0)}  # transcript-copy scans; S0 takes the union
COPY_RULE = {"excess_gt": 0.01, "ptb_gt": 3, "peak_ge": 10}
COPY_MIN_OVERLAP_S = 60.0
COLOCATED_S = 0.1
KEY_PATTERN = r"basis-[A-Za-z0-9]{20}-[0-9]{5}-[0-9]{5}"
SEGMENT_PATTERN = r"[A-Za-z0-9]{20}_[0-9]{5}"


# ---- release files -------------------------------------------------------------------------------------------------

def languages(root: Path) -> list[str]:
    return sorted(p.name for p in Path(root).iterdir() if (p / "segments.jsonl").exists())


def iter_segment_rows(root: Path, langs: list[str] | None = None) -> Iterator[dict]:
    """Segment rows in language order, then file order (one JSON line at a time)."""
    for lang in langs or languages(root):
        with open(Path(root) / lang / "segments.jsonl") as stream:
            for line in stream:
                yield json.loads(line)


def segment_index(root: Path) -> dict[str, tuple[str, int]]:
    """{segment_id: (language, byte offset of its line)}, so a worker reads one segment row with read_segment."""
    index = {}
    for lang in languages(root):
        offset = 0
        with open(Path(root) / lang / "segments.jsonl", "rb") as stream:
            for line in stream:
                match = re.match(rb'\{"segment_id": "([^"]+)"', line)
                segment_id = match.group(1).decode() if match else json.loads(line)["segment_id"]
                index[segment_id] = (lang, offset)
                offset += len(line)
    return index


def read_segment(root: Path, lang: str, offset: int) -> dict:
    with open(Path(root) / lang / "segments.jsonl", "rb") as stream:
        stream.seek(offset)
        return json.loads(stream.readline())


def lhotse_lengths(root: Path) -> dict[str, int]:
    lengths = {}
    with open(Path(root) / "manifests" / "lhotse" / "recordings.jsonl") as stream:
        for line in stream:
            row = json.loads(line)
            lengths[row["id"]] = int(row["num_samples"])
    return lengths


def checksums(root: Path) -> dict[str, str]:
    """{path relative to the release root: sha256} from CHECKSUMS.sha256 (`<hex>  <path>` lines)."""
    table = {}
    with open(Path(root) / "CHECKSUMS.sha256") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
            if match is None:
                raise ValueError(f"Unexpected CHECKSUMS.sha256 line: {line!r}")
            digest, path = match.groups()
            if table.get(path, digest) != digest:
                raise ValueError(f"CHECKSUMS.sha256 lists {path} with two digests")
            table[path] = digest
    return table


def release_revision(root: Path) -> str | None:
    """The HF revision recorded by the downloader (first line of a .cache/huggingface/download/*.metadata)."""
    cache = Path(root) / ".cache" / "huggingface" / "download"
    for path in sorted(cache.glob("*.metadata")) + sorted(cache.glob("*/segments.jsonl.metadata")):
        lines = path.read_text().splitlines()
        if lines:
            return lines[0].strip()
    return None


# ---- words ---------------------------------------------------------------------------------------------------------

def norm_word(word: str, lang: str | None = None) -> str:
    """NFKC, casefold, keep code points whose Unicode category starts with L, M or N (combining marks stay:
    Devanagari and Arabic need them)."""
    folded = unicodedata.normalize("NFKC", word).casefold()
    return "".join(ch for ch in folded if unicodedata.category(ch)[0] in "LMN")


def min_word_len(lang: str) -> int:
    return {"ja": 2, "zh": 2, "ko": 3}.get(lang, 4)


def segment_words(row: dict) -> dict[str, list[dict]]:
    """Per speaker: {"start", "end", "key"} for every transcript word (key = normalised word); a row without
    words contributes its own start and end with an empty key."""
    lang = row["language"]
    words: dict[str, list[dict]] = collections.defaultdict(list)
    for item in row["transcript"]:
        if item["words"]:
            for w in item["words"]:
                words[item["speaker"]].append({"start": w["start"], "end": w["end"], "key": norm_word(w["word"], lang)})
        else:
            words[item["speaker"]].append({"start": item["start"], "end": item["end"], "key": ""})
    return words


def long_word_times(words: list[dict], minimum: int) -> dict[str, np.ndarray]:
    index = collections.defaultdict(list)
    for w in words:
        if len(w["key"]) >= minimum:
            index[w["key"]].append(w["start"])
    return {k: np.array(sorted(v)) for k, v in index.items()}


def pair_diffs(tx: np.ndarray, ty: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """All tx[i] - ty[j] in the open interval (lo, hi); tx, ty sorted."""
    left = np.searchsorted(ty, tx - hi, "right")
    right = np.searchsorted(ty, tx - lo, "left")
    counts = np.maximum(right - left, 0)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0)
    idx = np.repeat(np.arange(len(tx)), counts)
    offs = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    d = tx[idx] - ty[left[idx] + offs]
    return d[(d > lo) & (d < hi)]


def echo_flags(words: dict[str, list[dict]], speaker: str, minimum: int,
               window: tuple[float, float] = ECHO_WINDOW_S, speakers: list[str] | None = None) -> list[bool]:
    """A long word of `speaker` is an echo word when any other speaker of the segment (`speakers`, default every
    speaker with words) says the same normalised word at t_x - t_y in the open interval `window`."""
    others = collections.defaultdict(list)
    for other in (speakers if speakers is not None else list(words)):
        if other != speaker:
            for key, times in long_word_times(words.get(other, []), minimum).items():
                others[key].append(times)
    pooled = {k: np.sort(np.concatenate(v)) for k, v in others.items()}
    flags = []
    for w in words.get(speaker, []):
        times = pooled.get(w["key"]) if len(w["key"]) >= minimum else None
        flags.append(times is not None and pair_diffs(np.array([w["start"]]), times, *window).size > 0)
    return flags


def clean_words(words: dict[str, list[dict]], speakers: list[str], lang: str,
                window: tuple[float, float] = ECHO_WINDOW_S) -> dict[str, list[dict]]:
    minimum = min_word_len(lang)
    return {s: [w for w, f in zip(words.get(s, []), echo_flags(words, s, minimum, window, speakers)) if not f]
            for s in speakers}


def lag_stat(wx: dict[str, np.ndarray], wy: dict[str, np.ndarray], nx: int, lo: float, hi: float) -> dict | None:
    """S0's transcript-bleed statistic for x against y: lags d = t_x - t_y of identical long words in (lo, hi),
    a 20 ms histogram smoothed over 3 bins; lstar is the smoothed peak's bin centre."""
    if nx == 0:
        return None
    lags = []
    for key, tx in wx.items():
        ty = wy.get(key)
        if ty is None:
            continue
        d = pair_diffs(tx, ty, lo, hi)
        if d.size:
            lags.append(d.round(3))
    bins = np.arange(lo, hi + 1e-9, 0.02)
    lags = np.concatenate(lags) if lags else np.zeros(0)
    h, _ = np.histogram(lags, bins=bins)
    smooth = np.convolve(h, np.ones(3), "same")
    k = int(np.argmax(smooth))
    peak = smooth[k]
    background = (h.sum() - peak) / max(len(h) - 3, 1) * 3
    return {"lstar": round(float((bins[k] + bins[k + 1]) / 2), 2), "peak": int(peak), "bg": float(background),
            "n": nx, "excess": float((peak - background) / nx), "ptb": float(peak / (background + 1))}


def copy_flagged(stat: dict | None) -> bool:
    return stat is not None and stat["excess"] > COPY_RULE["excess_gt"] and stat["ptb"] > COPY_RULE["ptb_gt"] \
        and stat["peak"] >= COPY_RULE["peak_ge"]


def copy_holders(x: str, y: str, lstar: float) -> list[str]:
    """The track(s) holding the copies for the statistic of x against y: tracks share one clock, so copies sit
    later than the original. lstar >= +0.1 s: in x; <= -0.1 s: in y; otherwise both (co-located)."""
    if lstar >= COLOCATED_S:
        return [x]
    if lstar <= -COLOCATED_S:
        return [y]
    return [x, y]


def transcript_copies(words: dict[str, list[dict]], speakers: list[str], on_call: dict[str, np.ndarray],
                      lang: str, ranges: dict = COPY_RANGES) -> tuple[dict[str, set], list[dict]]:
    """Every ordered pair with >= 60 s of common on-call time, over each range: the holders flagged per range
    and one row per flagged ordered pair."""
    minimum = min_word_len(lang)
    longw = {s: long_word_times(words.get(s, []), minimum) for s in speakers}
    nlong = {s: sum(len(v) for v in longw[s].values()) for s in speakers}
    holders = {name: set() for name in ranges}
    rows = []
    for i, x in enumerate(speakers):
        for y in speakers[i + 1:]:
            overlap = (on_call[x] & on_call[y]).sum() / cm.FPS
            if overlap < COPY_MIN_OVERLAP_S:
                continue
            for name, (lo, hi) in ranges.items():
                for a, b in ((x, y), (y, x)):
                    stat = lag_stat(longw[a], longw[b], nlong[a], lo, hi)
                    if copy_flagged(stat):
                        hit = copy_holders(a, b, stat["lstar"])
                        holders[name].update(hit)
                        rows.append({"range": name, "x": a, "y": b, **stat, "copies_in": sorted(hit)})
    return holders, rows


# ---- masks ---------------------------------------------------------------------------------------------------------

def segment_masks(row: dict, frames: int, pad: float = 0.1, gap: float = 0.3,
                  window: tuple[float, float] = ECHO_WINDOW_S) -> dict:
    """O (echo-word-free word masks), C (on call), M (muted), A = C & ~M and word interiors of every speaker
    of a segment row at 100 fps over `frames` frames."""
    speakers = [s["speaker"] for s in row["speakers"]]
    words = segment_words(row)
    clean = clean_words(words, speakers, row["language"], window)
    duration = frames / cm.FPS + 1.0
    o = {s: cm.word_mask(clean[s], duration, pad=pad, gap=gap, frames=frames) for s in speakers}
    c = {s["speaker"]: cm.span_mask([(x["start"], x["end"]) for x in s["on_call"]], frames) for s in row["speakers"]}
    m = cm.span_mask([(x["start"], x["end"]) for x in row["muted_spans"]], frames)
    return {"speakers": speakers, "words": words, "clean": clean, "o": o, "c": c, "m": m,
            "a": {s: c[s] & ~m for s in speakers},
            "spans": {s: cm.interiors(clean[s]) for s in speakers}}


# ---- keys and paths ------------------------------------------------------------------------------------------------

def segment_start(segment_id: str) -> int:
    if not re.fullmatch(SEGMENT_PATTERN, segment_id):
        raise ValueError(f"Unexpected segment id {segment_id}")
    return int(segment_id.rsplit("_", 1)[1])


def window_key(segment_id: str, offset_s: int) -> str:
    """Source record key basis-<conversation>-<segment start, 5 digits>-<offset s, 5 digits>: no '.' or '_'
    (the producer appends '_0' and readers split names on the last '_')."""
    conversation, start = segment_id.rsplit("_", 1)
    segment_start(segment_id)
    return f"basis-{conversation}-{int(start):05d}-{int(offset_s):05d}"


def parse_window_key(key: str) -> tuple[str, int]:
    match = re.fullmatch(r"basis-([A-Za-z0-9]{20})-([0-9]{5})-([0-9]{5})(?:_0)?", key)
    if match is None:
        raise ValueError(f"Not a Basis window key: {key}")
    return f"{match.group(1)}_{match.group(2)}", int(match.group(3))


def order_hash(*parts) -> str:
    """sha256 hex of "<SEED>|<part>|..." (the S0 tie-break and channel order)."""
    return hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()
