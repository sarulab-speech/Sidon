"""OpenYap-1k corpus helpers: conversation ids, transcripts, the ASCII word normaliser, echo words and window keys.

The release (TheAgenticDataCompany/open-yap-1k) holds one directory per conversation, conv_<12 hex>, with
speaker_{a,b}.flac (48 kHz mono 16-bit), speaker_{a,b}_transcript.json (word timings), speaker_{a,b}_meta.json,
speaker_{a,b}_dnsmos.json and meta.json. The corpus-independent helpers (run records, atomic writes, FLAC headers,
TSV, masks, the 120 s window grid) are re-exported from common.core. Only the standard library and NumPy; nothing
here decodes audio.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.core import (  # noqa: E402,F401  (re-exported)
    FPS, REPO, SR, WINDOW_S, WINDOW_SAMPLES, _json_default, clean_float, dilate, file_sha256, format_cell,
    full_windows, git_commit, interiors, load_json, params_hash, parse_cell, read_streaminfo, read_tsv, run_info,
    shift, streaminfo_header, word_mask, write_atomic, write_json, write_tsv)

NAME = "openyap-1k"  # name of the restoration input tree (restored/<NAME>/...)
SPEAKERS = ("a", "b")


def other(s: str) -> str:
    return "b" if s == "a" else "a"


# ---- conversations and transcripts ---------------------------------------------------------------------------------

def conversation_ids(root: Path) -> list[str]:
    return sorted(x for x in os.listdir(root) if x.startswith("conv_"))


def conv_hex(conv: str) -> str:
    if not re.fullmatch(r"conv_[0-9a-f]{12}", conv):
        raise ValueError(f"Unexpected conversation id {conv}")
    return conv[5:]


def load_words(root: Path, conv: str, s: str) -> list[dict]:
    """Transcript words of one speaker (entries of type 'word')."""
    transcript = load_json(Path(root) / conv / f"speaker_{s}_transcript.json")
    return [w for w in transcript["words"] if w.get("type", "word") == "word"]


def norm_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


PRESCREEN_ECHO_WINDOW_S = (-0.3, 1.5)


def echo_flags(words: list[dict], partner: list[dict],
               window: tuple[float, float] = PRESCREEN_ECHO_WINDOW_S) -> np.ndarray:
    """Words (normalised length >= 4) that the partner also says at d = own start - partner start inside the open
    interval `window` (s): transcript-level bleed or a shared word. The default window is the metadata screen's
    (-0.3..+1.5 s); the crosstalk screen passes its own (screen_crosstalk.PARAMS)."""
    lo, hi = window
    index: dict[str, list[float]] = {}
    for w in partner:
        index.setdefault(norm_word(w["word"]), []).append(w["start"])
    arrays = {k: np.array(sorted(v)) for k, v in index.items()}
    flags = np.zeros(len(words), bool)
    for i, w in enumerate(words):
        key = norm_word(w["word"])
        if len(key) < 4:
            continue
        times = arrays.get(key)
        if times is None:
            continue
        d = w["start"] - times
        if ((d > lo) & (d < hi)).any():
            flags[i] = True
    return flags


def clean_words(words: dict[str, list[dict]],
                window: tuple[float, float] = PRESCREEN_ECHO_WINDOW_S) -> dict[str, list[dict]]:
    """Each speaker's words without the echo-flagged ones."""
    result = {}
    for s in SPEAKERS:
        flags = echo_flags(words[s], words[other(s)], window)
        result[s] = [w for w, f in zip(words[s], flags) if not f]
    return result


def window_key(conv: str, offset_s: int) -> str:
    """Source record key: openyap-<12 hex>-<offset in s, 5 digits>; no '.' or '_' (the producer appends '_0' and
    readers split names on the last '_')."""
    return f"openyap-{conv_hex(conv)}-{int(offset_s):05d}"


def parse_window_key(key: str) -> tuple[str, int]:
    match = re.fullmatch(r"openyap-([0-9a-f]{12})-([0-9]{5})(?:_0)?", key)
    if match is None:
        raise ValueError(f"Not an OpenYap window key: {key}")
    return f"conv_{match.group(1)}", int(match.group(2))
