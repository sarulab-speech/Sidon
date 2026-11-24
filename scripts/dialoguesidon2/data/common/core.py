"""Corpus-independent helpers: the 48 kHz / 100 fps / 120 s grid, run records, atomic writes, FLAC
headers, TSV I/O, 100 fps word masks and the window grid.

Only the standard library and NumPy; nothing here decodes audio.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[4]
SR = 48_000
FPS = 100  # frames per second of every mask and energy track
HOP = SR // FPS  # 480 samples per frame
WINDOW_S = 120
WINDOW_SAMPLES = WINDOW_S * SR  # 5,760,000
WINDOW_FRAMES = WINDOW_S * FPS  # 12,000


# ---- run records ------------------------------------------------------------------------------------------------

def git_commit(repo: Path = REPO) -> dict:
    """HEAD of the checkout this code runs from, and whether tracked files differ from it."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        head, dirty = os.environ.get("DATA_CODE_COMMIT", "unknown"), None
    return {"commit": head, "dirty": dirty}


def run_info(stage: str, params: dict, **extra) -> dict:
    """The record written next to every output: stage, code commit, parameters, host, argv and time."""
    return {"stage": stage, **git_commit(), "params": params, "host": socket.gethostname(), "argv": sys.argv,
            "python": sys.executable, "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            **extra}


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def write_atomic(path: Path, data: bytes | str) -> None:
    """Write via a temporary file in the same directory and rename, so a reader never sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, mode) as stream:
            stream.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: Path, obj: Any) -> None:
    write_atomic(path, json.dumps(obj, indent=1, sort_keys=False, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def clean_float(value):
    """JSON-safe float: NaN and infinities become None."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def sanitize(obj):
    """NaN and infinities to None, NumPy scalars to Python numbers, tuples to lists (JSON-ready rows)."""
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def nan_to_none(value):
    """json.dumps default for rows that may still hold NumPy scalars."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    raise TypeError(type(value))


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    with open(path) as stream:
        return json.load(stream)


def npz_bytes(arrays: dict) -> bytes:
    import io
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


# ---- FLAC headers --------------------------------------------------------------------------------------------------

def read_streaminfo(path: Path) -> dict:
    """The FLAC stream-info block (the first 42 bytes): sample rate, channels, bits, total samples, MD5."""
    with open(path, "rb") as stream:
        head = stream.read(42)
    if head[:4] != b"fLaC":
        raise ValueError(f"Not a FLAC file: {path}")
    block_type, length = head[4] & 0x7F, int.from_bytes(head[5:8], "big")
    if block_type != 0 or length != 34:
        raise ValueError(f"First metadata block of {path} is not the stream-info block")
    info = head[8:42]
    packed = int.from_bytes(info[10:18], "big")
    return {"sr": packed >> 44, "ch": ((packed >> 41) & 7) + 1, "bps": ((packed >> 36) & 31) + 1,
            "ns": packed & ((1 << 36) - 1), "md5": info[18:].hex()}


def streaminfo_header(sr: int, channels: int, bits: int, samples: int) -> bytes:
    """A minimal FLAC header (magic + last stream-info block), for tests of read_streaminfo."""
    packed = (sr << 44) | ((channels - 1) << 41) | ((bits - 1) << 36) | samples
    info = struct.pack(">HH", 4096, 4096) + b"\0" * 6 + packed.to_bytes(8, "big") + b"\0" * 16
    return b"fLaC" + bytes([0x80]) + (34).to_bytes(3, "big") + info


# ---- masks -------------------------------------------------------------------------------------------------------

def word_mask(words: Iterable[dict], duration: float, pad: float = 0.1, gap: float = 0.3,
              frames: int | None = None) -> np.ndarray:
    """Frames (100 fps) covered by words padded by `pad` s, merging gaps shorter than `gap` s.
    `frames` sets the length (default int(duration * 100))."""
    m = np.zeros(int(duration * FPS) + 2, bool)
    intervals = sorted((w["start"], w["end"]) for w in words)
    merged: list[list[float]] = []
    for start, end in intervals:
        start -= pad
        end += pad
        if merged and start - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for start, end in merged:
        m[max(0, int(start * FPS)):max(0, int(end * FPS) + 1)] = True
    m = m[:int(duration * FPS)]
    if frames is not None:
        m = m[:frames] if len(m) >= frames else np.pad(m, (0, frames - len(m)))
    return m


def span_mask(spans: Iterable[tuple[float, float]], frames: int) -> np.ndarray:
    """Frames covered by [start, end) spans in seconds, with frame edges at round(t * 100)."""
    m = np.zeros(frames, bool)
    for start, end in spans:
        m[max(0, int(round(start * FPS))):max(0, int(round(end * FPS)))] = True
    return m


def dilate(mask: np.ndarray, frames: int) -> np.ndarray:
    """Binary dilation by `frames` on both sides."""
    if frames <= 0 or not mask.any():
        return mask.copy()
    n = len(mask)
    counts = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])  # counts[i] = mask[:i].sum()
    idx = np.arange(n)
    return (counts[np.minimum(idx + frames + 1, n)] - counts[np.maximum(idx - frames, 0)]) > 0


def shift(values: np.ndarray, lag: int, fill=False) -> np.ndarray:
    """out[t] = values[t - lag] (the partner `lag` frames earlier), `fill` outside the range."""
    out = np.full(len(values), fill, dtype=values.dtype)
    if lag >= 0:
        if lag < len(values):
            out[lag:] = values[:len(values) - lag]
    elif -lag < len(values):
        out[:lag] = values[-lag:]
    return out


def interiors(words: Iterable[dict], margin: float = 0.03) -> np.ndarray:
    """Word interiors [start + margin, end - margin] in seconds, sorted, as an (n, 2) array."""
    rows = [(w["start"] + margin, w["end"] - margin) for w in words if w["end"] - w["start"] > 2 * margin]
    return np.array(sorted(rows), float).reshape(-1, 2)


def full_windows(samples: int) -> int:
    """Full 120 s windows on the grid t = 0, 120, ... (no padded tails), from the FLAC header length."""
    return int((samples / SR) // WINDOW_S)


def run_frames(runs: np.ndarray, frames: int) -> np.ndarray:
    """Frames (480-sample hops) lying entirely inside one of the sample runs (start, length)."""
    delta = np.zeros(frames + 1, np.int64)
    for start, length in np.asarray(runs, np.int64).reshape(-1, 2):
        first = -(-int(start) // HOP)
        last = min((int(start) + int(length)) // HOP, frames)
        if last > first:
            delta[first] += 1
            delta[last] -= 1
    return np.cumsum(delta[:frames]) > 0


# ---- TSV ---------------------------------------------------------------------------------------------------------

def read_tsv(path: Path) -> list[dict]:
    with open(path) as stream:
        header = stream.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in stream if line.strip()]


def write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(format_cell(row.get(c)) for c in columns))
    write_atomic(path, "\n".join(lines) + "\n")


def format_cell(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        return "NA" if not np.isfinite(value) else repr(float(value))
    return str(value)


def parse_cell(value: str):
    if value == "NA":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_parts(text: str, count: int) -> list[int]:
    """'0-7', '8-15', '3', '0,2,5-6' -> sorted part indices, each in [0, count)."""
    parts: set[int] = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = (int(x) for x in item.split("-", 1))
            parts.update(range(lo, hi + 1))
        else:
            parts.add(int(item))
    if not parts or min(parts) < 0 or max(parts) >= count:
        raise ValueError(f"Parts {text!r} must lie in 0..{count - 1}")
    return sorted(parts)
