"""Restoration inputs and output checks (restoration itself: common/sidon_restore.py).

sidon_restore.py globs <input_dir>/**/*.flac and writes <output_dir>/<input_dir.stem>/<relative path>, and it never
skips work. So the pipelines build trees of symlinks (real directories, symlinked files) named after the corpus, a
pending tree holds only the tracks without a valid output, and the output check accepts a file only when it is mono
48 kHz PCM_16 with exactly the raw length (soundfile and the FLAC header) and a tail that decodes.
"""

from __future__ import annotations

import heapq
import os
from pathlib import Path

from . import core as cm

TAIL_FRAMES = 4800


def ensure_link(link: Path, raw: Path) -> bool:
    """Create link -> raw (True), accept an identical existing link (False), refuse anything else."""
    if link.is_symlink():
        if os.readlink(link) != str(raw):
            raise SystemExit(f"{link} points to {os.readlink(link)}, not {raw}")
        return False
    if link.exists():
        raise SystemExit(f"{link} exists and is not a symlink")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(raw)
    return True


def output_problem(path: Path, frames: int) -> str | None:
    """None when `path` is mono 48 kHz PCM_16 with exactly `frames` frames and a tail that decodes."""
    import soundfile as sf
    if not path.exists():
        return "missing"
    try:
        info = sf.info(str(path))
        if (info.samplerate, info.channels, info.subtype) != (cm.SR, 1, "PCM_16"):
            return f"format {info.samplerate} Hz {info.channels} ch {info.subtype}"
        header = cm.read_streaminfo(path)["ns"]
        if info.frames != frames or header != frames:
            return f"length {info.frames} (FLAC header {header}), raw {frames}"
        with sf.SoundFile(str(path)) as stream:
            stream.seek(frames - TAIL_FRAMES)
            tail = stream.read(TAIL_FRAMES, dtype="int16")
        if len(tail) != TAIL_FRAMES:
            return f"tail decoded {len(tail)} of {TAIL_FRAMES} frames"
    except Exception as error:  # a truncated or corrupt file
        return f"unreadable: {type(error).__name__}: {error}"
    return None


def lpt_parts(weights: dict[str, int], parts: int) -> list[list[str]]:
    """Longest processing time first: items by decreasing weight (then key) to the lightest part (lowest index
    on ties). Each part's items are returned sorted."""
    heap = [(0, i) for i in range(parts)]
    heapq.heapify(heap)
    result: list[list[str]] = [[] for _ in range(parts)]
    for key, weight in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
        total, index = heapq.heappop(heap)
        result[index].append(key)
        heapq.heappush(heap, (total + weight, index))
    return [sorted(part) for part in result]
