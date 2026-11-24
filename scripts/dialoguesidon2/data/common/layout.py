"""Source-shard layout of the data pipelines: speaker-graph components, the speaker-disjoint dev split, blocks of
consecutive windows, the seeded tar assignment, the plan hash and the tar writer.

A group (conversation) joins all of its participants in the speaker graph, so a dev speaker never appears in a train
group, not even as a participant who was left out of every pair.
"""

from __future__ import annotations

import collections
import hashlib
import heapq
import io
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import core as cm


def components(groups: dict[str, Iterable[str]]) -> list[list[str]]:
    """Connected components of the speaker graph (speakers joined within a group), as sorted group lists."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    members = {g: list(p) for g, p in groups.items()}
    for people in members.values():
        for other in people[1:]:
            ra, rb = find(people[0]), find(other)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    result = collections.defaultdict(list)
    for group, people in members.items():
        result[find(people[0])].append(group)
    return [sorted(v) for v in result.values()]


def dev_split(windows_per_group: dict[str, int], participants: dict[str, Iterable[str]], minimum: int) -> set[str]:
    """Components over the groups with kept windows, ordered by (kept windows, smallest group id), whole
    components until >= minimum windows (used for OpenYap-1k)."""
    groups = {g: participants[g] for g, n in windows_per_group.items() if n > 0}
    ordered = sorted(components(groups), key=lambda comp: (sum(windows_per_group[g] for g in comp), comp[0]))
    chosen: set[str] = set()
    total = 0
    for comp in ordered:
        if total >= minimum:
            break
        chosen.update(comp)
        total += sum(windows_per_group[g] for g in comp)
    return chosen


def dev_split_all(windows_per_group: dict[str, int], participants: dict[str, Iterable[str]],
                  minimum: int) -> tuple[set[str], list[dict]]:
    """Components over every group's participants (groups without kept windows included, so a
    left-out participant links groups too), sorted by (kept windows, smallest group id), components without
    kept windows skipped, whole components added until >= minimum windows. Returns the dev groups (all groups of
    the chosen components) and a row per chosen component (used for Basis)."""
    comps = components(participants)
    ordered = sorted(comps, key=lambda comp: (sum(windows_per_group.get(g, 0) for g in comp), comp[0]))
    chosen: set[str] = set()
    rows = []
    total = 0
    for comp in ordered:
        windows = sum(windows_per_group.get(g, 0) for g in comp)
        if windows == 0:
            continue
        if total >= minimum:
            break
        chosen.update(comp)
        total += windows
        rows.append({"groups": comp, "windows": windows})
    return chosen, rows


def blocks_of(offsets: list[int], block_max: int) -> list[list[int]]:
    """Runs of consecutive 120 s offsets, cut into blocks of at most block_max windows."""
    result, run = [], []
    for offset in sorted(offsets):
        if run and offset != run[-1] + cm.WINDOW_S:
            result.extend(run[i:i + block_max] for i in range(0, len(run), block_max))
            run = []
        run.append(offset)
    if run:
        result.extend(run[i:i + block_max] for i in range(0, len(run), block_max))
    return result


def assign(blocks: list[tuple[str, list[int]]], tars: int, seed: int) -> list[list[tuple[str, int, int]]]:
    """Shuffle blocks with default_rng(seed) and give each to the tar with the fewest windows (lowest index
    on ties). Returns per tar the (unit, offset, block id) of its windows in order."""
    order = np.random.default_rng(seed).permutation(len(blocks))
    heap = [(0, i) for i in range(tars)]
    heapq.heapify(heap)
    contents: list[list[tuple[str, int, int]]] = [[] for _ in range(tars)]
    for block_id in order:
        count, index = heapq.heappop(heap)
        unit, offsets = blocks[int(block_id)]
        contents[index].extend((unit, offset, int(block_id)) for offset in offsets)
        heapq.heappush(heap, (count + len(offsets), index))
    return contents


def plan_hash(rows: list[dict]) -> str:
    text = "\n".join(f"{r['split']}\t{r['tar']}\t{r['key']}" for r in rows)
    return hashlib.sha256(text.encode()).hexdigest()


def flac_bytes(samples: np.ndarray) -> bytes:
    import soundfile as sf
    stream = io.BytesIO()
    sf.write(stream, samples, cm.SR, format="FLAC", subtype="PCM_16")
    return stream.getvalue()


def read_window(paths: list[Path], offset_s: int, labels: list[str] | None = None) -> np.ndarray:
    """The given mono tracks' samples of one 120 s window as (frames, channels) int16, read losslessly."""
    import soundfile as sf
    channels = []
    for i, path in enumerate(paths):
        with sf.SoundFile(str(path)) as stream:
            stream.seek(offset_s * cm.SR)
            data = stream.read(cm.WINDOW_SAMPLES, dtype="int16", always_2d=False)
        if len(data) != cm.WINDOW_SAMPLES:
            raise ValueError(f"{labels[i] if labels else path} at {offset_s} s: read {len(data)} frames")
        channels.append(data)
    return np.stack(channels, 1)


def write_tar(target: Path, marker: Path, keys: list[str], records: Callable[[], Iterable[dict]],
              digest: str) -> str:
    """One source tar: written to <name>.tmp and renamed, then a done marker with the plan hash and the keys.
    An existing tar with a marker of the same plan is skipped; one of another plan is an error."""
    import webdataset as wds
    if target.exists() and marker.exists():
        done = cm.load_json(marker)
        if done["plan_hash"] == digest and done["keys"] == keys:
            return "skipped"
        raise ValueError(f"{target} was written for another plan; move the source directory aside")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as stream:
        writer = wds.TarWriter(stream)
        for record in records():
            writer.write(record)
        writer.close()
    os.replace(tmp, target)
    cm.write_json(marker, {"plan_hash": digest, "records": len(keys), "keys": keys, "bytes": target.stat().st_size})
    return "done"
