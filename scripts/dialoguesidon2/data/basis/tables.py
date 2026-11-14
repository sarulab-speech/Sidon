"""Readers of the Basis screen tables shared by the entry points: the S0 segment and window tables, the audio
verification results and readiness, and the tracks and windows each later stage works on. JSON and TSV only."""

from __future__ import annotations

import collections
from pathlib import Path

from common import core as cm

VERIFY_COLUMNS = ["path", "size", "mtime_ns", "sha256", "expected", "header", "ok"]
HEADER_COLUMNS = ["path", "size", "mtime_ns", "sr", "ch", "bps", "ns", "expected_ns", "status"]


def load_segments(screen: Path) -> list[dict]:
    return cm.load_json(screen / "s0_segments.json")


def load_windows(screen: Path) -> list[dict]:
    rows = []
    for row in cm.read_tsv(screen / "s0_windows.tsv"):
        parsed = {k: cm.parse_cell(v) if k not in ("segment_id", "conversation_id", "lang", "reason", "eligible",
                                                    "spk0", "spk1") else v for k, v in row.items()}
        parsed["keep"] = bool(parsed["keep"])
        parsed["muted"] = bool(parsed["muted"])
        parsed["eligible"] = [x for x in row["eligible"].split(",") if x]
        parsed["reason"] = "" if row["reason"] == "NA" else row["reason"]
        rows.append(parsed)
    return rows


def kept_windows(windows: list[dict]) -> dict[str, list[dict]]:
    """S0-kept windows by segment, in offset order."""
    result = collections.defaultdict(list)
    for w in windows:
        if w["keep"]:
            result[w["segment_id"]].append(w)
    return {k: sorted(v, key=lambda w: w["offset_s"]) for k, v in result.items()}


def segment_tasks(segments: list[dict], windows: list[dict]) -> dict[str, dict]:
    """Per segment that the screen, the DNSMOS scores and restoration work on: the tested tracks (pair tracks of
    S0-kept windows and the controls) and each tested track's windows {offset: partner} (a control's windows are those
    where it would be eligible, with the top eligible partner)."""
    kept = kept_windows(windows)
    tasks = {}
    for s in segments:
        if not kept.get(s["segment_id"]) and not s["controls"]:
            continue
        per_track = collections.defaultdict(dict)
        for w in kept.get(s["segment_id"], []):
            per_track[w["spk0"]][w["offset_s"]] = w["spk1"]
            per_track[w["spk1"]][w["offset_s"]] = w["spk0"]
        for c in s["control_windows"]:
            per_track[c["x"]][c["offset_s"]] = c["partner"]
        tasks[s["segment_id"]] = {"tested": sorted(set(per_track) | set(s["controls"])),
                                  "windows": {x: dict(sorted(v.items())) for x, v in per_track.items()},
                                  "controls": list(s["controls"])}
    return tasks


def read_verify(screen: Path) -> dict[str, dict]:
    """The verification rows by release path (all languages)."""
    rows = {}
    for path in sorted((screen / "verify").glob("*.tsv")):
        if path.name in ("quarantine.tsv", "headers.tsv"):
            continue
        for row in cm.read_tsv(path):
            rows[row["path"]] = row
    return rows


def verified(screen: Path) -> set[str]:
    return {p for p, row in read_verify(screen).items() if row["ok"] == "1"}


def ready_segments(screen: Path) -> set[str]:
    path = screen / "ready.json"
    return set(cm.load_json(path)["ready"]) if path.exists() else set()


def segment_ready(segment: dict, ok_paths: set[str]) -> bool:
    return all(rel in ok_paths for rel in segment["tracks"].values())


def track_frames(segment: dict) -> int:
    return segment["samples"] // cm.HOP
