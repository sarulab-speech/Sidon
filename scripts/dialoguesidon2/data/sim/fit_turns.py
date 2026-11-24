"""Fit conversational turn-taking on screened OpenYap-1k windows (CPU stage; reads word masks only, no audio).

The released simulation used the table shipped as sim/turn_fit.json; this script recomputes such a table from the
outputs of the OpenYap-1k pipeline:

* the windows: the train windows of the OpenYap source manifest (WORK/source/manifest.tsv, split == train), which
  passed every screen; with --decided, the script also checks that each of them is kept (keep == 1) in that
  decisions table (openyap/screen_crosstalk.py rescore writes s1_windows_decided_<tag>.tsv);
* the activity: the stored word masks o_a / o_b of the crosstalk screen's frames/<conv>.npz (100 fps, packed bits,
  np.unpackbits(...)[:frames]; echo words removed, padded 0.1 s, gaps below 0.3 s filled), over
  [offset, offset + 120 s) of both tracks. The npz `lags` arrays are bleed lags, not track offsets, and are not used.

Fitted with sim/base.py segment (shared with the planner and the S1 check): turn duration, within-turn pause,
floor-transfer offset (FTO, negative = overlap), interjections (spurts <= 1.5 s strictly inside a spurt of the other
speaker: durations and the rate per second of turn time), and per-speaker activity, union activity and overlap share
per window. Writes OUT/fit.json (1,001-point quantile tables with n and p10/p50/p90, the rates and a summary; the
layout of sim/turn_fit.json) and OUT/windows.tsv (per-window statistics).

  python sim/fit_turns.py --manifest OY_WORK/source/manifest.tsv --frames OY_WORK/screen/frames --out SIM_WORK/fit
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import base as sc  # noqa: E402
from sim.base import cm  # noqa: E402

WINDOW_COLUMNS = ["conv", "offset_s", "activity_a", "activity_b", "activity_union", "overlap_share", "turns",
                  "transfers", "interjections", "turn_seconds"]
SHIPPED = Path(__file__).resolve().parent / "turn_fit.json"


def fit_windows(manifest: Path, decided: Path | None = None) -> list[tuple[str, int]]:
    """The manifest's train windows; with `decided`, each must be kept there."""
    windows = sorted({(r["conv"], int(r["offset_s"])) for r in cm.read_tsv(manifest) if r["split"] == "train"})
    if not windows:
        raise SystemExit(f"No train windows in {manifest}")
    if decided is not None:
        kept = {(r["conv"], int(r["offset_s"])) for r in cm.read_tsv(decided) if r["keep"] in ("1", "True")}
        outside = [w for w in windows if w not in kept]
        if outside:
            raise SystemExit(f"{len(outside)} manifest train windows are not kept in {decided} (e.g. {outside[:3]})")
    return windows


def window_hash(windows: list[tuple[str, int]]) -> str:
    return hashlib.sha256("".join(f"{c}\t{o}\n" for c, o in windows).encode()).hexdigest()


def conversation_masks(frames_dir: Path, conv: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(frames_dir / f"{conv}.npz")
    frames = int(data["frames"])
    return (np.unpackbits(data["o_a"])[:frames].astype(bool), np.unpackbits(data["o_b"])[:frames].astype(bool))


def analyse_window(a: np.ndarray, b: np.ndarray) -> dict:
    if len(a) != sc.FRAMES or len(b) != sc.FRAMES:
        raise ValueError(f"window masks have {len(a)} and {len(b)} frames, not {sc.FRAMES}")
    return {**sc.activity(a, b), **sc.segment(a, b)}


def conversation_job(job) -> list[dict]:
    frames_dir, conv, offsets = job
    a, b = conversation_masks(frames_dir, conv)
    rows = []
    for offset in offsets:
        region = slice(offset * sc.FPS, offset * sc.FPS + sc.FRAMES)
        rows.append({"conv": conv, "offset_s": offset, **analyse_window(a[region], b[region])})
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Quantile tables and the interjection rate over the analysed windows."""
    pooled = collections.defaultdict(list)
    for r in rows:
        for name in ("turn_s", "pause_s", "fto_s", "interjection_s"):
            pooled[name] += r[name]
        pooled["activity_speaker"] += [r["activity_a"], r["activity_b"]]
        for name in ("activity_a", "activity_b", "activity_union", "overlap_share"):
            pooled[name].append(r[name])
    turn_seconds = sum(r["turn_seconds"] for r in rows)
    interjections = sum(r["interjections"] for r in rows)
    tables = {name: sc.table(values) for name, values in pooled.items()}
    return {"tables": tables,
            "rates": {"interjections": interjections, "turn_seconds": turn_seconds,
                      "interjections_per_turn_s": interjections / turn_seconds if turn_seconds else 0.0},
            "summary": {name: {k: tables[name][k] for k in ("n", "p10", "p50", "p90")} for name in tables}}


def fit(windows: list[tuple[str, int]], frames_dir: Path, processes: int) -> tuple[dict, list[dict]]:
    by_conv = collections.defaultdict(list)
    for conv, offset in windows:
        by_conv[conv].append(offset)
    jobs = [(frames_dir, conv, offsets) for conv, offsets in sorted(by_conv.items())]
    rows = []
    with Pool(processes) as pool:
        for part in pool.imap(conversation_job, jobs, chunksize=4):
            rows += part
    return aggregate(rows), rows


def fit_text(record: dict) -> str:
    """JSON with one line per table, so the file stays readable and diffs stay small."""
    lines = ["{"]
    keys = [k for k in record if k != "tables"]
    for k in keys:
        lines.append(f" {json.dumps(k)}: {json.dumps(record[k])},")
    lines.append(' "tables": {')
    names = list(record["tables"])
    for i, name in enumerate(names):
        lines.append(f"  {json.dumps(name)}: {json.dumps(record['tables'][name])}" + ("," if i < len(names) - 1 else ""))
    lines.append(" }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def load_fit(path: Path = SHIPPED) -> dict:
    record = cm.load_json(path)
    for name in ("turn_s", "pause_s", "fto_s"):
        if not record["tables"][name]["q"]:
            raise SystemExit(f"{path}: the {name} table is empty")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="OpenYap source manifest.tsv")
    parser.add_argument("--frames", type=Path, required=True, help="frames/ of the OpenYap crosstalk screen")
    parser.add_argument("--decided", type=Path, default=None,
                        help="optional s1_windows_decided_<tag>.tsv: every train window must be kept there")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)
    windows = fit_windows(args.manifest, args.decided)
    result, rows = fit(windows, args.frames, args.processes)
    cm.write_tsv(args.out / "windows.tsv", [{**r, "transfers": len(r["fto_s"])} for r in rows], WINDOW_COLUMNS)
    record = {"fit": {"source": "OpenYap-1k word masks of the screened train windows", "windows": len(windows),
                      "windows_sha256": window_hash(windows), "fps": sc.FPS, "window_s": sc.TRACK_S,
                      "interjection_max_s": sc.PARAMS["conversation"]["interjection_max_s"]},
              **result}
    cm.write_atomic(args.out / "fit.json", fit_text(cm.sanitize(record)))
    cm.write_json(args.out / "fit_run.json", cm.run_info("fit", sc.PARAMS, manifest=str(args.manifest),
                                                         frames=str(args.frames), windows=len(windows)))
    print(json.dumps({"windows": len(windows), "rates": result["rates"],
                      "summary": {k: {p: v[p] for p in ("p10", "p50", "p90")} for k, v in result["summary"].items()}},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
