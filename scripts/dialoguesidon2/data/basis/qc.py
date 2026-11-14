"""Checks of the producer output of the Basis source tars (shards/<writer>/{train,valid}/t<NN>/).

* headers (tar headers only): the 11 members with the expected sizes, unique keys, per-task record counts against
  the producer's own partition of the source tars (common/shards_qc.py).
* finalize: the 16 done markers, the header check, at most --max-shortfall windows missing per task (the producer
  drops all-quiet chunks and NaN records by itself), the dev count; then train/ and valid/ are made read-only and
  N_SHARDS.txt holds the number of training shards.
* audio: --count random shards (first record of each): level, silence share, 99% bandwidth and the zero-run count per
  target channel, and the BLR of the final 24 kHz targets against the partner and every left-out participant (and
  pooled) at the raw-audio screen's lags on the window's transcript masks. Writes report.json.

  python scripts/dialoguesidon2/data/basis/qc.py --shards WORK/shards/basis_24k_120s --source WORK/source headers
  python scripts/dialoguesidon2/data/basis/qc.py --shards ... --source ... finalize [--max-shortfall N]
  python scripts/dialoguesidon2/data/basis/qc.py --shards ... --source ... audio --out WORK/qc --screen WORK/screen \
      --root BASIS_ROOT
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basis import corpus as bs  # noqa: E402
from basis import screen as sn  # noqa: E402
from basis import tables as tb  # noqa: E402
from common import core as cm  # noqa: E402
from common import crosstalk as xt  # noqa: E402
from common import shards_qc as sq  # noqa: E402


def target_blr(record: dict, context: dict) -> list[dict]:
    """Per target channel: BLR/EXC of the final 24 kHz target against every other track and pooled, at the raw-audio
    screen's lags (crosstalk.analyse on the window's masks, the target as x's energy)."""
    key = record["names.pickle"][0]
    sid, offset = bs.parse_window_key(key)
    segment = context["segments"][sid]
    manifest = context["manifest"][bs.window_key(sid, offset)]
    row = bs.read_segment(context["root"], *context["index"][sid])
    frames = segment["samples"] // cm.HOP
    masks = bs.segment_masks(row, frames)
    region = slice(offset * cm.FPS, offset * cm.FPS + cm.WINDOW_FRAMES)
    payload = cm.load_json(context["screen"] / "rows" / f"{sid}.json")
    wav = record["input_wav.pth"][0].numpy()
    result = []
    for channel, x in enumerate((manifest["spk0"], manifest["spk1"])):
        e = sq.frame_energy(wav[channel], 24_000, bandpass=True)[:cm.WINDOW_FRAMES]
        features = {s: {"energy": e if s == x else None, "zero_runs": np.zeros((0, 2), np.int64),
                        "clip_runs": np.zeros((0, 2), np.int64), "samples": cm.WINDOW_SAMPLES,
                        "frames": cm.WINDOW_FRAMES} for s in masks["speakers"]}
        fixed = {(p["x"], p["y"]): {"conv": p.get("lag_used"), "blocks": []} for p in payload["pairs"] if p["x"] == x}
        analysis = xt.analyse(sid, features, {s: masks["o"][s][region] for s in masks["speakers"]},
                              {x: np.zeros((0, 2))}, sn.PARAMS, tested=[x],
                              active={s: masks["a"][s][region] for s in masks["speakers"]}, windows={x: [0]},
                              fixed_lags=fixed, window_min_frames=sn.PARAMS["n_track"]["window_min_frames"],
                              lag_fallback=True, pooled=True, extra=False)
        for w in analysis["windows"]:
            role = "pooled" if w["y"] == "*" else ("partner" if w["y"] == manifest[f"spk{1 - channel}"] else "left_out")
            result.append({"channel": channel, "y": w["y"], "role": role, "blr": w["blr_conv"], "exc": w["exc_conv"]})
    return result


def audio(out: Path, shards: Path, source: Path, screen: Path, root: Path, count: int, seed: int) -> int:
    rng = random.Random(seed)
    tars = sorted((shards / "train").rglob("*.tar"))
    context = {"segments": {s["segment_id"]: s for s in tb.load_segments(screen)}, "screen": screen, "root": root,
               "manifest": {r["key"]: r for r in cm.read_tsv(source / "manifest.tsv")},
               "index": bs.segment_index(root)}
    rows = []
    for path in rng.sample(tars, min(count, len(tars))):
        record = sq.first_record(path)
        wav = record["input_wav.pth"][0].numpy()
        row = {"shard": str(path), "names": record["names.pickle"]}
        for i in range(2):
            row[f"ch{i}"] = sq.channel_stats(wav[i])
        try:
            blr = target_blr(record, context)
            row["blr_target_rows"] = blr
            row["blr_target"] = [r["blr"] for r in blr]
        except Exception as error:
            row["blr_target_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    report = {"records": rows, "summary": sq.summarize(rows)}
    roles = {}
    for row in rows:
        for r in row.get("blr_target_rows", []):
            if r["blr"] is not None and np.isfinite(r["blr"]):
                roles.setdefault(r["role"], []).append(r["blr"])
    report["summary"]["blr_target_by_role"] = {
        role: {"n": len(v), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
               "above_minus30": int(sum(b >= -30 for b in v))} for role, v in roles.items()}
    cm.write_json(out / "report.json", cm.run_info("qc-audio", {"count": count, "seed": seed}, report=report))
    print(json.dumps(report["summary"], indent=1, default=str), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", type=Path, required=True, help="the producer output (.../basis_24k_120s)")
    parser.add_argument("--source", type=Path, required=True, help="the source tars written by cut.py")
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("headers")
    h.add_argument("--max-shortfall", type=int, default=0)
    f = sub.add_parser("finalize")
    f.add_argument("--max-shortfall", type=int, default=0, help="windows a task may miss (default 0)")
    a = sub.add_parser("audio")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--screen", type=Path, required=True)
    a.add_argument("--root", type=Path, required=True, help="the downloaded release")
    a.add_argument("--count", type=int, default=300)
    a.add_argument("--seed", type=int, default=bs.SEED)
    args = parser.parse_args(argv)
    if args.command == "headers":
        report = sq.headers(args.shards, args.source if (args.source / "manifest.tsv").exists() else None,
                            max_shortfall=args.max_shortfall)
        cm.write_json(args.shards / "qc_headers.json", cm.run_info("qc-headers", {}, report=report))
        print(json.dumps({k: report.get(k) for k in ("pass", "train_shards", "train_records", "task_mismatches",
                                                      "task_shortfalls", "valid_records", "duplicate_keys")}),
              flush=True)
        return 0 if report["pass"] else 1
    if args.command == "finalize":
        return sq.finalize(args.shards, args.source, count_file="N_SHARDS.txt", max_shortfall=args.max_shortfall,
                           params={"tasks": sq.TASKS, "corpus": bs.NAME, "max_shortfall": args.max_shortfall})
    return audio(args.out, args.shards, args.source, args.screen, args.root, args.count, args.seed)


if __name__ == "__main__":
    sys.exit(main())
