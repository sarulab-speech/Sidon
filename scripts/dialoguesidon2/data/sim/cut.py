"""Stereo assembly of the simulated conversations into the producer's source shards (CPU stage).

Refuses unless the S0 check passed and the S1 check passed (or --accept-s1-stop is given) for the current plan, and
when the output holds tars of another plan. The simulated tracks are not restored (LibriTTS-R is already clean); the
stereo records are the clean 24 kHz tracks upsampled 2x (base.upsample_48k). Then:

* manifest.tsv: key, split, tar, style, speaker_a, speaker_b, chapters_a, chapters_b, utterances_a, utterances_b,
  offsets_a, offsets_b (start of each trimmed utterance in 24 kHz samples from the record start; negative when cut
  at the start), activity_a, activity_b, activity_union, overlap_share (the S0 check's VAD on the clean tracks).
* train/data-NNNNN.tar (128; record n in tar (n // 2) % 128) and dev/data-00000.tar, records in n order, each
  written to <name>.tmp and renamed with done/<split>-<n>.json (common/layout.py write_tar, the plan hash over
  split, tar and key). Records hold exactly two members: `flac`, stereo 48 kHz PCM_16 of exactly 5,760,000 frames
  (ch0 = speaker A, ch1 = speaker B, the clean tracks upsampled with resample_poly(2, 1)), and
  `original_path.txt`, a JSON description of the record (style, speakers, chapters, utterance ids, offsets, gains,
  kinds, whole-track scales, upsampling, seed, plan hash, clean paths).
* summary.json with "status": "complete" once every tar is written (run/produce.sh checks it).

  python sim/cut.py --work SIM_WORK --processes 64
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import base as sc  # noqa: E402
from sim import qc  # noqa: E402
from sim import simulate as sm  # noqa: E402
from sim.base import cm  # noqa: E402
from common import layout as lo  # noqa: E402

MANIFEST_COLUMNS = ["key", "split", "tar", "style", "speaker_a", "speaker_b", "chapters_a", "chapters_b",
                    "utterances_a", "utterances_b", "offsets_a", "offsets_b", "activity_a", "activity_b",
                    "activity_union", "overlap_share"]


def manifest_rows(plan_rows: list[dict], s0_rows: dict[str, dict]) -> list[dict]:
    """Train tars in order, records in n order within a tar, then dev."""
    ordered = sorted(plan_rows, key=lambda r: (r["split"] != "train", r["tar"], r["n"]))
    rows = []
    for r in ordered:
        s0 = s0_rows[r["key"]]
        rows.append({**{c: r[c] for c in MANIFEST_COLUMNS[:12]},
                     **{c: s0[c] for c in ("activity_a", "activity_b", "activity_union", "overlap_share")}})
    return rows


def description(row: dict, done: dict, work: Path, digest: str) -> str:
    channels = {c: sm.channel(row, c) for c in ("a", "b")}
    record = {"style": row["style"], "split": row["split"], "n": row["n"], "seed": sc.SEED, "plan_hash": digest,
              "speakers": [row["speaker_a"], row["speaker_b"]],
              "chapters": {c: [x for x in row[f"chapters_{c}"].split(",") if x] for c in ("a", "b")},
              "utterances": {c: [e[0] for e in v] for c, v in channels.items()},
              "offsets": {c: [e[1] for e in v] for c, v in channels.items()},
              "gains_db": {c: [e[2] for e in v] for c, v in channels.items()},
              "kinds": {c: "".join(e[3] for e in v) for c, v in channels.items()},
              "scale": {c: done.get(f"scale_{c}") for c in ("a", "b")},
              "clean": [str(work / "clean" / sc.track_relpath(row["split"], row["tar"], row["key"], c))
                        for c in ("a", "b")],
              "restored": None,  # the simulated tracks are not restored
              "upsample": "scipy.signal.resample_poly(x, 2, 1), float64, int16 (sim/base.py upsample_48k)"}
    return json.dumps(record, separators=(",", ":"))


def stereo_48k(paths: list[Path]) -> tuple[np.ndarray, list[float]]:
    """The two clean 24 kHz tracks as (5,760,000, 2) int16 at 48 kHz, and each channel's overshoot scale."""
    import soundfile as sf
    channels, scales = [], []
    for path in paths:
        problem = sc.clean_problem(path)
        if problem:
            raise ValueError(f"{path}: {problem}")
        x, _ = sf.read(str(path), dtype="int16")
        y, scale = sc.upsample_48k(x)
        if len(y) != sc.OUT_FRAMES:
            raise ValueError(f"{path}: {len(y)} frames after upsampling, expected {sc.OUT_FRAMES}")
        channels.append(y)
        scales.append(scale)
    return np.stack(channels, 1), scales


def write_tar(job) -> str:
    split, index, entries, work, out, digest = job  # entries: (plan row, description text)
    clean = work / "clean"
    keys = [row["key"] for row, _ in entries]

    def records():
        for row, text in entries:
            paths = [clean / sc.track_relpath(row["split"], row["tar"], row["key"], c) for c in ("a", "b")]
            samples, scales = stereo_48k(paths)
            meta = json.loads(text)
            meta["upsample_scale"] = scales
            yield {"__key__": row["key"], "flac": lo.flac_bytes(samples),
                   "original_path.txt": json.dumps(meta, separators=(",", ":"))}

    return lo.write_tar(out / split / f"data-{index:05d}.tar", out / "done" / f"{split}-{index:05d}.json", keys,
                        records, digest)


def _safe_write(job):
    try:
        return write_tar(job)
    except Exception as error:
        return f"failed: {type(error).__name__}: {error}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, required=True, help="the simulation's work directory (SIM_WORK)")
    parser.add_argument("--out", type=Path, default=None, help="default: WORK/source")
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--accept-s1-stop", action="store_true", help="proceed although the S1 check failed")
    args = parser.parse_args(argv)
    work = args.work
    out = args.out or work / "source"
    plan_rows, digest, s1 = qc.require_gates(work, args.accept_s1_stop)
    s0_rows = {r["key"]: cm.load_json(work / "qc" / "s0" / r["split"] / f"{r['key']}.json") for r in plan_rows}
    rows = manifest_rows(plan_rows, s0_rows)
    layout = lo.plan_hash(rows)
    summary_path = out / "summary.json"
    if summary_path.exists() and cm.load_json(summary_path).get("plan_hash") != layout and \
            any((out / split).glob("*.tar") for split in ("train", "dev")):
        raise SystemExit(f"{out} holds shards of another plan; write to a new --out")
    cm.write_tsv(out / "manifest.tsv", rows, MANIFEST_COLUMNS)
    by_key = {r["key"]: r for r in plan_rows}
    groups = collections.defaultdict(list)
    for r in rows:
        row = by_key[r["key"]]
        done = cm.load_json(work / "clean" / "done" / f"{row['key']}.json")
        groups[(row["split"], row["tar"])].append((row, description(row, done, work, digest)))
    jobs = [(split, index, entries, work, out, layout) for (split, index), entries in sorted(groups.items())]
    tars = sc.PARAMS["split"]["train_tars"]
    if sorted(index for split, index, *_ in jobs if split == "train") != list(range(tars)) or \
            [index for split, index, *_ in jobs if split == "dev"] != [0]:
        raise SystemExit(f"The plan does not fill {tars} train tars and one dev tar")
    counts = {split: sum(r["split"] == split for r in rows) for split in ("train", "dev")}
    summary = cm.run_info("cut", sc.PARAMS, plan_hash=layout, sim_plan_hash=digest, s1=s1, counts=counts,
                          max_uses=sc.PARAMS["reuse"]["max_uses"],  # the plan's (qc.load_plan)
                          styles={s: dict(collections.Counter(r["style"] for r in rows if r["split"] == s))
                                  for s in ("train", "dev")},
                          hours_train=counts["train"] * sc.TRACK_S / 3600)
    status, failed = collections.Counter(), []
    with Pool(args.processes) as pool:
        for job, result in zip(jobs, pool.imap(_safe_write, jobs)):
            status[result.split(":")[0]] += 1
            if result.startswith("failed"):
                failed.append((job[0], job[1], result))
                print(f"tar {job[0]} {job[1]}: {result}", flush=True)
    complete = not failed
    cm.write_json(summary_path, cm.sanitize({**summary, "status": "complete" if complete else "incomplete",
                                             "write_status": dict(status), "write_failures": failed}))
    print(json.dumps({"counts": counts, "tars": len(jobs), **status}), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
