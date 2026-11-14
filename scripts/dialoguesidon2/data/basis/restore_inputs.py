"""Restoration inputs and output checks of the Basis pipeline (restoration: common/sidon_restore.py with Sidon v0.1,
through run/restore.sh).

* build (symlinks only): the restoration list is every track in the pair of at least one S0-kept window plus the
  transcript-copy controls (restored for the restored-audio check of cut.py, never trained on). The tracks are
  balanced by length into 16 parts (longest processing time first): restore_input/p<NN>/basis-1500/<release path>
  symlinks to the raw tracks (real directories, symlinked files), and restore_input/restore_list.tsv. Links are made
  for all tracks; the later commands restore only verified ones.
* require-v --parts 0-7: refuses (exit 1) unless the audio verification has finished for the parts: ready.json
  exists, and every present track of the parts has a verification row (verify/<lang>.tsv) whose size and mtime match
  the file. Absent tracks (a partial download) are reported, not refused.
* pending --parts 0-7 --into DIR: the same check first, then per part a tree with only the verified tracks whose
  restored output is missing or invalid: DIR/p<NN>/basis-1500/<release path>. Restoration writes
  <output_dir>/<input_dir name>/<relative path>, so every part's outputs land in restored/basis-1500/<release path>.
  Unverified tracks are left pending (listed).
* verify --parts 0-15 (output check): every expected output of the parts is mono 48 kHz PCM_16 with the raw length and
  a tail that decodes; restored/basis-1500.check.p<NN>.json per part.

  python scripts/dialoguesidon2/data/basis/restore_inputs.py --root BASIS_ROOT --screen WORK/screen \\
      --data-root WORK build
  python scripts/dialoguesidon2/data/basis/restore_inputs.py ... pending --parts 0-15 --into SCRATCH/pending
  python scripts/dialoguesidon2/data/basis/restore_inputs.py ... verify --parts 0-15
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basis import corpus as bs  # noqa: E402
from basis import tables as tb  # noqa: E402
from common import core as cm  # noqa: E402
from common import restore as rs  # noqa: E402

PARTS = 16
CLASSES = ("verified", "quarantined", "absent", "unhashed", "stale")
LIST_COLUMNS = ["part", "segment_id", "speaker", "role", "path", "raw", "frames"]


def restore_list(segments: list[dict], windows: list[dict]) -> list[dict]:
    """(segment, speaker, role) of every pair track of an S0-kept window ("pair") and every control."""
    kept = tb.kept_windows(windows)
    items = {}
    for s in segments:
        sid = s["segment_id"]
        for w in kept.get(sid, []):
            for x in (w["spk0"], w["spk1"]):
                items[(sid, x)] = "pair"
        for x in s["controls"]:
            items.setdefault((sid, x), "control")
    by_id = {s["segment_id"]: s for s in segments}
    return [{"segment_id": sid, "speaker": x, "role": role, "path": by_id[sid]["tracks"][x],
             "frames": by_id[sid]["samples"]} for (sid, x), role in sorted(items.items())]


def build(root: Path, screen: Path, data_root: Path, parts: int = PARTS) -> dict:
    rows = restore_list(tb.load_segments(screen), tb.load_windows(screen))
    assignment = rs.lpt_parts({r["path"]: r["frames"] for r in rows}, parts)
    part_of = {path: i for i, part in enumerate(assignment) for path in part}
    created = existing = 0
    for r in rows:
        r["part"] = part_of[r["path"]]
        r["raw"] = str(root / r["path"])
        link = data_root / "restore_input" / f"p{r['part']:02d}" / bs.NAME / r["path"]
        if rs.ensure_link(link, root / r["path"]):
            created += 1
        else:
            existing += 1
    rows.sort(key=lambda r: (r["part"], r["path"]))
    cm.write_tsv(data_root / "restore_input" / "restore_list.tsv", rows, LIST_COLUMNS)
    hours = [sum(r["frames"] for r in rows if r["part"] == i) / cm.SR / 3600 for i in range(parts)]
    counts = {"tracks": len(rows), "pairs": sum(r["role"] == "pair" for r in rows),
              "controls": sum(r["role"] == "control" for r in rows),
              "track_hours": round(sum(r["frames"] for r in rows) / cm.SR / 3600, 1),
              "part_hours": [round(h, 2) for h in hours], "part_ratio": max(hours) / min(hours) if min(hours) else None,
              "links_created": created, "links_existing": existing}
    cm.write_json(data_root / "restore_input" / "RUN_INFO.json",
                  cm.run_info("restore-inputs", {"name": bs.NAME, "parts": parts}, screen=str(screen), counts=counts))
    return counts


def read_list(data_root: Path) -> list[dict]:
    rows = cm.read_tsv(data_root / "restore_input" / "restore_list.tsv")
    for r in rows:
        r["part"], r["frames"] = int(r["part"]), int(r["frames"])
    return rows


def _check(job):
    path, frames, restored = job
    return path, rs.output_problem(restored / path, frames)


def check(data_root: Path, parts: list[int], processes: int) -> dict[str, str | None]:
    restored = data_root / "restored" / bs.NAME
    jobs = [(r["path"], r["frames"], restored) for r in read_list(data_root) if r["part"] in parts]
    with Pool(processes) as pool:
        return dict(pool.map(_check, jobs, chunksize=8))


def classify(root: Path, screen: Path, rows: list[dict]) -> dict[str, str]:
    """The verification state of each listed track now: absent (no file), unhashed (no verification row), stale
    (the file's size or mtime differs from its row), verified (current row, sha256 and header ok) or quarantined
    (current row, not ok)."""
    v_rows = tb.read_verify(screen)
    result = {}
    for r in rows:
        try:
            stat = (root / r["path"]).stat()
        except FileNotFoundError:
            result[r["path"]] = "absent"
            continue
        row = v_rows.get(r["path"])
        if row is None:
            result[r["path"]] = "unhashed"
        elif (int(row["size"]), int(row["mtime_ns"])) != (stat.st_size, stat.st_mtime_ns):
            result[r["path"]] = "stale"
        else:
            result[r["path"]] = "verified" if row["ok"] == "1" else "quarantined"
    return result


def v_status(root: Path, screen: Path, data_root: Path, parts: list[int]) -> dict:
    """Whether the verification (verify_audio.py checksums) has finished for the tracks of `parts`: ready.json exists
    and every present track has a current verification row (same size and mtime as the file). Absent tracks are
    counted, not refused."""
    rows = [r for r in read_list(data_root) if r["part"] in parts]
    classes = classify(root, screen, rows)
    count = collections.Counter(classes.values())
    first = {c: next(p for p, v in sorted(classes.items()) if v == c) for c in count}
    ready_json = (screen / "ready.json").exists()
    problems = []
    if not ready_json:
        problems.append(f"no {screen / 'ready.json'}")
    if count["unhashed"]:
        problems.append(f"{count['unhashed']} present tracks without a verification row (e.g. {first['unhashed']})")
    if count["stale"]:
        problems.append(f"{count['stale']} tracks changed since they were verified (e.g. {first['stale']})")
    if rows and not count["verified"] and not problems:
        problems.append("no track of these parts is verified")
    return {"parts": parts, "tracks": len(rows), **{c: count[c] for c in CLASSES}, "ready_json": ready_json,
            "complete": not problems, "problems": problems}


def require_v(root: Path, screen: Path, data_root: Path, parts: list[int]) -> dict:
    """v_status, printed; SystemExit(1) with the reasons unless the verification has finished for these parts."""
    status = v_status(root, screen, data_root, parts)
    print(json.dumps({"verification": status}), flush=True)
    if not status["complete"]:
        raise SystemExit("The verification has not finished for parts " + ",".join(map(str, parts)) + ": "
                         + "; ".join(status["problems"]) + ". Run verify_audio.py checksums first.")
    return status


def pending(root: Path, screen: Path, data_root: Path, into: Path, parts: list[int], processes: int) -> dict:
    require_v(root, screen, data_root, parts)
    rows = [r for r in read_list(data_root) if r["part"] in parts]
    classes = classify(root, screen, rows)
    problems = check(data_root, parts, processes)
    todo, unverified = [], []
    for r in rows:
        if problems[r["path"]] is None:
            continue
        if classes[r["path"]] != "verified":
            unverified.append(r["path"])
            continue
        todo.append(r)
        rs.ensure_link(into / f"p{r['part']:02d}" / bs.NAME / r["path"], root / r["path"])
    counts = {"parts": parts, "expected": len(rows), "valid": sum(problems[r["path"]] is None for r in rows),
              "pending": len(todo), "unverified": len(unverified),
              "pending_parts": sorted({r["part"] for r in todo}),
              "problems": sorted({(problems[r["path"]] or "").split(" ")[0] for r in todo})}
    print(json.dumps(counts), flush=True)
    return counts


def verify(data_root: Path, parts: list[int], processes: int) -> int:
    problems = check(data_root, parts, processes)
    part_of = {r["path"]: r["part"] for r in read_list(data_root)}
    failed = 0
    for part in parts:
        failures = [{"path": p, "problem": v} for p, v in sorted(problems.items()) if part_of[p] == part and v]
        expected = sum(1 for p in problems if part_of[p] == part)
        failed += bool(failures)
        cm.write_json(data_root / "restored" / f"{bs.NAME}.check.p{part:02d}.json",
                      cm.run_info("restore-check", {"name": bs.NAME, "part": part}, expected=expected, failures=failures,
                                  passed=not failures))
    total = sum(v is not None for v in problems.values())
    print(json.dumps({"parts": parts, "expected": len(problems), "failures": total}), flush=True)
    return 0 if not failed else 1


def g1_complete(data_root: Path, parts: int = PARTS) -> tuple[bool, list[int]]:
    """Whether every part has a passed output check (cut.py reruns `verify` anyway)."""
    missing = [p for p in range(parts) if not (data_root / "restored" / f"{bs.NAME}.check.p{p:02d}.json").exists()
               or not cm.load_json(data_root / "restored" / f"{bs.NAME}.check.p{p:02d}.json")["passed"]]
    return not missing, missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the downloaded release")
    parser.add_argument("--screen", type=Path, required=True, help="the screen directory")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="the work directory (restore_input/ and restored/ below it)")
    parser.add_argument("--processes", type=int, default=16)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    r = sub.add_parser("require-v")
    r.add_argument("--parts", default=f"0-{PARTS - 1}")
    p = sub.add_parser("pending")
    p.add_argument("--parts", default=f"0-{PARTS - 1}")
    p.add_argument("--into", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--parts", default=f"0-{PARTS - 1}")
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build(args.root, args.screen, args.data_root), indent=1), flush=True)
        return 0
    parts = cm.parse_parts(args.parts, PARTS)
    if args.command == "require-v":
        require_v(args.root, args.screen, args.data_root, parts)
        return 0
    if args.command == "pending":
        pending(args.root, args.screen, args.data_root, args.into, parts, args.processes)
        return 0
    return verify(args.data_root, parts, args.processes)


if __name__ == "__main__":
    sys.exit(main())
