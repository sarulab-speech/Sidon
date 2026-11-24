"""Restoration inputs and output checks of OpenYap-1k (restoration: common/sidon_restore.py via run/restore.sh).

* build (symlinks and FLAC headers only): <work>/restore_input/openyap-1k/<conv>/speaker_{a,b}.flac as symlinks to
  the raw tracks, inside real directories, for every conversation with a window kept by the metadata screen plus the
  transcript-bleed conversations (restored as controls of the post-restoration check, never used for training);
  restore_list.tsv records each track's raw length (FLAC header). The restoration writes
  <work>/restored/openyap-1k/<conv>/speaker_{a,b}.flac.
* pending: a tree with only the tracks whose restored output is missing or invalid, named .../openyap-1k so the
  outputs land in the same place (the restoration itself never skips work).
* verify: every expected output is mono 48 kHz PCM_16, its length equals the raw length (FLAC header), and its last
  0.1 s decodes. Writes restored/openyap-1k.verify.json; exits 1 on any failure.

  python scripts/dialoguesidon2/data/openyap/restore_inputs.py --root OPENYAP/conversations --s0 WORK/screen \\
      --work WORK build
  python scripts/dialoguesidon2/data/openyap/restore_inputs.py ... pending --into SCRATCH/pending
  python scripts/dialoguesidon2/data/openyap/restore_inputs.py ... verify
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import restore as rs  # noqa: E402
from openyap import corpus as oc  # noqa: E402

NAME = oc.NAME
LIST_COLUMNS = ["conv", "s", "role", "raw", "frames"]


def restore_set(s0_rows: list[dict]) -> list[tuple[str, str]]:
    """(conversation, role): the metadata screen's conversations with kept windows, and the transcript-bleed
    controls."""
    chosen = {}
    for r in s0_rows:
        if r.get("conv_ok") and r.get("n_keep", 0) > 0:
            chosen[r["c"]] = "candidate"
        elif r.get("transcript_bleed"):
            chosen[r["c"]] = "control"
    return sorted(chosen.items())


def build(s0_dir: Path, root: Path, data_root: Path) -> dict:
    s0_rows = oc.load_json(s0_dir / "s0_conversations.json")
    tree = data_root / "restore_input" / NAME
    rows, created, existing = [], 0, 0
    for conv, role in restore_set(s0_rows):
        (tree / conv).mkdir(parents=True, exist_ok=True)
        for s in oc.SPEAKERS:
            raw = root / conv / f"speaker_{s}.flac"
            if rs.ensure_link(tree / conv / f"speaker_{s}.flac", raw):
                created += 1
            else:
                existing += 1
            rows.append({"conv": conv, "s": s, "role": role, "raw": str(raw),
                         "frames": oc.read_streaminfo(raw)["ns"]})
    oc.write_tsv(data_root / "restore_input" / "restore_list.tsv", rows, LIST_COLUMNS)
    hours = sum(r["frames"] for r in rows) / oc.SR / 3600
    counts = {"conversations": len(rows) // 2, "candidates": sum(r["role"] == "candidate" for r in rows) // 2,
              "controls": sum(r["role"] == "control" for r in rows) // 2, "tracks": len(rows),
              "track_hours": hours, "links_created": created, "links_existing": existing}
    oc.write_json(data_root / "restore_input" / "RUN_INFO.json",
                  oc.run_info("restore-inputs", {"name": NAME}, s0=str(s0_dir), counts=counts))
    return counts


def output_ok(job) -> tuple[str, str, str | None]:
    """(conv, s, problem or None) for one expected restored track."""
    conv, s, frames, restored = job
    return conv, s, rs.output_problem(restored / conv / f"speaker_{s}.flac", frames)


def check_all(data_root: Path, processes: int) -> list[tuple[str, str, str | None]]:
    rows = oc.read_tsv(data_root / "restore_input" / "restore_list.tsv")
    restored = data_root / "restored" / NAME
    jobs = [(r["conv"], r["s"], int(r["frames"]), restored) for r in rows]
    with Pool(processes) as pool:
        return pool.map(output_ok, jobs, chunksize=8)


def pending(data_root: Path, into: Path, processes: int) -> dict:
    results = check_all(data_root, processes)
    tree = into / NAME
    rows = {(r["conv"], r["s"]): r for r in oc.read_tsv(data_root / "restore_input" / "restore_list.tsv")}
    todo = [(c, s, problem) for c, s, problem in results if problem is not None]
    for conv, s, _ in todo:
        (tree / conv).mkdir(parents=True, exist_ok=True)
        link = tree / conv / f"speaker_{s}.flac"
        if not link.is_symlink():
            link.symlink_to(rows[(conv, s)]["raw"])
    counts = {"expected": len(results), "valid": len(results) - len(todo), "pending": len(todo),
              "problems": sorted({p.split(" ")[0] for _, _, p in todo})}
    print(json.dumps(counts), flush=True)
    return counts


def verify(data_root: Path, processes: int) -> int:
    results = check_all(data_root, processes)
    failures = [{"conv": c, "s": s, "problem": p} for c, s, p in results if p is not None]
    record = oc.run_info("restore-verify", {"name": NAME}, expected=len(results), failures=failures,
                         passed=not failures)
    oc.write_json(data_root / "restored" / f"{NAME}.verify.json", record)
    print(json.dumps({"expected": len(results), "failures": len(failures)}), flush=True)
    return 0 if not failures else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the release's conversations/ directory")
    parser.add_argument("--s0", type=Path, required=True, help="directory of the S0 outputs")
    parser.add_argument("--work", type=Path, required=True, help="data directory (restore_input/, restored/)")
    parser.add_argument("--processes", type=int, default=16)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    p = sub.add_parser("pending")
    p.add_argument("--into", type=Path, required=True)
    sub.add_parser("verify")
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build(args.s0, args.root, args.work), indent=1))
        return 0
    if args.command == "pending":
        pending(args.work, args.into, args.processes)
        return 0
    return verify(args.work, args.processes)


if __name__ == "__main__":
    sys.exit(main())
