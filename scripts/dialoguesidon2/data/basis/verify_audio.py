"""Verification of the downloaded Basis audio: which tracks are present and intact, and which segments are ready.
A download may be partial.

* headers (FLAC headers only): the stream header against the Lhotse length (48 kHz, mono, 16-bit, num_samples). Writes
  verify/headers.tsv and the `audio` status of every track in s0_segments.json (absent, header_ok, header_bad).
* checksums: sha256 of every present track against CHECKSUMS.sha256, and its FLAC header. Writes verify/<lang>.tsv
  (path, size, mtime_ns, sha256, expected, header, ok), verify/quarantine.tsv (mismatches: download again) and
  ready.json (a segment is ready when every one of its tracks verifies, because the bleed screen needs every other
  track of the segment). A file whose size and mtime are unchanged since its last hash is not hashed again, so a
  rerun after more audio has arrived hashes only the new files.

  python scripts/dialoguesidon2/data/basis/verify_audio.py --root BASIS_ROOT --screen WORK/screen headers
  python scripts/dialoguesidon2/data/basis/verify_audio.py --root BASIS_ROOT --screen WORK/screen checksums
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basis import corpus as bs  # noqa: E402
from basis import tables as tb  # noqa: E402
from common import core as cm  # noqa: E402


def header_status(path: Path, expected: int) -> dict:
    """The FLAC header of one track: absent, header_ok (48 kHz mono 16-bit with the expected length) or header_bad."""
    if not path.exists():
        return {"status": "absent"}
    stat = path.stat()
    row = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "expected_ns": expected}
    try:
        info = cm.read_streaminfo(path)
    except (OSError, ValueError) as error:
        return {**row, "status": "header_bad", "error": str(error)}
    row.update(sr=info["sr"], ch=info["ch"], bps=info["bps"], ns=info["ns"])
    ok = (info["sr"], info["ch"], info["bps"], info["ns"]) == (cm.SR, 1, 16, expected)
    return {**row, "status": "header_ok" if ok else "header_bad"}


def headers(root: Path, screen: Path) -> dict:
    segments = tb.load_segments(screen)
    rows = []
    counts = collections.Counter()
    for segment in segments:
        status = {}
        for speaker, rel in segment["tracks"].items():
            result = header_status(root / rel, segment["samples"])
            status[speaker] = result["status"]
            counts[result["status"]] += 1
            rows.append({"path": rel, **result})
        segment["audio"] = status
    cm.write_tsv(screen / "verify" / "headers.tsv", rows, tb.HEADER_COLUMNS)
    cm.write_json(screen / "s0_segments.json", segments)
    complete = sum(all(v == "header_ok" for v in s["audio"].values()) for s in segments)
    summary = {"tracks": len(rows), **dict(counts), "segments_with_all_headers_ok": complete,
               "segments": len(segments)}
    cm.write_json(screen / "verify" / "headers_run.json", cm.run_info("V-headers", {}, root=str(root), counts=summary))
    return summary


def _hash(job) -> dict:
    root, rel, expected_sha, expected_ns = job
    path = root / rel
    stat = path.stat()
    result = header_status(path, expected_ns)
    digest = cm.file_sha256(path)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError(f"{rel} changed while it was hashed")
    return {"path": rel, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest,
            "expected": expected_sha, "header": result["status"],
            "ok": digest == expected_sha and result["status"] == "header_ok"}


def checksums(root: Path, screen: Path, processes: int) -> dict:
    segments = tb.load_segments(screen)
    table = bs.checksums(root)
    previous = tb.read_verify(screen)
    rows: dict[str, dict] = {}
    jobs = []
    for segment in segments:
        for rel in segment["tracks"].values():
            path = root / rel
            if not path.exists():
                continue
            stat = path.stat()
            old = previous.get(rel)
            if old is not None and int(old["size"]) == stat.st_size and int(old["mtime_ns"]) == stat.st_mtime_ns:
                rows[rel] = {**old, "ok": old["ok"] == "1"}
                continue
            jobs.append((root, rel, table.get(rel), segment["samples"]))
    jobs.sort(key=lambda j: -(j[0] / j[1]).stat().st_size)
    print(f"V: {len(rows)} tracks verified before, {len(jobs)} to hash", flush=True)
    done = 0
    with Pool(processes) as pool:
        for row in pool.imap_unordered(_hash, jobs, chunksize=1):
            rows[row["path"]] = row
            done += 1
            if done % 500 == 0:
                print(f"V {done}/{len(jobs)}", flush=True)
    by_lang = collections.defaultdict(list)
    for rel in sorted(rows):
        by_lang[rel.split("/", 1)[0]].append(rows[rel])
    for lang, items in by_lang.items():
        cm.write_tsv(screen / "verify" / f"{lang}.tsv", items, tb.VERIFY_COLUMNS)
    quarantine = [r for r in rows.values() if not r["ok"]]
    cm.write_tsv(screen / "verify" / "quarantine.tsv", sorted(quarantine, key=lambda r: r["path"]),
                 tb.VERIFY_COLUMNS)
    ok_paths = {p for p, r in rows.items() if r["ok"]}
    ready, not_ready = [], {}
    for segment in segments:
        missing = [rel for rel in segment["tracks"].values() if rel not in rows]
        bad = [rel for rel in segment["tracks"].values() if rel in rows and not rows[rel]["ok"]]
        if not missing and not bad:
            ready.append(segment["segment_id"])
        else:
            not_ready[segment["segment_id"]] = {"absent": len(missing), "failed": len(bad)}
    summary = {"tracks": sum(len(s["tracks"]) for s in segments), "present": len(rows), "verified": len(ok_paths),
               "quarantined": len(quarantine), "hashed_now": len(jobs), "segments": len(segments),
               "ready": len(ready), "not_ready": len(not_ready)}
    cm.write_json(screen / "ready.json", {**cm.run_info("V", {"checksums": "CHECKSUMS.sha256"}, root=str(root),
                                                        revision=bs.release_revision(root), counts=summary),
                                          "ready": ready, "not_ready": not_ready})
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the downloaded release")
    parser.add_argument("--screen", type=Path, required=True, help="the screen directory (S0 outputs)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("headers")
    c = sub.add_parser("checksums")
    c.add_argument("--processes", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    if args.command == "headers":
        summary = headers(args.root, args.screen)
    else:
        summary = checksums(args.root, args.screen, args.processes)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
