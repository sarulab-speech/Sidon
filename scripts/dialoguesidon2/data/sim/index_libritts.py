"""Index the LibriTTS-R train subsets for the simulator (CPU stage; decodes every utterance).

Input: --libritts DIR with DIR/<subset>/data-NNNNN.tar.gz for the subsets train-clean-100, train-clean-360 and
train-other-500. Each tar.gz holds members <spk>/<chapter>/<spk>_<chapter>_<para>_<sent>.wav.audio.wav (the
original 24 kHz PCM_16 mono WAV files of LibriTTS-R), so speaker, chapter and sentence order come from the path.
sim/pack_libritts.py writes this layout from the official release; how the utterances are spread over tars does not
change the plan (only how the PCM cache is addressed).

* run: one worker per tar.gz, streamed once. Per utterance: the edge trim (frames above max(-70 dBFS, peak frame
  - 40 dB), keeping 50 ms), the active level (power mean of those frames), the peak, and the raw VAD runs used
  for placement (frames above active level - 24 dB, i.e. -50 dBFS after equalisation to -26 dBFS, merged across
  gaps below 0.5 s; 100 fps, relative to the trimmed start). Writes OUT/pcm/<subset>/data-NNNNN.npy (the
  trimmed int16 samples of the tar's utterances, concatenated, so that simulate.py reads them by offset with
  mmap instead of decompressing the tar.gz again) and then OUT/tars/<subset>/data-NNNNN.tsv (the done marker).
  Members that are not such WAVs are skipped and listed (data-NNNNN.skipped.json). Resumable per tar.
* merge: refuses unless every tar of the three subsets has its rows; writes OUT/index.tsv (sorted by speaker,
  chapter, paragraph, sentence; utterance ids unique) and OUT/summary.json (counts, hours, speakers, short
  utterances), with the run record.

  python sim/index_libritts.py --libritts LIBRITTS_DIR --out WORK/index run --processes 64
  python sim/index_libritts.py --libritts LIBRITTS_DIR --out WORK/index merge
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import base as sc  # noqa: E402
from sim.base import cm  # noqa: E402

COLUMNS = ["utt", "speaker", "chapter", "para", "sent", "subset", "tar", "member", "samples", "trim_start", "length",
           "pcm_offset", "active_db", "peak", "runs", "usable"]
MEMBER = re.compile(r"(?:.*/)?(\d+)/(\d+)/((\d+)_(\d+)_(\d+)_(\d+))\.wav\.audio\.wav")


def tar_number(path: Path) -> int:
    return int(re.fullmatch(r"data-(\d+)\.tar\.gz", path.name).group(1))


def source_tars(libritts: Path, subsets=sc.SUBSETS) -> list[tuple[str, Path]]:
    return [(subset, path) for subset in subsets for path in sorted((libritts / subset).glob("data-*.tar.gz"))]


def analyse(x: np.ndarray) -> dict:
    """Trim, level, peak and placement runs of one utterance (int16, 24 kHz)."""
    trim, vad = sc.PARAMS["trim"], sc.PARAMS["vad"]
    db = sc.frame_db(x)
    if not len(db) or not np.isfinite(db).any():
        return {"usable": False}
    threshold = max(trim["floor_dbfs"], float(db.max()) + trim["rel_peak_db"])
    active = np.flatnonzero(db > threshold)
    if not len(active):
        return {"usable": False}
    keep = int(round(trim["keep_s"] * sc.SR))
    # frame_energy's frame i averages samples [i * hop, (i + 2) * hop).
    start = max(0, int(active[0]) * sc.HOP - keep)
    end = min(len(x), (int(active[-1]) + 2) * sc.HOP + keep)
    level = float(10 * np.log10(np.mean(10 ** (db[active] / 10))))
    y = x[start:end]
    runs = sc.raw_runs(sc.frame_db(y) > level + vad["rel_level_db"], int(round(vad["raw_merge_s"] * sc.FPS)))
    return {"usable": bool(len(runs)), "trim_start": start, "length": end - start, "active_db": round(level, 3),
            "peak": int(np.abs(y.astype(np.int32)).max()) if len(y) else 0, "runs": sc.format_runs(runs),
            "samples": y}


def index_tar(job) -> str:
    import soundfile as sf
    subset, path, out = job
    number = tar_number(path)
    rows_path = out / "tars" / subset / f"data-{number:05d}.tsv"
    pcm_path = out / "pcm" / subset / f"data-{number:05d}.npy"
    if rows_path.exists() and pcm_path.exists():
        return "skipped"
    rows, chunks, offset, skipped = [], [], 0, []
    with tarfile.open(path, "r|gz") as stream:
        for member in stream:
            if not member.isfile():
                continue
            match = MEMBER.fullmatch(member.name)
            if match is None:
                skipped.append(member.name)
                continue
            speaker, chapter, utt, spk2, chapter2, para, sent = match.groups()
            if (speaker, chapter) != (spk2, chapter2):
                raise ValueError(f"{path}: {member.name} has inconsistent ids")
            x, sr = sf.read(io.BytesIO(stream.extractfile(member).read()), dtype="int16", always_2d=False)
            if sr != sc.SR or x.ndim != 1:
                raise ValueError(f"{path}: {member.name} is {sr} Hz with shape {x.shape}")
            result = analyse(x)
            row = {"utt": utt, "speaker": speaker, "chapter": chapter, "para": int(para), "sent": int(sent),
                   "subset": subset, "tar": number, "member": member.name, "samples": len(x),
                   "usable": result["usable"]}
            if result["usable"]:
                row.update({k: result[k] for k in ("trim_start", "length", "active_db", "peak", "runs")},
                           pcm_offset=offset)
                chunks.append(result["samples"])
                offset += result["length"]
            rows.append(row)
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
    del chunks
    pcm_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pcm_path.with_name(f".{pcm_path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as stream:
        np.save(stream, pcm.astype(np.int16, copy=False))
    os.replace(tmp, pcm_path)
    if skipped:
        cm.write_json(rows_path.with_suffix(".skipped.json"), skipped)
    cm.write_tsv(rows_path, rows, COLUMNS)
    return "done"


def _safe(job):
    try:
        return index_tar(job)
    except Exception as error:
        return f"failed: {type(error).__name__}: {error}"


def run(libritts: Path, out: Path, processes: int) -> int:
    jobs = [(subset, path, out) for subset, path in source_tars(libritts)]
    if not jobs:
        raise SystemExit(f"No data-*.tar.gz under {libritts}/{{{','.join(sc.SUBSETS)}}}")
    cm.write_json(out / "index_start.json", cm.run_info("index", sc.PARAMS, libritts=str(libritts), tars=len(jobs)))
    status, failures = collections.Counter(), []
    with Pool(processes) as pool:
        for job, result in zip(jobs, pool.imap(_safe, jobs)):
            status[result.split(":")[0]] += 1
            if result.startswith("failed"):
                failures.append({"tar": str(job[1]), "error": result})
                print(f"{job[1]}: {result}", flush=True)
    print(json.dumps({"tars": len(jobs), **status}), flush=True)
    if failures:
        cm.write_json(out / "index_failures.json", failures)
        return 1
    return 0


def read_rows(path: Path) -> list[dict]:
    rows = cm.read_tsv(path)
    for r in rows:
        for k in ("para", "sent", "tar", "samples"):
            r[k] = int(r[k])
        r["usable"] = r["usable"] == "1"
        if r["usable"]:
            for k in ("trim_start", "length", "pcm_offset", "peak"):
                r[k] = int(r[k])
            r["active_db"] = float(r["active_db"])
    return rows


def sort_key(row: dict):
    return int(row["speaker"]), int(row["chapter"]), row["para"], row["sent"], row["utt"]


def merge(libritts: Path, out: Path) -> int:
    expected = source_tars(libritts)
    missing = [f"{s}/{p.name}" for s, p in expected
               if not (out / "tars" / s / f"data-{tar_number(p):05d}.tsv").exists()
               or not (out / "pcm" / s / f"data-{tar_number(p):05d}.npy").exists()]
    if missing or not expected:
        print(f"Index incomplete: {len(missing)} of {len(expected)} tars have no rows (e.g. {missing[:3]})", flush=True)
        return 1
    rows, skipped = [], []
    for subset, path in expected:
        rows += read_rows(out / "tars" / subset / f"data-{tar_number(path):05d}.tsv")
        side = out / "tars" / subset / f"data-{tar_number(path):05d}.skipped.json"
        skipped += [f"{subset}/{path.name}:{name}" for name in (cm.load_json(side) if side.exists() else [])]
    ids = collections.Counter(r["utt"] for r in rows)
    duplicates = [u for u, n in ids.items() if n > 1]
    if duplicates:
        raise SystemExit(f"{len(duplicates)} utterance ids occur twice (e.g. {duplicates[:3]})")
    rows.sort(key=sort_key)
    cm.write_tsv(out / "index.tsv", rows, COLUMNS)
    usable = [r for r in rows if r["usable"]]
    subsets_of = collections.defaultdict(set)
    for r in rows:
        subsets_of[r["speaker"]].add(r["subset"])
    pad = sc.PARAMS["vad"]["pad_s"]
    short = sum(1 for r in usable if (lambda runs: (runs[-1, 1] - runs[0, 0]) / sc.FPS + 2 * pad)(
        sc.parse_runs(r["runs"])) <= sc.PARAMS["conversation"]["interjection_max_s"])
    summary = {
        "utterances": len(rows), "usable": len(usable), "unusable": len(rows) - len(usable),
        "skipped_members": len(skipped), "skipped_examples": skipped[:10],
        "speakers": len(subsets_of), "speakers_in_several_subsets": sum(len(v) > 1 for v in subsets_of.values()),
        "hours_raw": sum(r["samples"] for r in rows) / sc.SR / 3600,
        "hours_trimmed": sum(r["length"] for r in usable) / sc.SR / 3600,
        "by_subset": {s: {"utterances": sum(r["subset"] == s for r in rows),
                          "speakers": sum(s in v for v in subsets_of.values()),
                          "hours_trimmed": sum(r["length"] for r in usable if r["subset"] == s) / sc.SR / 3600}
                      for s in sc.SUBSETS},
        "short_utterances": short,
        "active_db": sc.pct([r["active_db"] for r in usable]),
        "seconds_per_utterance": sc.pct([r["length"] / sc.SR for r in usable]),
    }
    cm.write_json(out / "summary.json", cm.run_info("index", sc.PARAMS, libritts=str(libritts), tars=len(expected),
                                                    summary=summary))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def load_index(path: Path) -> list[dict]:
    """index.tsv rows (usable and not)."""
    return read_rows(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--libritts", type=Path, required=True, help="directory with <subset>/data-NNNNN.tar.gz")
    parser.add_argument("--out", type=Path, required=True, help="index directory (WORK/index)")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--processes", type=int, default=os.cpu_count())
    sub.add_parser("merge")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.libritts, args.out, args.processes)
    return merge(args.libritts, args.out)


if __name__ == "__main__":
    sys.exit(main())
