"""Checks of the producer's output shards (records of 11 members, see the README).

* headers (tar headers only): every record has the 11 members with the expected byte sizes (names.pickle aside: its
  size follows the key length, 30-80 bytes); keys are unique; per producer task the record count equals the windows
  of its source tars (np.array_split of the producer's own shard listing).
* task_check: the per-task check run after each producer task (record count, member problems, errors in its log);
  writes the task's done marker.
* finalize: a done marker per task, the header check, the totals against the source manifest (a task may fall short
  by at most `max_shortfall` windows, which the producer drops by itself: all-quiet chunks and NaN records), then
  train/ and valid/ made read-only and a count file with the number of training shards.
* audio helpers: first record of a shard, frame energies, level, silence, 99% bandwidth and the packet-loss zero-run
  count (exact-zero runs of 20-200 ms inside a target).
"""

from __future__ import annotations

import collections
import io
import json
import os
import pathlib
import pickle
import re
import stat
import tarfile
from pathlib import Path

import numpy as np

from . import core as cm

TASKS = 16
# Byte sizes of the members of every record written with config/data/dialoguesidon2_producer.yaml: 2 channels x
# 120 s at 24 kHz (and 16 kHz), float32, torch.save. names.pickle varies with the key length.
MEMBER_SIZES = {"clean_16k_mixture.pth": 7681577, "clean_mixture.pth": 11521577, "input_wav.pth": 23041577,
                "input_wav_lens.pth": 1577, "noisy_16k_mixture.pth": 7681577, "noisy_input_wav.pth": 23041577,
                "noisy_input_wav16k.pth": 15361577, "noisy_mixture.pth": 11521577, "sample_rate.pickle": 18,
                "sr.index": 5, "names.pickle": None}


def producer_partition(source_train: Path, tasks: int = TASKS) -> list[list[str]]:
    """The producer's split of its source shards (glob_wds then np.array_split), in its listing order."""
    paths = list(map(str, pathlib.Path(source_train).glob("**/*.tar.gz")))
    paths += list(map(str, pathlib.Path(source_train).glob("**/*.tar")))
    return [list(part) for part in np.array_split(paths, tasks)]


def expected_per_task(source: Path, tasks: int = TASKS) -> dict:
    """Windows per producer task (t00, t01, ...) from the source manifest, and the dev windows ("dev")."""
    counts = collections.Counter()
    for row in cm.read_tsv(source / "manifest.tsv"):
        counts[(row["split"], int(row["tar"]))] += 1
    result = {}
    for index, part in enumerate(producer_partition(source / "train", tasks)):
        numbers = [int(re.search(r"data-(\d+)\.tar$", p).group(1)) for p in part]
        result[f"t{index:02d}"] = {"source_tars": [os.path.basename(p) for p in part],
                                   "records": sum(counts[("train", n)] for n in numbers)}
    result["dev"] = sum(v for (split, _), v in counts.items() if split == "dev")
    return result


def scan_tar(path: Path) -> dict:
    """Records and member problems of one producer shard, from headers only."""
    records = collections.defaultdict(dict)
    with tarfile.open(path) as stream:
        for member in stream:
            key, suffix = member.name.split(".", 1)
            records[key][suffix] = member.size
    problems = []
    for key, members in records.items():
        if set(members) != set(MEMBER_SIZES):
            problems.append(f"{key}: members {sorted(members)}")
            continue
        for suffix, size in MEMBER_SIZES.items():
            if size is not None and members[suffix] != size:
                problems.append(f"{key}: {suffix} has {members[suffix]} bytes, expected {size}")
        if not 30 <= members["names.pickle"] <= 80:
            problems.append(f"{key}: names.pickle has {members['names.pickle']} bytes")
    return {"records": len(records), "keys": sorted(records), "problems": problems}


def headers(shards: Path, source: Path | None, tasks: int = TASKS, max_shortfall: int = 0) -> dict:
    """The header check of a producer output directory (shards/{train,valid}/t<NN>/*.tar)."""
    report = {"splits": {}, "problems": [], "duplicate_keys": 0}
    seen = set()
    for split in ("train", "valid"):
        per_task = {}
        for task in sorted(p for p in (shards / split).glob("*") if p.is_dir()):
            tars = sorted(task.glob("*.tar"))
            scans = [scan_tar(t) for t in tars]
            for scan in scans:
                report["problems"].extend(scan["problems"])
                for key in scan["keys"]:
                    report["duplicate_keys"] += key in seen
                    seen.add(key)
            per_task[task.name] = {"shards": len(tars), "records": sum(s["records"] for s in scans)}
        report["splits"][split] = per_task
    if source is not None:
        expected = expected_per_task(source, tasks)
        mismatched, short = {}, {}
        for t, v in expected.items():
            if t == "dev":
                continue
            found = report["splits"]["train"].get(t, {}).get("records", 0)
            if found != v["records"]:
                target = short if 0 < v["records"] - found <= max_shortfall else mismatched
                target[t] = {"found": found, "expected": v["records"]}
        valid = sum(v["records"] for v in report["splits"]["valid"].values())
        report["expected"] = expected
        report["task_mismatches"] = mismatched
        report["task_shortfalls"] = short
        report["valid_records"] = valid
        report["pass"] = not report["problems"] and not report["duplicate_keys"] and not mismatched \
            and expected["dev"] - max_shortfall <= valid <= expected["dev"]
    else:
        report["pass"] = not report["problems"] and not report["duplicate_keys"]
    report["train_shards"] = sum(v["shards"] for v in report["splits"]["train"].values())
    report["train_records"] = sum(v["records"] for v in report["splits"]["train"].values())
    return report


def task_check(out: Path, name: str, index: int, expected: dict | None, log: Path | None,
               max_shortfall: int = 0) -> dict:
    """The check of one finished producer task `name` (t<NN>, shard index `index`) in the output directory `out`:
    its record count against `expected` (expected_per_task; None when the source has no manifest, as for the base
    root, where only the presence of records is checked), member problems and errors in its log. Writes
    out/_done/<name>.json when it passes; returns the record either way."""
    scans = [scan_tar(t) for t in sorted((out / "train" / name).glob("*.tar"))]
    records = sum(s["records"] for s in scans)
    problems = [p for s in scans for p in s["problems"]]
    valid = None
    if index == 0:
        valid = sum(scan_tar(t)["records"] for t in sorted((out / "valid" / name).glob("*.tar")))
    errors = {}
    if log is not None and log.exists():
        text = log.read_text(errors="replace")
        errors = {"Traceback": text.count("Traceback"), "ValueError": len(re.findall(r"\bValueError\b", text))}
    clean = not problems and not any(errors.values())
    if expected is None:
        want = dev = None
        ok = clean and records > 0 and (valid is None or valid > 0)
    else:
        want, dev = expected[name]["records"], expected["dev"]
        ok = clean and want - max_shortfall <= records <= want \
            and (valid is None or dev - max_shortfall <= valid <= dev)
    record = cm.run_info("produce-task", {"task": name, "shard_idx": index, "max_shortfall": max_shortfall},
                         records=records, expected=want, valid_records=valid, expected_dev=dev,
                         shards=len(scans), problems=problems[:20], log_errors=errors, passed=ok)
    if ok:
        cm.write_json(out / "_done" / f"{name}.json", record)
    return record


def finalize(shards: Path, source: Path | None, tasks: int = TASKS, count_file: str = "N_SHARDS.txt",
             max_shortfall: int = 0, params: dict | None = None) -> int:
    """Final check of a producer output directory; on success makes it read-only and writes `count_file`. Without
    `source` (no manifest, as for the base root) the record counts are not compared."""
    missing = [f"t{i:02d}" for i in range(tasks) if not (shards / "_done" / f"t{i:02d}.json").exists()]
    if missing:
        print(f"Tasks without a done marker: {missing}", flush=True)
        return 1
    report = headers(shards, source, tasks, max_shortfall)
    cm.write_json(shards / "qc_final.json", cm.run_info("finalize", params if params is not None else
                                                        {"tasks": tasks, "max_shortfall": max_shortfall},
                                                        report=report))
    if not report["pass"]:
        print(json.dumps({k: report.get(k) for k in ("task_mismatches", "valid_records", "duplicate_keys")}),
              flush=True)
        print(f"Final check FAILED; see {shards / 'qc_final.json'}", flush=True)
        return 1
    for split in ("train", "valid"):
        for path in sorted((shards / split).rglob("*"), reverse=True) + [shards / split]:
            mode = path.stat().st_mode
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    cm.write_atomic(shards / count_file, f"{report['train_shards']}\n")
    print(f"Final check passed: {report['train_records']} training records in {report['train_shards']} shards "
          f"(shortfalls: {report.get('task_shortfalls') or 'none'})", flush=True)
    return 0


# ---- audio QC ------------------------------------------------------------------------------------------------

def first_record(path: Path) -> dict:
    import torch
    record = {}
    key = None
    with tarfile.open(path) as stream:
        for member in stream:
            k, suffix = member.name.split(".", 1)
            if key is None:
                key = k
            if k != key:
                break
            data = stream.extractfile(member).read()
            if suffix.endswith(".pth"):
                record[suffix] = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            elif suffix.endswith(".pickle"):
                record[suffix] = pickle.loads(data)
            else:
                record[suffix] = data.decode()
    record["__key__"] = key
    return record


def frame_energy(x: np.ndarray, sr: int, bandpass: bool = False) -> np.ndarray:
    """dBFS per 10 ms frame over 20 ms windows (optionally 150-3,800 Hz band-passed), floored at -110 dB."""
    from scipy.signal import butter, sosfilt
    if bandpass:
        x = sosfilt(butter(4, [150, 3800], btype="bandpass", fs=sr, output="sos"), x)
    hop = sr // 100
    frames = len(x) // hop
    s = (x[:frames * hop].reshape(-1, hop).astype(np.float64) ** 2).sum(1)
    s = np.concatenate([s, [0.0]])
    return 10 * np.log10(np.maximum((s[:-1] + s[1:]) / (2 * hop), 1e-11))


def channel_stats(x: np.ndarray, sr: int = 24_000) -> dict:
    from scipy.signal import welch
    e = frame_energy(x, sr)
    top = np.sort(e)[int(0.7 * len(e)):]
    f, p = welch(x.astype(np.float64), fs=sr, nperseg=2048)
    cumulative = np.cumsum(p) / max(p.sum(), 1e-30)
    zero = x == 0
    padded = np.zeros(len(zero) + 2, np.int8)
    padded[1:-1] = zero
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = ends - starts
    interior = (starts > 0) & (ends < len(x))
    gaps = int(((lengths >= sr // 50) & (lengths <= sr // 5 + sr // 50) & interior).sum())
    return {"active_db": float(top.mean()), "silence_frac": float((e < -60).mean()),
            "bandwidth99_hz": float(f[np.searchsorted(cumulative, 0.99)]), "zero_gaps_20_200ms": gaps}


def summarize(rows: list[dict]) -> dict:
    channels = [r[f"ch{i}"] for r in rows for i in range(2)]
    pct = lambda values: {p: float(np.percentile(values, p)) for p in (10, 50, 90)} if len(values) else {}  # noqa
    summary = {name: pct([c[name] for c in channels]) for name in ("active_db", "silence_frac", "bandwidth99_hz")}
    summary["zero_gap_track_share"] = float(np.mean([c["zero_gaps_20_200ms"] >= 3 for c in channels])) \
        if channels else None
    blr = [b for r in rows for b in r.get("blr_target", []) if b is not None and np.isfinite(b)]
    if blr:
        summary["blr_target"] = pct(blr)
        summary["blr_target_above_minus30"] = int(sum(b >= -30 for b in blr))
    return summary


def audio_report(tars: list[Path], count: int, seed: int) -> list[dict]:
    """channel_stats of both target channels of the first record of `count` randomly chosen shards."""
    import random
    rng = random.Random(seed)
    rows = []
    for path in rng.sample(sorted(tars), min(count, len(tars))):
        record = first_record(path)
        wav = record["input_wav.pth"][0].numpy()
        rows.append({"shard": str(path), "names": record["names.pickle"],
                     **{f"ch{i}": channel_stats(wav[i]) for i in range(2)}})
    return rows
