"""Checks of the simulated conversations: S0 and S1 on the clean tracks, and the producer output.

* s0 (CPU, after render): per record of WORK/clean/plan.tsv, decodes both clean tracks: mono 24 kHz PCM_16 of
  exactly 2,880,000 samples, not all zero, each channel >= 5% active and speech in every 30 s quarter (the energy
  VAD mask of sim/base.py), the key pattern and n, style = parity of n; whole set: unique keys and the plan's reuse
  cap (at most max_uses uses per utterance and ceil(max_uses / 2) per style, never twice in one record, never twice
  with the same partner). Writes WORK/qc/s0/<split>/<key>.json (also the turn statistics S1 needs; resumable),
  WORK/qc/s0.tsv and WORK/qc/S0.json.
* s1 (whole set): conversation-style records against the turn-taking table (union activity, overlap share, FTO and
  turn length: p10/p50/p90 reported, medians within +-20%, +-0.01 s for the seconds-valued ones, the masks' frame;
  turn length and overlap share are reported only); overlap-style records (per-channel activity median >= 0.85,
  overlap-share median >= 0.7); 50/50 by style in every split and every train tar (+-1); dev speakers disjoint from
  train speakers; top-10 speaker share of the train records' speaker positions <= 0.073; record counts, hours and
  utterances by number of uses (reported). Writes WORK/qc/S1.json.
* headers / finalize (producer output): common/shards_qc.py against the source manifest; finalize makes train/ and
  valid/ read-only and writes N_SHARDS.txt.
* audio (producer output): level, silence share, 99% bandwidth and the zero-run profile per target channel of the
  first record of randomly chosen shards, per style too; with --exports N, the producer's noisy mixture, both targets
  and the clean 24 kHz tracks of N records (half per style) as WAV files.

  python sim/qc.py --work SIM_WORK s0 --processes 64
  python sim/qc.py --work SIM_WORK s1
  python sim/qc.py --work SIM_WORK --shards SHARDS/sim_24k_120s finalize
  python sim/qc.py --work SIM_WORK --shards SHARDS/sim_24k_120s audio --out SIM_WORK/qc/audio
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import base as sc  # noqa: E402
from sim import fit_turns  # noqa: E402
from sim import simulate as sm  # noqa: E402
from sim.base import cm  # noqa: E402
from common import shards_qc as sq  # noqa: E402

S0_COLUMNS = ["key", "split", "n", "style", "ok", "activity_a", "activity_b", "activity_union", "overlap_share",
              "interjections", "reasons"]


def load_plan(work: Path) -> tuple[list[dict], str]:
    """The plan rows and hash; the stage then records and checks the reuse cap the plan was made with."""
    clean = work / "clean"
    if not (clean / "plan.json").exists():
        raise SystemExit(f"No plan in {clean}")
    record = cm.load_json(clean / "plan.json")
    digest = record["plan_hash"]
    if cm.file_sha256(clean / "plan.tsv") != digest:
        raise SystemExit(f"{clean / 'plan.tsv'} does not match plan.json")
    sc.PARAMS["reuse"]["max_uses"] = sm.plan_max_uses(record)
    return sm.read_plan(clean / "plan.tsv"), digest


# ---- S0 -----------------------------------------------------------------------------------------------------------

def s0_record(job) -> str:
    import soundfile as sf
    row, clean, out, tag = job
    target = out / "s0" / row["split"] / f"{row['key']}.json"
    if target.exists() and cm.load_json(target).get("tag") == tag:
        return "skipped"
    reasons = []
    if not sc.KEY_RE.fullmatch(row["key"]):
        reasons.append("key pattern")
    elif sc.parse_key(row["key"]) != row["n"]:
        reasons.append("key digits differ from n")
    if row["style"] != sc.style_of(row["n"]):
        reasons.append("style is not the parity of n")
    masks = {}
    for c in ("a", "b"):
        path = clean / sc.track_relpath(row["split"], row["tar"], row["key"], c)
        problem = sc.clean_problem(path)
        if problem:
            reasons.append(f"{c}: {problem}")
            continue
        x, _ = sf.read(str(path), dtype="int16")
        if not x.any():
            reasons.append(f"{c}: all zero")
        masks[c] = sc.vad_mask(x)
    result = {"key": row["key"], "split": row["split"], "n": row["n"], "style": row["style"], "tag": tag}
    if len(masks) == 2:
        act = sc.activity(masks["a"], masks["b"])
        minimum = sc.PARAMS["record"]["min_activity"]
        for c in ("a", "b"):
            if act[f"activity_{c}"] < minimum:
                reasons.append(f"{c}: {act[f'activity_{c}']:.3f} active (< {minimum})")
        silent = [i for i, ok in enumerate(sc.quarters_active(masks["a"] | masks["b"])) if not ok]
        if silent:
            reasons.append(f"no speech in 30 s quarter(s) {silent}")
        result.update(act)
        result.update(sc.segment(masks["a"], masks["b"]))
    result["reasons"] = reasons
    result["ok"] = not reasons
    cm.write_json(target, result)
    return "done"


def _safe(function, job):
    try:
        return function(job)
    except Exception as error:
        return f"failed: {type(error).__name__}: {error}"


def _safe_s0(job):
    return _safe(s0_record, job)


def reuse_problems(rows: list[dict], max_uses: int) -> list[str]:
    """Utterances whose uses break the reuse cap: more than max_uses in all or ceil(max_uses / 2) in one style,
    twice in one record, or twice with the same partner. With max_uses 1: every utterance used twice."""
    per_style = -(-max_uses // 2)
    total, styles, partners, problems = collections.Counter(), collections.Counter(), collections.defaultdict(list), set()
    for r in rows:
        for c, other in (("a", "b"), ("b", "a")):
            ids = [u for u in r[f"utterances_{c}"].split(",") if u]
            problems.update(u for u, k in collections.Counter(ids).items() if k > 1)
            for u in set(ids):
                total[u] += 1
                styles[(u, r["style"])] += 1
                partners[u].append(r[f"speaker_{other}"])
    problems.update(u for u, k in total.items() if k > max_uses)
    problems.update(u for (u, _), k in styles.items() if k > per_style)
    problems.update(u for u, p in partners.items() if len(p) != len(set(p)))
    return sorted(problems)


def uses_histogram(rows: list[dict]) -> dict[str, int]:
    """Number of utterances by how many records use them."""
    total = collections.Counter(u for r in rows for c in ("a", "b") for u in set(r[f"utterances_{c}"].split(",")) if u)
    counts = collections.Counter(total.values())
    return {str(k): counts[k] for k in sorted(counts)}


def s0(work: Path, processes: int) -> int:
    rows, digest = load_plan(work)
    out = work / "qc"
    tag = cm.params_hash({"params": sc.PARAMS, "plan_hash": digest})
    jobs = [(r, work / "clean", out, tag) for r in rows]
    status, failed = collections.Counter(), []
    with Pool(processes) as pool:
        for job, result in zip(jobs, pool.imap(_safe_s0, jobs, chunksize=4)):
            status[result.split(":")[0]] += 1
            if result.startswith("failed"):
                failed.append((job[0]["key"], result))
    records = [cm.load_json(out / "s0" / r["split"] / f"{r['key']}.json") for r in rows
               if (out / "s0" / r["split"] / f"{r['key']}.json").exists()]
    cm.write_tsv(out / "s0.tsv", [{**r, "reasons": "; ".join(r["reasons"]) or "NA"} for r in records], S0_COLUMNS)
    duplicates = [k for k, n in collections.Counter(r["key"] for r in rows).items() if n > 1]
    max_uses = sc.PARAMS["reuse"]["max_uses"]  # the plan's (load_plan)
    reused = reuse_problems(rows, max_uses)
    bad = [{"key": r["key"], "reasons": r["reasons"]} for r in records if not r["ok"]]
    passed = not failed and len(records) == len(rows) and not bad and not duplicates and not reused
    cm.write_json(out / "S0.json", cm.run_info("S0", sc.PARAMS, plan_hash=digest, records=len(rows),
                                               checked=len(records), failed_records=len(bad), failures=bad[:100],
                                               worker_failures=failed[:50], duplicate_keys=duplicates[:20],
                                               max_uses=max_uses, reuse_problems=len(reused),
                                               reuse_examples=reused[:20], status=dict(status), **{"pass": passed}))
    print(json.dumps({"records": len(rows), "failed_records": len(bad), "worker_failures": len(failed),
                      "duplicate_keys": len(duplicates), "max_uses": max_uses, "reuse_problems": len(reused),
                      "pass": passed}), flush=True)
    return 0 if passed else 1


# ---- S1 -----------------------------------------------------------------------------------------------------------

def balance(rows: list[dict]) -> dict:
    styles = collections.Counter(r["style"] for r in rows)
    return {"overlap": styles["overlap"], "conversation": styles["conversation"],
            "pass": abs(styles["overlap"] - styles["conversation"]) <= 1}


def top_share(rows: list[dict], top: int = 10) -> float:
    positions = collections.Counter(r[f"speaker_{c}"] for r in rows for c in ("a", "b"))
    return sum(n for _, n in positions.most_common(top)) / max(1, sum(positions.values()))


def s1_checks(rows: list[dict], records: dict[str, dict], fit: dict) -> dict:
    """The S1 checks (pure). records: S0 rows by key."""
    cfg = sc.PARAMS["s1"]
    checks = {}
    conversation = [records[r["key"]] for r in rows if r["style"] == "conversation"]
    overlap = [records[r["key"]] for r in rows if r["style"] == "overlap"]
    measured = {"activity_union": [r["activity_union"] for r in conversation],
                "overlap_share": [r["overlap_share"] for r in conversation],
                "fto_s": [v for r in conversation for v in r["fto_s"]],
                "turn_s": [v for r in conversation for v in r["turn_s"]]}
    for name, values in measured.items():
        sim, ref = sc.pct(values), {p: fit["summary"][name][p] for p in ("p10", "p50", "p90")}
        floor = cfg["frame_floor_s"] if name.endswith("_s") else 0.0
        checks[f"conversation_{name}"] = {"sim": sim, "fit": ref, "n": len(values),
                                          "pass": sc.within(sim["p50"], ref["p50"], cfg["tolerance"], floor),
                                          "report_only": name in cfg.get("report_only", ())}
    channel = [r[f"activity_{c}"] for r in overlap for c in ("a", "b")]
    share = [r["overlap_share"] for r in overlap if r["overlap_share"] is not None]
    checks["overlap_channel_activity"] = {**sc.pct(channel), "min_median": cfg["overlap_activity_min"],
                                         "pass": bool(channel) and float(np.median(channel)) >= cfg["overlap_activity_min"]}
    checks["overlap_share"] = {**sc.pct(share), "min_median": cfg["overlap_share_min"],
                               "pass": bool(share) and float(np.median(share)) >= cfg["overlap_share_min"]}
    by_split = collections.defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    checks["styles_per_split"] = {s: balance(v) for s, v in sorted(by_split.items())}
    checks["styles_per_split"]["pass"] = all(v["pass"] for v in checks["styles_per_split"].values())
    tars = collections.defaultdict(list)
    for r in by_split["train"]:
        tars[r["tar"]].append(r)
    unbalanced = {t: balance(v) for t, v in sorted(tars.items()) if not balance(v)["pass"]}
    checks["styles_per_train_tar"] = {"tars": len(tars), "unbalanced": unbalanced, "pass": not unbalanced}
    speakers = {s: {r[f"speaker_{c}"] for r in v for c in ("a", "b")} for s, v in by_split.items()}
    shared = sorted(speakers.get("dev", set()) & speakers.get("train", set()))
    checks["dev_speakers_disjoint"] = {"dev_speakers": len(speakers.get("dev", ())), "shared": shared,
                                       "pass": not shared}
    share10 = top_share(by_split["train"])
    checks["top10_speaker_share"] = {"train": share10, "max": cfg["top10_share_max"],
                                     "pass": share10 <= cfg["top10_share_max"]}
    checks["counts"] = {s: {"records": len(v), "hours": len(v) * sc.TRACK_S / 3600,
                            "utterances_by_uses": uses_histogram(v)} for s, v in by_split.items()}
    checks["counts"]["pass"] = bool(by_split["train"]) and bool(by_split["dev"])
    return checks


def s1(work: Path, fit_path: Path) -> int:
    rows, digest = load_plan(work)
    s0_record_path = work / "qc" / "S0.json"
    if not s0_record_path.exists() or not cm.load_json(s0_record_path)["pass"] \
            or cm.load_json(s0_record_path)["plan_hash"] != digest:
        raise SystemExit(f"S0 has not passed for plan {digest[:12]} ({s0_record_path})")
    records = {r["key"]: cm.load_json(work / "qc" / "s0" / r["split"] / f"{r['key']}.json") for r in rows}
    checks = s1_checks(rows, records, fit_turns.load_fit(fit_path))
    passed = all(v["pass"] for v in checks.values() if not v.get("report_only"))
    cm.write_json(work / "qc" / "S1.json", cm.run_info("S1", sc.PARAMS, plan_hash=digest,
                                                       fit={"path": str(fit_path), "sha256": cm.file_sha256(fit_path)},
                                                       checks=checks, **{"pass": passed}))
    print(json.dumps({k: v["pass"] for k, v in checks.items()} | {"pass": passed}), flush=True)
    return 0 if passed else 1


def s1_status(work: Path, accept: bool = False) -> dict:
    """S1's record for the current plan; SystemExit unless it passed or `accept` is set."""
    _, digest = load_plan(work)
    path = work / "qc" / "S1.json"
    if not path.exists():
        raise SystemExit(f"S1 has not run ({path})")
    record = cm.load_json(path)
    if record["plan_hash"] != digest:
        raise SystemExit(f"{path} is for another plan")
    if not record["pass"] and not accept:
        failed = [k for k, v in record["checks"].items() if not v["pass"]]
        raise SystemExit(f"S1 failed ({failed}); stop, or proceed anyway with --accept-s1-stop")
    return {"pass": record["pass"], "accepted": None if record["pass"] else True}


def require_gates(work: Path, accept: bool = False) -> tuple[list[dict], str, dict]:
    """The plan rows and hash, once S0 passed and S1 passed (or `accept`)."""
    rows, digest = load_plan(work)
    record = work / "qc" / "S0.json"
    if not record.exists() or not cm.load_json(record)["pass"] or cm.load_json(record)["plan_hash"] != digest:
        raise SystemExit(f"S0 has not passed for plan {digest[:12]} ({record})")
    return rows, digest, s1_status(work, accept)


# ---- producer output ----------------------------------------------------------------------------------------------

def record_style(record: dict) -> str:
    return sc.style_of(sc.parse_key(record["names.pickle"][0]))


def audio(out: Path, shards: Path, source: Path, work: Path, count: int, exports: int, seed: int) -> int:
    """Level, silence, bandwidth and zero-run statistics of the producer's targets, overall and per style, and
    optional WAV exports."""
    import soundfile as sf
    rng = random.Random(seed)
    manifest = {r["key"]: r for r in cm.read_tsv(source / "manifest.tsv")}
    tars = sorted((shards / "train").rglob("*.tar"))
    order = rng.sample(tars, len(tars))
    rows = []
    for path in order[:count]:
        record = sq.first_record(path)
        wav = record["input_wav.pth"][0].numpy()
        rows.append({"shard": str(path), "names": record["names.pickle"], "style": record_style(record),
                     **{f"ch{i}": sq.channel_stats(wav[i]) for i in range(2)}})
    report = {"records": rows, "summary": sq.summarize(rows)}
    for style in sc.STYLES:
        part = [r for r in rows if r["style"] == style]
        report[f"summary_{style}"] = {"records": len(part), **(sq.summarize(part) if part else {})}
    want = {style: exports // 2 for style in sc.STYLES}
    want[sc.STYLES[0]] += exports % 2
    exported = []
    if exports:
        (out / "exports").mkdir(parents=True, exist_ok=True)
    for path in order:  # sampled shards first, then the rest until both styles are filled
        if not any(want.values()):
            break
        record = sq.first_record(path)
        style = record_style(record)
        if not want[style]:
            continue
        want[style] -= 1
        key = record["names.pickle"][0][:-2]
        stem = f"{len(exported):02d}_{key}"
        wav = record["input_wav.pth"][0].numpy()
        for i in range(2):
            sf.write(str(out / "exports" / f"{stem}_target{i}.wav"), wav[i], 24_000, subtype="PCM_16")
        sf.write(str(out / "exports" / f"{stem}_noisy_mixture.wav"), record["noisy_mixture.pth"][0].numpy(), 24_000,
                 subtype="PCM_16")
        row = manifest[key]
        for c in ("a", "b"):
            clean, sr = sf.read(str(work / "clean" / sc.track_relpath(row["split"], int(row["tar"]), key, c)),
                                dtype="int16")
            sf.write(str(out / "exports" / f"{stem}_clean_{c}.wav"), clean, sr, subtype="PCM_16")
        exported.append({"key": key, "stem": stem, "style": style, "shard": str(path)})
    cm.write_json(out / "report.json", cm.run_info("audio", {"count": count, "exports": exports, "seed": seed},
                                                   report=report, exported=exported, missing_exports=want))
    print(json.dumps({k: v for k, v in report.items() if k.startswith("summary")}, indent=1, default=str), flush=True)
    return 0 if not any(want.values()) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, required=True, help="the simulation's work directory (SIM_WORK)")
    parser.add_argument("--shards", type=Path, default=None, help="producer output (default: WORK/shards/sim_24k_120s)")
    parser.add_argument("--source", type=Path, default=None, help="source tars (default: WORK/source)")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("s0")
    s.add_argument("--processes", type=int, default=os.cpu_count())
    s1_parser = sub.add_parser("s1")
    s1_parser.add_argument("--fit", type=Path, default=fit_turns.SHIPPED)
    for name in ("headers", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--max-shortfall", type=int, default=0,
                       help="windows a producer task may lack (the producer drops all-quiet chunks and NaN records)")
    a = sub.add_parser("audio")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--count", type=int, default=300)
    a.add_argument("--exports", type=int, default=20)
    a.add_argument("--seed", type=int, default=sc.SEED)
    args = parser.parse_args(argv)
    shards = args.shards or args.work / "shards" / sc.WRITER
    source = args.source or args.work / "source"
    if args.command == "s0":
        return s0(args.work, args.processes)
    if args.command == "s1":
        return s1(args.work, args.fit)
    if args.command == "headers":
        report = sq.headers(shards, source if (source / "manifest.tsv").exists() else None,
                            max_shortfall=args.max_shortfall)
        cm.write_json(shards / "qc_headers.json", cm.run_info("headers", {}, report=report))
        print(json.dumps({k: report.get(k) for k in ("pass", "train_shards", "train_records", "task_mismatches",
                                                      "task_shortfalls", "valid_records", "duplicate_keys")}),
              flush=True)
        return 0 if report["pass"] else 1
    if args.command == "finalize":
        return sq.finalize(shards, source, count_file="N_SHARDS.txt", max_shortfall=args.max_shortfall,
                           params={"tasks": sq.TASKS, "corpus": sc.NAME, "max_shortfall": args.max_shortfall})
    return audio(args.out, shards, source, args.work, args.count, args.exports, args.seed)


if __name__ == "__main__":
    sys.exit(main())
