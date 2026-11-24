"""S1 of the Basis pipeline: bleed and integrity screen of the raw tracks against every other track of the segment.

Signal path (common/crosstalk.py): 60 s decode blocks, exact-zero runs >= 20 ms and clipped runs (>= 3 samples with
|x| >= 32,700), resample_poly(1, 3) to 16 kHz, 150-3,800 Hz band-pass, 100 fps energies floored at -110 dB, lags
-3.5..+3.5 s in 10 ms steps, 10-minute blocks. Statistics: crosstalk.analyse for N tracks, per ordered pair (x, y)
with >= 60 s of common on-call time a lag search, third parties removed from the per-pair frames, a pooled statistic
over every other track, window statistics against every y with >= 5 s of A_y in the window and pooled.

Commands (resumable; finished units are skipped):

* features: per track, every track of each ready segment that has an S0-kept window or a control, longest first:
  features/<release path>.npz (energy float32 at 100 fps, zero runs, clip runs).
* run: per ready segment once all its tracks have features: frames/<segment>.npz (rho per pair, O and A masks),
  rows/<segment>.json (written last); then tracks.jsonl, pairs.jsonl and windows.tsv.
* clip-table (no decoding): the clipped-sample share of every screened track-window (s1_clip_windows.tsv), which the
  clip-share rule (window_clip_share_max) needs.
* rescore (tables only): the decisions under a thresholds file (default: basis/thresholds.json), written to
  s1_windows_decided_<tag>.tsv with counts in s1_rescore_<tag>.json. cut.py applies the same decisions itself; this
  command only reports them.

  python scripts/dialoguesidon2/data/basis/screen.py --root BASIS_ROOT --screen WORK/screen features --processes 64
  python scripts/dialoguesidon2/data/basis/screen.py --root BASIS_ROOT --screen WORK/screen run --processes 64
  python scripts/dialoguesidon2/data/basis/screen.py --root BASIS_ROOT --screen WORK/screen clip-table
  python scripts/dialoguesidon2/data/basis/screen.py --root BASIS_ROOT --screen WORK/screen rescore
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
from basis import corpus as bs  # noqa: E402
from basis import tables as tb  # noqa: E402
from common import core as cm  # noqa: E402
from common import crosstalk as xt  # noqa: E402
from common import decisions as dd  # noqa: E402

THRESHOLDS = Path(__file__).resolve().parent / "thresholds.json"

# The signal path and statistics of the OpenYap screen (without its coherence diagnostic), plus the N-track
# settings.
PARAMS = {
    "version": 2,
    "decode_block_samples": 2_880_000,
    "resample": {"up": 1, "down": 3, "context_samples": 90},
    "bandpass_hz": [150.0, 3800.0], "bandpass_order": 4,
    "frame": {"window_16k": 320, "hop_16k": 160},
    "floor_db": -110.0,
    "words": {"pad_s": 0.1, "gap_s": 0.3, "echo_words_removed": True, "echo_window_s": list(bs.ECHO_WINDOW_S),
              "normaliser": "basis (NFKC, casefold, L/M/N)"},
    "guard_frames": 40,
    "lag_frames": [-350, 350],
    "lag_block_frames": 60_000, "min_block_frames": 3_000,
    "prominence_band_frames": [25, 60],
    "floor_percentile": 10,
    "min_level_frames": 500, "min_own_frames": 300,
    "zero_run_samples": 960, "clip_level": 32700, "clip_run_samples": 3, "interior_margin_s": 0.03,
    "n_track": {"pair_min_frames": 6_000, "window_min_frames": 500, "lag_fallback": "lower median, else 0",
                "pooled": True, "zero_frames": "480-sample hops inside an exact-zero run"},
}
# Defaults under the thresholds file: the OpenYap screen's bleed and integrity thresholds plus the N-track keys.
# dnsmos_track_min / dnsmos_window_min set to null switch that DNSMOS rule off. The released data used
# basis/thresholds.json, which sets every key.
DEFAULT_THRESHOLDS = {
    "track_prominence": 0.045, "track_blr_db": -42.0, "track_exc_db": 6.0,
    "window_blr_db": -35.0, "window_exc_db": 3.0, "window_pooled": True,
    "window_zero_runs_max": 0, "window_zero_runs_gated_max": 1_000_000_000, "window_clip_runs_max": 300,
    "gate_runs_per_hour_min": 100.0,
    "dnsmos_track_min": 2.5, "dnsmos_window_min": 2.3, "dnsmos_clip_coverage": 0.25, "dnsmos_min_clips": 3,
    "floor_max_db": None,
    "s2_window_blr_db": -30.0, "s2_window_exc_db": 3.0,
}
SEED = bs.SEED
WINDOW_COLUMNS = ["seg", "offset_s", "x", "y", "lag_conv", "blr_conv", "exc_conv", "np_conv", "lag_blk", "blr_blk",
                  "exc_blk", "np_blk", "zr", "clip", "zero_s"]
DECIDED_COLUMNS = ["segment_id", "offset_s", "spk0", "spk1", "s0_keep", "screened", "track_reject", "window_bleed",
                   "integrity", "dnsmos", "fallback", "keep", "n_other_on_call"]
CLIP_TABLE = "s1_clip_windows.tsv"
CLIP_COLUMNS = ["seg", "x", "offset_s", "clip_runs", "clip_samples", "clip_share", "clip_max"]


def params_tag() -> str:
    return cm.params_hash(PARAMS)


def feature_path(screen: Path, rel: str) -> Path:
    return screen / "features" / (rel[:-len(".flac")] + ".npz")


# ---- phase A: features per track -------------------------------------------------------------------------------

def track_features(job) -> str:
    root, screen, rel, samples, tag = job
    target = feature_path(screen, rel)
    if target.exists():
        with np.load(target) as data:
            if str(data["params_hash"]) == tag:
                return "skipped"
    info = cm.read_streaminfo(root / rel)
    if (info["sr"], info["ch"], info["bps"], info["ns"]) != (cm.SR, 1, 16, samples):
        raise ValueError(f"{rel}: FLAC header {info}, expected 48 kHz mono 16-bit with {samples} samples")
    features = xt.stream_features(xt.int16_blocks(root / rel, PARAMS["decode_block_samples"]), samples, PARAMS)
    cm.write_atomic(target, cm.npz_bytes({"e": features["energy"].astype(np.float32),
                                          "zero_runs": features["zero_runs"], "clip_runs": features["clip_runs"],
                                          "samples": np.array(samples), "frames": np.array(features["frames"]),
                                          "params_hash": np.array(tag)}))
    return "done"


def load_features(screen: Path, rel: str) -> dict:
    with np.load(feature_path(screen, rel)) as data:
        return {"energy": data["e"].astype(np.float64), "zero_runs": data["zero_runs"].reshape(-1, 2),
                "clip_runs": data["clip_runs"].reshape(-1, 2), "samples": int(data["samples"]),
                "frames": int(data["frames"]), "params_hash": str(data["params_hash"])}


def _safe(function, job):
    try:
        return function(job)
    except Exception as error:  # reported per unit; a rerun resumes
        return f"failed: {type(error).__name__}: {error}"


def _safe_features(job):
    return _safe(track_features, job)


# ---- phase B: statistics per segment ---------------------------------------------------------------------------

def analyse_segment(row: dict, segment: dict, task: dict, features: dict) -> dict:
    """crosstalk.analyse over every track of the segment for its tested tracks and their windows."""
    frames = segment["samples"] // cm.HOP
    masks = bs.segment_masks(row, frames, PARAMS["words"]["pad_s"], PARAMS["words"]["gap_s"],
                             tuple(PARAMS["words"]["echo_window_s"]))
    order = masks["speakers"]
    windows = {x: [int(off) // cm.WINDOW_S for off in task["windows"].get(x, {})] for x in task["tested"]}
    result = xt.analyse(segment["segment_id"], {s: features[s] for s in order}, masks["o"], masks["spans"], PARAMS,
                        tested=task["tested"], active=masks["a"], windows=windows,
                        pair_min_frames=PARAMS["n_track"]["pair_min_frames"],
                        window_min_frames=PARAMS["n_track"]["window_min_frames"], lag_fallback=True, pooled=True)
    result["masks"] = masks
    return result


def screen_segment(job) -> str:
    root, screen, segment, task, location, tag = job
    sid = segment["segment_id"]
    rows_path = screen / "rows" / f"{sid}.json"
    if rows_path.exists():
        return "skipped"
    features = {}
    for speaker, rel in segment["tracks"].items():
        features[speaker] = load_features(screen, rel)
        if features[speaker]["params_hash"] != tag or features[speaker]["samples"] != segment["samples"]:
            raise ValueError(f"{rel}: features of other parameters or length")
    row = bs.read_segment(root, *location)
    result = analyse_segment(row, segment, task, features)
    payload = segment_payload(sid, result, tag)
    arrays = {f"rho__{x}__{y}": rho.astype(np.float16) for (x, y), rho in result["rho"].items()}
    for s in result["masks"]["speakers"]:
        arrays[f"o__{s}"] = np.packbits(result["masks"]["o"][s])
        arrays[f"a__{s}"] = np.packbits(result["masks"]["a"][s])
    arrays["frames"] = np.array(result["frames"])
    arrays["lags"] = result["lags"]
    cm.write_atomic(screen / "frames" / f"{sid}.npz", cm.npz_bytes(arrays))
    cm.write_atomic(rows_path, json.dumps(cm.sanitize(payload), default=cm.nan_to_none))
    return "done"


def segment_payload(sid: str, result: dict, tag: str) -> dict:
    integrity = {(r["offset_s"], r["x"]): r for r in result["integrity"]}
    windows = []
    for w in result["windows"]:
        extra = integrity[(w["offset_s"], w["x"])]
        windows.append({"seg": sid, **w, "zr": extra["zr"], "clip": extra["clip"], "zero_s": extra.get("zero_s")})
    tracks = [{"seg": sid, **t} for t in result["tracks"].values()]
    pairs = [{"seg": sid, **p} for p in result["pairs"].values()]
    return {"params_hash": tag, "segment_id": sid, "tracks": tracks, "pairs": pairs, "windows": windows}


def _safe_segment(job):
    return _safe(screen_segment, job)


def collect(screen: Path, segment_ids: list[str], tag: str) -> dict:
    tracks, pairs, windows = [], [], []
    for sid in segment_ids:
        payload = cm.load_json(screen / "rows" / f"{sid}.json")
        if payload["params_hash"] != tag:
            raise SystemExit(f"rows/{sid}.json was written with parameters {payload['params_hash']}, not {tag}")
        tracks.extend(payload["tracks"])
        pairs.extend(payload["pairs"])
        windows.extend(payload["windows"])
    cm.write_atomic(screen / "tracks.jsonl", "".join(json.dumps(t) + "\n" for t in tracks))
    cm.write_atomic(screen / "pairs.jsonl", "".join(json.dumps(p) + "\n" for p in pairs))
    cm.write_tsv(screen / "windows.tsv", windows, WINDOW_COLUMNS)
    return {"tracks": len(tracks), "pairs": len(pairs), "window_rows": len(windows)}


def load_screen(screen: Path) -> dict:
    with open(screen / "tracks.jsonl") as stream:
        tracks = [json.loads(line) for line in stream]
    with open(screen / "pairs.jsonl") as stream:
        pairs = [json.loads(line) for line in stream]
    windows = [{k: cm.parse_cell(v) if k not in ("seg", "x", "y") else v for k, v in row.items()}
               for row in cm.read_tsv(screen / "windows.tsv")]
    return {"tracks": tracks, "pairs": pairs, "windows": windows}


# ---- decisions ---------------------------------------------------------------------------------------------------

def load_quality(screen: Path) -> dict:
    """The DNSMOS summaries (dnsmos_windows.py summarize): {(seg, x): row} and {(seg, x, offset): row}; empty when
    the scores were not computed."""
    tracks, windows = {}, {}
    if (screen / "q_tracks.tsv").exists():
        for row in cm.read_tsv(screen / "q_tracks.tsv"):
            tracks[(row["seg"], row["x"])] = {k: cm.parse_cell(v) for k, v in row.items() if k not in ("seg", "x")}
    if (screen / "q_windows.tsv").exists():
        for row in cm.read_tsv(screen / "q_windows.tsv"):
            windows[(row["seg"], row["x"], int(row["offset_s"]))] = {k: cm.parse_cell(v) for k, v in row.items()
                                                                     if k not in ("seg", "x")}
    return {"tracks": tracks, "windows": windows}


def track_reasons(screen_rows: dict, quality: dict, th: dict) -> dict[tuple[str, str], list[str]]:
    """Per (segment, tested track): why the track is excluded from its segment (bleed, DNSMOS, noise floor)."""
    reasons = collections.defaultdict(list)
    for p in screen_rows["pairs"]:
        if dd.pair_bleeds(p, th):
            reasons[(p["seg"], p["x"])].append(f"bleed:{p['y']}")
    for t in screen_rows["tracks"]:
        key = (t["seg"], t["x"])
        reasons.setdefault(key, [])
        q = quality["tracks"].get(key)
        if th["dnsmos_track_min"] is not None and q is not None and q.get("median") is not None \
                and q["median"] < th["dnsmos_track_min"]:
            reasons[key].append("dnsmos_track")
        floor_max = th.get("floor_max_db")
        if floor_max is not None and t.get("floor") is not None and t["floor"] > float(floor_max):
            reasons[key].append("floor")
    return dict(reasons)


def pairs_by_track(screen_rows: dict) -> dict[tuple[str, str], dict[str, dict]]:
    """x's pair rows by y, per (segment, x)."""
    result = collections.defaultdict(dict)
    for p in screen_rows["pairs"]:
        result[(p["seg"], p["x"])][p["y"]] = p
    return result


def window_bleed(rows: list[dict], pairs: dict, th: dict, offset_s: int, prefix: str = "") -> list[str]:
    """The window bleed rule of x in one window (decisions.window_bleed_rows with the optional lag gate of
    window_lag_s / window_prominence built from x's S1 pair rows); S1 and S2 (prefix 's2_') share it."""
    return dd.window_bleed_rows(rows, th, prefix=prefix, pooled=th.get("window_pooled", True),
                                allowed=dd.window_gate(pairs, th, offset_s))


def load_clip(screen: Path) -> dict:
    """The clip table (`clip-table`): {(seg, x, offset): row} with clip_runs, clip_samples and clip_share."""
    path = screen / CLIP_TABLE
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `screen.py clip-table` first")
    return {(r["seg"], r["x"], int(r["offset_s"])): {k: cm.parse_cell(v) for k, v in r.items()
                                                     if k not in ("seg", "x", "offset_s")}
            for r in cm.read_tsv(path)}


def decide(segments: list[dict], s0_windows: list[dict], screen_rows: dict, quality: dict, th: dict,
           fallback: bool = False, clip: dict | None = None) -> tuple[dict, list[dict]]:
    """S1 decisions (with the optional DNSMOS rules) over S0's windows: the excluded tracks and one row per S0
    window. `clip` (load_clip) is needed when the thresholds set window_clip_share_max."""
    excluded = track_reasons(screen_rows, quality, th)
    tracks = {(t["seg"], t["x"]): t for t in screen_rows["tracks"]}
    pairs = pairs_by_track(screen_rows)
    rows_by = collections.defaultdict(list)
    for w in screen_rows["windows"]:
        rows_by[(w["seg"], int(w["offset_s"]), w["x"])].append(w)
    decided = []
    for w0 in s0_windows:
        row = {"segment_id": w0["segment_id"], "offset_s": int(w0["offset_s"]), "spk0": w0.get("spk0"),
               "spk1": w0.get("spk1"), "s0_keep": bool(w0["keep"]), "n_other_on_call": w0.get("n_other_on_call")}
        if not row["s0_keep"]:
            row.update(screened=False, track_reject="", window_bleed="", integrity="", dnsmos="", fallback=False,
                       keep=False)
            decided.append(row)
            continue
        pair = (w0["spk0"], w0["spk1"])
        screened = all((w0["segment_id"], x) in tracks and rows_by.get((w0["segment_id"], row["offset_s"], x))
                       for x in pair)
        rejected, bleed, integrity, dnsmos = [], [], [], []
        if screened:
            for x in pair:
                key = (w0["segment_id"], x)
                rejected += [f"{x}:{r}" for r in excluded.get(key, [])]
                rows = rows_by[(w0["segment_id"], row["offset_s"], x)]
                bleed += [f"{x}:{y}" for y in window_bleed(rows, pairs.get(key, {}), th, row["offset_s"])]
                zr = max((r.get("zr") or 0) for r in rows)
                clip_runs = max((r.get("clip") or 0) for r in rows)
                share = None
                if th.get("window_clip_share_max") is not None:
                    found = (clip or {}).get((w0["segment_id"], x, row["offset_s"]))
                    if found is None:
                        raise ValueError(f"No clip-table row for {w0['segment_id']} {x} {row['offset_s']}")
                    share = found["clip_share"]
                integrity += [f"{x}:{r}" for r in dd.integrity_fails(zr, clip_runs, dd.gated(tracks[key], th), th,
                                                                     share)]
                q = quality["windows"].get((w0["segment_id"], x, row["offset_s"]))
                if th["dnsmos_window_min"] is not None and q is not None and q.get("median") is not None \
                        and (q.get("scored") or 0) >= th["dnsmos_min_clips"] and q["median"] < th["dnsmos_window_min"]:
                    dnsmos.append(x)
        other = bool(fallback and (w0.get("n_other_on_call") or 0) > 0)
        row.update(screened=screened, track_reject=",".join(rejected), window_bleed=",".join(bleed),
                   integrity=",".join(integrity), dnsmos=",".join(dnsmos), fallback=other)
        row["keep"] = bool(screened and not rejected and not bleed and not integrity and not dnsmos and not other)
        decided.append(row)
    return excluded, decided


def rule_attribution(rows: list[dict]) -> dict:
    """Per rule, the screened S0-kept windows it rejects (whatever else rejects them) and those it alone rejects."""
    def rules(r):
        found = set()
        for part in filter(None, (r["track_reject"] or "").split(",")):
            reason = part.split(":")[1]
            found.add({"bleed": "track_bleed", "dnsmos_track": "track_dnsmos"}.get(reason, f"track_{reason}"))
        if r["window_bleed"]:
            found.add("window_bleed")
        for part in filter(None, (r["integrity"] or "").split(",")):
            found.add(part.split(":")[1])
        if r["dnsmos"]:
            found.add("window_dnsmos")
        if r["fallback"]:
            found.add("fallback")
        return found
    any_rule, alone = collections.Counter(), collections.Counter()
    for r in rows:
        if not r["screened"]:
            continue
        found = rules(r)
        any_rule.update(found)
        if len(found) == 1:
            alone.update(found)
    return {"any": dict(sorted(any_rule.items())), "alone": dict(sorted(alone.items()))}


def rescore(screen: Path, decisions: dict, tag: str) -> dict:
    """Decisions over S0's windows under `decisions` (a thresholds file), written to s1_windows_decided_<tag>.tsv,
    with counts and the excluded tracks in s1_rescore_<tag>.json."""
    th = {**DEFAULT_THRESHOLDS, **decisions.get("thresholds", {})}
    fallback = bool(decisions.get("fallback_no_excluded_on_call", False))
    segments = tb.load_segments(screen)
    s0_windows = tb.load_windows(screen)
    screen_rows = load_screen(screen)
    quality = load_quality(screen)
    clip = load_clip(screen) if th.get("window_clip_share_max") is not None else None
    excluded, decided = decide(segments, s0_windows, screen_rows, quality, th, fallback, clip)
    lag_gated = th.get("track_lag_s") is not None
    kept = [r for r in decided if r["keep"]]
    s0_kept = [r for r in decided if r["s0_keep"]]
    lang = {s["segment_id"]: s["lang"] for s in segments}
    counts = {
        "s0_windows_kept": len(s0_kept), "screened": sum(r["screened"] for r in s0_kept),
        "after_track_rule": sum(r["screened"] and not r["track_reject"] for r in s0_kept),
        "after_window_bleed": sum(r["screened"] and not r["track_reject"] and not r["window_bleed"] for r in s0_kept),
        "after_integrity": sum(r["screened"] and not r["track_reject"] and not r["window_bleed"] and not r["integrity"]
                               for r in s0_kept),
        "after_dnsmos": sum(r["screened"] and not r["track_reject"] and not r["window_bleed"] and not r["integrity"]
                            and not r["dnsmos"] for r in s0_kept),
        "windows_kept": len(kept), "hours_kept": round(len(kept) * cm.WINDOW_S / 3600, 1),
        "kept_other_on_call": sum((r["n_other_on_call"] or 0) > 0 for r in kept),
        "tracks_excluded": sum(1 for v in excluded.values() if v),
        "tracks_excluded_by_reason": dd.counter(r.split(":")[0] for v in excluded.values() for r in v),
        "kept_by_language": dict(collections.Counter(lang[r["segment_id"]] for r in kept).most_common()),
        "quality_present": bool(quality["tracks"]),
    }
    if lag_gated:
        counts["windows_failing_each_rule"] = rule_attribution(s0_kept)
        counts["hours_kept_by_language"] = {k: round(v * cm.WINDOW_S / 3600, 1)
                                            for k, v in counts["kept_by_language"].items()}
    cm.write_tsv(screen / f"s1_windows_decided_{tag}.tsv", decided, DECIDED_COLUMNS)
    report = cm.run_info("S1-rescore", PARAMS, thresholds=th, fallback_no_excluded_on_call=fallback, counts=counts,
                         tracks_excluded={f"{g}/{x}": v for (g, x), v in sorted(excluded.items()) if v})
    cm.write_json(screen / f"s1_rescore_{tag}.json", cm.sanitize(report))
    return {"counts": counts}


def plan_units(root: Path, screen: Path) -> tuple[list[dict], dict, set[str]]:
    segments = tb.load_segments(screen)
    tasks = tb.segment_tasks(segments, tb.load_windows(screen))
    ready = tb.ready_segments(screen)
    return segments, tasks, ready


def command_features(args) -> int:
    segments, tasks, ready = plan_units(args.root, args.screen)
    tag = params_tag()
    jobs = []
    for s in segments:
        if s["segment_id"] in tasks and s["segment_id"] in ready:
            for rel in s["tracks"].values():
                jobs.append((args.root, args.screen, rel, s["samples"], tag))
    jobs.sort(key=lambda j: (-j[3], j[2]))
    status, failures = collections.Counter(), []
    cm.write_json(args.screen / "s1_features_start.json",
                  cm.run_info("S1-features", PARAMS, params_hash=tag, tracks=len(jobs),
                              segments_planned=len(tasks), segments_ready=len(set(tasks) & ready)))
    with Pool(args.processes, maxtasksperchild=16) as pool:
        for job, result in zip(jobs, pool.imap(_safe_features, jobs)):
            status[result.split(":")[0]] += 1
            if result.startswith("failed"):
                failures.append((job[2], result))
                print(f"{job[2]}: {result}", flush=True)
            if sum(status.values()) % 500 == 0:
                print(f"features {sum(status.values())}/{len(jobs)} {dict(status)}", flush=True)
    cm.write_json(args.screen / "s1_features_failures.json", failures)
    print(json.dumps({"features": dict(status), "failures": len(failures)}), flush=True)
    return 1 if failures else 0


def command_run(args) -> int:
    segments, tasks, ready = plan_units(args.root, args.screen)
    tag = params_tag()
    index = bs.segment_index(args.root)
    jobs, waiting = [], []
    for s in segments:
        sid = s["segment_id"]
        if sid not in tasks or sid not in ready:
            continue
        if not all(feature_path(args.screen, rel).exists() for rel in s["tracks"].values()):
            waiting.append(sid)
            continue
        jobs.append((args.root, args.screen, s, tasks[sid], index[sid], tag))
    jobs.sort(key=lambda j: -j[2]["samples"] * len(j[2]["tracks"]) * max(1, len(j[3]["tested"])))
    (args.screen / "rows").mkdir(parents=True, exist_ok=True)
    stale = [j[2]["segment_id"] for j in jobs if (args.screen / "rows" / f"{j[2]['segment_id']}.json").exists()
             and cm.load_json(args.screen / "rows" / f"{j[2]['segment_id']}.json")["params_hash"] != tag]
    if stale:
        raise SystemExit(f"{len(stale)} segments were screened with other parameters (e.g. {stale[0]}); use a new --screen")
    cm.write_json(args.screen / "s1_run_start.json", cm.run_info("S1", PARAMS, params_hash=tag, segments=len(jobs),
                                                                 waiting_for_features=len(waiting)))
    status, failures = collections.Counter(), []
    with Pool(args.processes, maxtasksperchild=4) as pool:
        for job, result in zip(jobs, pool.imap(_safe_segment, jobs)):
            status[result.split(":")[0]] += 1
            if result.startswith("failed"):
                failures.append((job[2]["segment_id"], result))
                print(f"{job[2]['segment_id']}: {result}", flush=True)
            if sum(status.values()) % 50 == 0:
                print(f"S1 {sum(status.values())}/{len(jobs)} {dict(status)}", flush=True)
    cm.write_json(args.screen / "s1_failures.json", failures)
    done = sorted(j[2]["segment_id"] for j in jobs if (args.screen / "rows" / f"{j[2]['segment_id']}.json").exists())
    counts = collect(args.screen, done, tag)
    record = cm.run_info("S1", PARAMS, params_hash=tag, status=dict(status), failures=len(failures),
                         segments_planned=len(tasks), segments_screened=len(done), waiting_for_features=waiting,
                         not_ready=sorted(set(tasks) - ready), **counts)
    cm.write_json(args.screen / "s1_run.json", record)
    print(json.dumps({k: record[k] for k in ("status", "failures", "segments_planned", "segments_screened")}), flush=True)
    return 1 if failures else 0


def clip_rows(job) -> list[dict]:
    """Clip statistics of one segment's tested track-windows from the screen's run lists (features/*.npz clip_runs,
    the A masks of frames/<segment>.npz; nothing is decoded): runs starting in A_x inside the window, as windows.tsv's
    `clip`, and the clipped samples they hold."""
    screen, segment, windows = job
    sid = segment["segment_id"]
    frames = segment["samples"] // cm.HOP
    with np.load(screen / "frames" / f"{sid}.npz") as data:
        active = {x: np.unpackbits(data[f"a__{x}"])[:frames].astype(bool) for x in windows}
    rows = []
    for x, offsets in sorted(windows.items()):
        with np.load(feature_path(screen, segment["tracks"][x])) as data:
            runs = xt.runs_in(data["clip_runs"].reshape(-1, 2), active[x])
        for offset in sorted(offsets):
            lo, hi = offset * cm.SR, (offset + cm.WINDOW_S) * cm.SR
            inside = runs[(runs[:, 0] >= lo) & (runs[:, 0] < hi)] if len(runs) else runs
            samples = int(inside[:, 1].sum()) if len(inside) else 0
            rows.append({"seg": sid, "x": x, "offset_s": offset, "clip_runs": int(len(inside)),
                         "clip_samples": samples, "clip_share": samples / cm.WINDOW_SAMPLES,
                         "clip_max": int(inside[:, 1].max()) if len(inside) else 0})
    return rows


def command_clip_table(args) -> int:
    """The clip table (CLIP_TABLE) of every S1 window row, for the clip-share rule (window_clip_share_max)."""
    target = args.screen / CLIP_TABLE
    if target.exists():
        print(f"{target} exists; nothing to do", flush=True)
        return 0
    segments = {s["segment_id"]: s for s in tb.load_segments(args.screen)}
    windows = collections.defaultdict(lambda: collections.defaultdict(set))
    runs_by = {}
    for w in load_screen(args.screen)["windows"]:
        windows[w["seg"]][w["x"]].add(int(w["offset_s"]))
        runs_by[(w["seg"], w["x"], int(w["offset_s"]))] = w.get("clip") or 0
    jobs = [(args.screen, segments[sid], dict(per)) for sid, per in sorted(windows.items())]
    table = []
    with Pool(args.processes) as pool:
        for result in pool.imap(clip_rows, jobs, chunksize=4):
            table.extend(result)
    mismatch = sum(1 for r in table if r["clip_runs"] != runs_by[(r["seg"], r["x"], r["offset_s"])])
    if mismatch:
        raise SystemExit(f"{mismatch} track-windows count other clip runs than windows.tsv")
    cm.write_tsv(target, table, CLIP_COLUMNS)
    print(json.dumps({"track_windows": len(table), "segments": len(jobs)}), flush=True)
    return 0


def command_rescore(args) -> int:
    result = rescore(args.screen, cm.load_json(args.decisions), args.tag)
    print(json.dumps(cm.sanitize(result), indent=1), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the downloaded release")
    parser.add_argument("--screen", type=Path, required=True, help="the screen directory (S0 and verification)")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("features")
    f.add_argument("--processes", type=int, default=os.cpu_count())
    r = sub.add_parser("run")
    r.add_argument("--processes", type=int, default=os.cpu_count())
    t = sub.add_parser("clip-table", help="clipped-sample share per screened track-window (no decoding)")
    t.add_argument("--processes", type=int, default=2)
    s = sub.add_parser("rescore")
    s.add_argument("--decisions", type=Path, default=THRESHOLDS, help="thresholds file (default: the shipped one)")
    s.add_argument("--tag", default="released")
    args = parser.parse_args(argv)
    return {"features": command_features, "run": command_run, "clip-table": command_clip_table,
            "rescore": command_rescore}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
