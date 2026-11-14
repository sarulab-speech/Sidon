"""Cut of the Basis pipeline: S2 on the restored pair tracks, the speaker-disjoint dev split, blocks of at most 5
windows, and the source tars of the producer.

Decodes the restored tracks. Refuses when a segment with an S0-kept window is not ready (audio verification),
screened (S1), scored (DNSMOS; only when a DNSMOS rule is set) or restored (output check) unless --allow-missing
lists it, and when the output directory holds tars of another plan. Phases, each resumable:

1. s2: per segment with windows that pass S1 (and per control segment), BLR/EXC of the restored pair track x
   against every other track y at the raw-audio screen's lags (conversation and block) and pooled, from x's
   restored energy and the transcripts (the other tracks are not restored); s2/<segment>.json. Asserts that a
   restored track is mono 48 kHz PCM_16 and exactly as long as the raw track.
2. plan: windows kept = S0 keep, S1 keep (the thresholds file and its fallback switch) and S2 keep (BLR >= -30 dB
   and EXC >= 3 dB against any y or pooled, at either lag, rejects). Stops (exit 3, no tar) when S2 removes more
   than 3% of the windows that passed S1, unless --accept-s2-stop is given. The controls' S2 hit rate on their
   S1-flagged windows is reported; it stops only if the thresholds file sets "s2_controls": "stop". The dev split
   takes whole components of the participant graph (every participant of every segment of a conversation),
   smallest first, until >= 64 windows. Train windows are cut into blocks of <= 5 consecutive offsets per segment,
   shuffled with numpy default_rng(20261013) and given to the train tar with the fewest windows (256 tars). Writes
   manifest.tsv and plan.json.
3. write: one tar per job of the pool (written to <name>.tmp and renamed, with done/<split>-<n>.json). Records:
   `flac` (stereo 48 kHz PCM_16, 5,760,000 frames, channel 0 = restored spk0, channel 1 = restored spk1 in S0's
   channel order, samples copied losslessly) and `original_path.txt`; keys basis-<conversation>-<segment
   start>-<offset> (5 digits each). summary.json gets "status": "complete" when every tar is written.

  python scripts/dialoguesidon2/data/basis/cut.py --root BASIS_ROOT --screen WORK/screen --data-root WORK \
      --processes 64
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
from basis import restore_inputs as ri  # noqa: E402
from basis import screen as sn  # noqa: E402
from basis import tables as tb  # noqa: E402
from common import core as cm  # noqa: E402
from common import crosstalk as xt  # noqa: E402
from common import decisions as dd  # noqa: E402
from common import layout as lo  # noqa: E402

PARAMS = {
    "version": 1, "seed": bs.SEED, "train_tars": 256, "block_max_windows": 5, "dev_min_windows": 64,
    "s2": {"max_removed_share": 0.03, "min_control_hit_rate": 0.80, "min_control_windows": 10,
           "controls_default": "report"},
    "record": {"frames": cm.WINDOW_SAMPLES, "channels": 2, "subtype": "PCM_16", "sample_rate": cm.SR},
}
MANIFEST_COLUMNS = (["key", "segment_id", "conversation_id", "lang", "offset_s", "split", "tar", "block", "position",
                     "spk0", "spk1", "n_eligible", "n_other_on_call", "union", "third_only"]
                    + [f"{f}_{c}" for c in (0, 1) for f in ("bandwidth", "gated", "ovr", "s1_blr_max", "s1_exc_max",
                                                            "s2_blr_max", "s2_exc_max")])


# ---- S2 --------------------------------------------------------------------------------------------------------

def check_restored(path: Path, frames: int) -> None:
    import soundfile as sf
    info = sf.info(str(path))
    if (info.samplerate, info.channels, info.subtype, info.frames) != (cm.SR, 1, "PCM_16", frames):
        raise ValueError(f"{path}: {info.samplerate} Hz, {info.channels} ch, {info.subtype}, {info.frames} frames; "
                         f"expected 48000 Hz mono PCM_16 with the raw length {frames}")


def s2_segment(job) -> str:
    root, restored, out, segment, tested, location, fixed, tag = job
    sid = segment["segment_id"]
    target = out / "s2" / f"{sid}.json"
    if target.exists():
        return "skipped"
    frames = segment["samples"] // cm.HOP
    features = {s: {"energy": None, "zero_runs": np.zeros((0, 2), np.int64), "clip_runs": np.zeros((0, 2), np.int64),
                    "samples": segment["samples"], "frames": frames} for s in segment["speakers"]}
    for x in tested:
        path = restored / segment["tracks"][x]
        check_restored(path, segment["samples"])
        features[x] = xt.stream_features(xt.int16_blocks(path, sn.PARAMS["decode_block_samples"]), segment["samples"],
                                         sn.PARAMS)
    row = bs.read_segment(root, *location)
    masks = bs.segment_masks(row, frames, sn.PARAMS["words"]["pad_s"], sn.PARAMS["words"]["gap_s"],
                             tuple(sn.PARAMS["words"]["echo_window_s"]))
    result = xt.analyse(sid, {s: features[s] for s in masks["speakers"]}, masks["o"], masks["spans"], sn.PARAMS,
                        tested=list(tested), active=masks["a"],
                        windows={x: [int(o) // cm.WINDOW_S for o in offs] for x, offs in tested.items()},
                        fixed_lags=fixed, pair_min_frames=sn.PARAMS["n_track"]["pair_min_frames"],
                        window_min_frames=sn.PARAMS["n_track"]["window_min_frames"], lag_fallback=True, pooled=True)
    payload = sn.segment_payload(sid, result, tag)
    cm.write_atomic(target, json.dumps(cm.sanitize(payload), default=cm.nan_to_none))
    return "done"


def _safe(function, job):
    try:
        return function(job)
    except Exception as error:
        return f"failed: {type(error).__name__}: {error}"


def _safe_s2(job):
    return _safe(s2_segment, job)


def fixed_lags(payload: dict, tested) -> dict:
    """The raw-audio screen's lags of each tested track against every other track (the lag used, with its fallback,
    and the block lags)."""
    return {(p["x"], p["y"]): {"conv": p.get("lag_used"), "blocks": [b["lag"] for b in p.get("blocks", [])]}
            for p in payload["pairs"] if p["x"] in tested}


# ---- plan ------------------------------------------------------------------------------------------------------

def rows_by_window(rows: list[dict]) -> dict:
    result = collections.defaultdict(list)
    for w in rows:
        result[(w["seg"], int(w["offset_s"]), w["x"])].append(w)
    return result


def max_stat(rows: list[dict], name: str):
    values = [r.get(f"{name}_{kind}") for r in rows for kind in ("conv", "blk")]
    values = [v for v in values if v is not None and np.isfinite(v)]
    return max(values) if values else None


def plan(segments: list[dict], s0_windows: list[dict], decided: list[dict], s1_rows: dict, s2_rows: dict,
         th: dict, decisions: dict, params: dict = PARAMS, pairs: dict | None = None) -> dict:
    """Decisions, stop conditions, dev split and tar layout. Pure (no IO). s1_rows / s2_rows: the screens'
    window rows by (segment, offset, x); pairs: the raw-audio screen's pair rows by (segment, x) and y
    (screen.pairs_by_track), which the window rule's optional lag gate (window_lag_s) needs; S2 uses the same gate."""
    by_id = {s["segment_id"]: s for s in segments}
    if th.get("window_lag_s") is not None and pairs is None:
        raise ValueError("window_lag_s needs S1's pair rows")
    pairs = pairs or {}
    passed = [r for r in decided if r["keep"]]
    s2_reject = {}
    for r in passed:
        rejected = []
        for x in (r["spk0"], r["spk1"]):
            rows = s2_rows.get((r["segment_id"], r["offset_s"], x))
            if not rows:
                raise ValueError(f"No S2 result for {r['segment_id']} {r['offset_s']} {x}")
            rejected += [f"{x}:{y}" for y in sn.window_bleed(rows, pairs.get((r["segment_id"], x), {}), th,
                                                             r["offset_s"], prefix="s2_")]
        s2_reject[(r["segment_id"], r["offset_s"])] = rejected
    removed = sum(bool(v) for v in s2_reject.values())
    rule = params["s2"]
    control_flagged, control_hits = [], []
    for s in segments:
        for c in s["control_windows"]:
            key = (s["segment_id"], c["offset_s"], c["x"])
            if key not in s2_rows:  # not measured: the control is not restored or not screened
                continue
            found = pairs.get((s["segment_id"], c["x"]), {})
            if sn.window_bleed(s1_rows.get(key, []), found, th, c["offset_s"]):
                control_flagged.append(key)
                if sn.window_bleed(s2_rows.get(key, []), found, th, c["offset_s"], prefix="s2_"):
                    control_hits.append(key)
    removed_share = removed / max(len(passed), 1)
    hit_rate = len(control_hits) / len(control_flagged) if control_flagged else float("nan")
    controls_mode = decisions.get("s2_controls", rule["controls_default"])
    stops, notes = [], []
    if removed_share > rule["max_removed_share"]:
        stops.append(f"S2 removes {removed_share:.2%} of the windows that passed S1 (> {rule['max_removed_share']:.0%})")
    controls_message = (f"S2 catches {len(control_hits)} of {len(control_flagged)} S1-flagged control windows "
                        f"(the stop rule needs >= {rule['min_control_hit_rate']:.0%} of >= {rule['min_control_windows']})")
    if len(control_flagged) < rule["min_control_windows"] or not hit_rate >= rule["min_control_hit_rate"]:
        (stops if controls_mode == "stop" else notes).append(controls_message)
    kept = [(r["segment_id"], r["offset_s"]) for r in passed if not s2_reject[(r["segment_id"], r["offset_s"])]]
    per_group = collections.Counter(by_id[sid]["conversation_id"] for sid, _ in kept)
    participants = collections.defaultdict(list)
    for s in segments:
        for x in s["speakers"]:
            if x not in participants[s["conversation_id"]]:
                participants[s["conversation_id"]].append(x)
    dev_groups, dev_components = lo.dev_split_all(per_group, participants, params["dev_min_windows"])
    train_offsets = collections.defaultdict(list)
    dev_windows = []
    for sid, offset in kept:
        if by_id[sid]["conversation_id"] in dev_groups:
            dev_windows.append((sid, offset))
        else:
            train_offsets[sid].append(offset)
    blocks = [(sid, b) for sid in sorted(train_offsets) for b in lo.blocks_of(train_offsets[sid],
                                                                               params["block_max_windows"])]
    tars = lo.assign(blocks, params["train_tars"], params["seed"])
    dev_windows.sort()
    train_people = {x for sid in train_offsets for x in participants[by_id[sid]["conversation_id"]]}
    dev_people = {x for g in dev_groups for x in participants[g]}
    if train_people & dev_people:
        raise AssertionError("dev and train share a participant")
    lang = collections.Counter(by_id[sid]["lang"] for sid, _ in dev_windows)
    counts = {
        "s0_windows_kept": sum(r["s0_keep"] for r in decided), "after_s1": len(passed),
        "s2_removed": removed, "s2_removed_share": removed_share,
        "control_tracks": sum(len(s["controls"]) for s in segments),
        "control_s1_flagged_windows": len(control_flagged), "control_s2_hits": len(control_hits),
        "control_hit_rate": hit_rate, "controls_mode": controls_mode,
        "kept": len(kept), "train_windows": sum(len(t) for t in tars), "dev_windows": len(dev_windows),
        "dev_components": len(dev_components), "dev_conversations": len({by_id[s]["conversation_id"]
                                                                         for s, _ in dev_windows}),
        "dev_by_language": dict(lang.most_common()), "dev_participants": len(dev_people),
        "train_participants": len(train_people), "train_segments": len(train_offsets), "blocks": len(blocks),
        "tar_windows_min": min(len(t) for t in tars), "tar_windows_max": max(len(t) for t in tars),
        "hours_train": sum(len(t) for t in tars) * cm.WINDOW_S / 3600,
        "train_other_on_call": None,
    }
    w0 = {(w["segment_id"], int(w["offset_s"])): w for w in s0_windows}
    counts["train_other_on_call"] = sum(1 for t in tars for sid, off, _ in t if (w0[(sid, off)]["n_other_on_call"] or 0) > 0)
    return {"counts": counts, "stops": stops, "notes": notes, "tars": tars, "dev": dev_windows,
            "s2_reject": s2_reject, "control_flagged": control_flagged, "control_hits": control_hits}


def manifest_rows(layout: dict, segments: list[dict], s0_windows: list[dict], screen_rows: dict, quality: dict,
                  s1_rows: dict, s2_rows: dict, th: dict) -> list[dict]:
    by_id = {s["segment_id"]: s for s in segments}
    w0 = {(w["segment_id"], int(w["offset_s"])): w for w in s0_windows}
    tracks = {(t["seg"], t["x"]): t for t in screen_rows["tracks"]}
    rows = []

    def row(sid, offset, split, tar, block, position):
        w = w0[(sid, offset)]
        segment = by_id[sid]
        r = {"key": bs.window_key(sid, offset), "segment_id": sid, "conversation_id": segment["conversation_id"],
             "lang": segment["lang"], "offset_s": offset, "split": split, "tar": tar, "block": block,
             "position": position, "spk0": w["spk0"], "spk1": w["spk1"], "n_eligible": w["n_eligible"],
             "n_other_on_call": w["n_other_on_call"], "union": w["union"], "third_only": w["third_only"]}
        for c, x in enumerate((w["spk0"], w["spk1"])):
            q = quality["windows"].get((sid, x, offset)) or {}
            r[f"bandwidth_{c}"] = segment["bandwidth_hz"].get(x)
            r[f"gated_{c}"] = dd.gated(tracks[(sid, x)], th)
            r[f"ovr_{c}"] = q.get("median")
            for stage, source in (("s1", s1_rows), ("s2", s2_rows)):
                found = source.get((sid, offset, x), [])
                r[f"{stage}_blr_max_{c}"] = max_stat(found, "blr")
                r[f"{stage}_exc_max_{c}"] = max_stat(found, "exc")
        return r

    for index, contents in enumerate(layout["tars"]):
        for position, (sid, offset, block) in enumerate(contents):
            rows.append(row(sid, offset, "train", index, block, position))
    for position, (sid, offset) in enumerate(layout["dev"]):
        rows.append(row(sid, offset, "dev", 0, -1, position))
    return rows


# ---- write -----------------------------------------------------------------------------------------------------

def write_tar(job) -> str:
    split, index, entries, restored, out, digest = job  # entries: (segment_id, offset, spk0, spk1, rel0, rel1)
    keys = [bs.window_key(sid, offset) for sid, offset, *_ in entries]

    def records():
        for (sid, offset, spk0, spk1, rel0, rel1), key in zip(entries, keys):
            samples = lo.read_window([restored / rel0, restored / rel1], offset, [f"{sid} {spk0}", f"{sid} {spk1}"])
            stem = rel0[:-len(f".{spk0}.flac")]
            yield {"__key__": key, "flac": lo.flac_bytes(samples),
                   "original_path.txt": f"{restored}/{stem}.{{{spk0},{spk1}}}.flac offset={offset * cm.SR} "
                                        f"frames={cm.WINDOW_SAMPLES}"}

    return lo.write_tar(out / split / f"data-{index:05d}.tar", out / "done" / f"{split}-{index:05d}.json", keys,
                        records, digest)


def _safe_write(job):
    return _safe(write_tar, job)


# ---- command ---------------------------------------------------------------------------------------------------

def load_decisions(path: Path) -> dict:
    return cm.load_json(path)


def readiness(segments: list[dict], s0_windows: list[dict], screen: Path, data_root: Path,
              restored_ok: set[str] | None, dnsmos_required: bool = True) -> dict[str, list[str]]:
    """Per segment with an S0-kept window, what is missing: ready (verification), rows (raw-audio screen), dnsmos
    (only when a DNSMOS rule is set), restored (output check)."""
    kept = tb.kept_windows(s0_windows)
    ready = tb.ready_segments(screen)
    missing = {}
    for s in segments:
        sid = s["segment_id"]
        if sid not in kept:
            continue
        reasons = []
        if sid not in ready:
            reasons.append("not_ready")
        if not (screen / "rows" / f"{sid}.json").exists():
            reasons.append("not_screened")
        if dnsmos_required and not (screen / "dnsmos" / f"{sid}.json").exists():
            reasons.append("no_dnsmos")
        if restored_ok is not None:
            pair = {x for w in kept[sid] for x in (w["spk0"], w["spk1"])}
            if any(s["tracks"][x] not in restored_ok for x in pair):
                reasons.append("not_restored")
        if reasons:
            missing[sid] = reasons
    return missing


def dnsmos_rules_on(th: dict) -> bool:
    """Whether a DNSMOS rule is set (the released thresholds set neither)."""
    return th.get("dnsmos_track_min") is not None or th.get("dnsmos_window_min") is not None


def restored_ok_paths(data_root: Path) -> set[str]:
    """Release paths with a valid restored output according to the 16 output-check records (none without the
    list)."""
    if not (data_root / "restore_input" / "restore_list.tsv").exists():
        return set()
    rows = ri.read_list(data_root)
    failures = set()
    for part in range(ri.PARTS):
        path = data_root / "restored" / f"{bs.NAME}.check.p{part:02d}.json"
        if not path.exists():
            failures.update(r["path"] for r in rows if r["part"] == part)
            continue
        failures.update(f["path"] for f in cm.load_json(path)["failures"])
    return {r["path"] for r in rows} - failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the downloaded release")
    parser.add_argument("--screen", type=Path, required=True, help="the screen directory")
    parser.add_argument("--decisions", type=Path, default=sn.THRESHOLDS,
                        help="thresholds file (default: the shipped one)")
    parser.add_argument("--data-root", type=Path, required=True, help="the work directory (restore_input/, restored/)")
    parser.add_argument("--restored", type=Path, default=None, help="default: <data root>/restored/basis-1500")
    parser.add_argument("--out", type=Path, default=None, help="default: <data root>/source")
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--allow-missing", type=Path, default=None,
                        help="TSV with a segment_id column (and an optional note): segments to leave out although "
                             "they are not ready, screened, scored or restored")
    parser.add_argument("--accept-s2-stop", action="store_true", help="write the tars despite an S2 stop condition")
    parser.add_argument("--plan-only", action="store_true", help="stop after manifest.tsv and plan.json")
    args = parser.parse_args(argv)
    args.restored = args.restored or args.data_root / "restored" / bs.NAME
    args.out = args.out or args.data_root / "source"
    decisions = load_decisions(args.decisions)
    th = {**sn.DEFAULT_THRESHOLDS, **decisions.get("thresholds", {})}
    fallback = bool(decisions.get("fallback_no_excluded_on_call", False))
    segments = tb.load_segments(args.screen)
    s0_windows = tb.load_windows(args.screen)
    allowed = {}
    if args.allow_missing:
        allowed = {r["segment_id"]: r.get("note", "") for r in cm.read_tsv(args.allow_missing)}
    restored_ok = restored_ok_paths(args.data_root)
    missing = readiness(segments, s0_windows, args.screen, args.data_root, restored_ok, dnsmos_rules_on(th))
    refused = {sid: reasons for sid, reasons in missing.items() if sid not in allowed}
    if refused:
        cm.write_json(args.out / "missing.json", refused)
        raise SystemExit(f"{len(refused)} segments with S0-kept windows are not ready, screened, scored or restored "
                         f"(e.g. {next(iter(refused.items()))}); see {args.out / 'missing.json'} or pass "
                         "--allow-missing with them")
    left_out = set(missing) & set(allowed)
    s0_used = [w if w["segment_id"] not in left_out else {**w, "keep": False} for w in s0_windows]
    screen_rows = sn.load_screen(args.screen)
    quality = sn.load_quality(args.screen)
    clip = sn.load_clip(args.screen) if th.get("window_clip_share_max") is not None else None
    excluded, decided = sn.decide(segments, s0_used, screen_rows, quality, th, fallback, clip)
    s1_rows = rows_by_window(screen_rows["windows"])
    index = bs.segment_index(args.root)
    tag = cm.params_hash({"cut": PARAMS, "s1": sn.PARAMS})
    passing = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in decided:
        if r["keep"]:
            for x in (r["spk0"], r["spk1"]):
                passing[r["segment_id"]][x].append(r["offset_s"])
    controls_skipped = []
    for s in segments:
        for c in s["control_windows"]:
            usable = s["segment_id"] not in left_out and s["tracks"][c["x"]] in restored_ok and \
                (args.screen / "rows" / f"{s['segment_id']}.json").exists()
            if usable:
                passing[s["segment_id"]][c["x"]].append(c["offset_s"])
            else:
                controls_skipped.append(f"{s['segment_id']}/{c['x']}/{c['offset_s']}")
    by_id = {s["segment_id"]: s for s in segments}
    jobs = []
    for sid, tested in sorted(passing.items()):
        payload = cm.load_json(args.screen / "rows" / f"{sid}.json")
        tested = {x: sorted(set(v)) for x, v in tested.items()}
        jobs.append((args.root, args.restored, args.out, by_id[sid], tested, index[sid], fixed_lags(payload, tested),
                     tag))
    (args.out / "s2").mkdir(parents=True, exist_ok=True)
    cm.write_json(args.out / "cut_start.json", cm.run_info("C", PARAMS, thresholds=th, decisions=str(args.decisions),
                                                           segments=len(jobs), allow_missing=allowed))
    failures = []
    jobs.sort(key=lambda j: -j[3]["samples"] * len(j[4]))
    with Pool(args.processes) as pool:
        for job, result in zip(jobs, pool.imap(_safe_s2, jobs)):
            if result.startswith("failed"):
                failures.append((job[3]["segment_id"], result))
                print(f"S2 {job[3]['segment_id']}: {result}", flush=True)
    if failures:
        cm.write_json(args.out / "s2_failures.json", failures)
        print(f"S2 incomplete: {len(failures)} segments failed; fix and rerun", flush=True)
        return 1
    s2_list = []
    for job in jobs:
        payload = cm.load_json(args.out / "s2" / f"{job[3]['segment_id']}.json")
        if payload["params_hash"] != tag:
            raise SystemExit(f"s2/{job[3]['segment_id']}.json was computed with other parameters")
        s2_list.extend(payload["windows"])
    s2_rows = rows_by_window(s2_list)
    layout = plan(segments, s0_used, decided, s1_rows, s2_rows, th, decisions,
                  pairs=sn.pairs_by_track(screen_rows))
    rows = manifest_rows(layout, segments, s0_used, screen_rows, quality, s1_rows, s2_rows, th)
    digest = lo.plan_hash(rows)
    summary = cm.run_info("C", PARAMS, thresholds=th, fallback_no_excluded_on_call=fallback,
                          decisions_file=str(args.decisions), dnsmos_required=dnsmos_rules_on(th),
                          counts=layout["counts"], stops=layout["stops"], notes=layout["notes"],
                          s2_stop_accepted=args.accept_s2_stop, plan_hash=digest, restored=str(args.restored),
                          allow_missing=allowed, left_out_segments=sorted(left_out),
                          control_windows_skipped=controls_skipped,
                          tracks_excluded=sum(1 for v in excluded.values() if v))
    existing = args.out / "plan.json"
    if existing.exists() and cm.load_json(existing)["plan_hash"] != digest and \
            any((args.out / split).glob("*.tar") for split in ("train", "dev")):
        raise SystemExit(f"{args.out} holds shards of another plan; write to a new --out")
    cm.write_tsv(args.out / "manifest.tsv", rows, MANIFEST_COLUMNS)
    cm.write_json(args.out / "plan.json", cm.sanitize({"plan_hash": digest, "counts": layout["counts"],
                                                       "stops": layout["stops"], "notes": layout["notes"]}))
    controls_rows = [w for w in s2_list if (w["seg"], w["x"]) in {(s["segment_id"], x) for s in segments
                                                                  for x in s["controls"]}]
    cm.write_tsv(args.out / "controls_s2.tsv", controls_rows, sn.WINDOW_COLUMNS)
    print(json.dumps(cm.sanitize({"counts": layout["counts"], "stops": layout["stops"], "notes": layout["notes"]}),
                     indent=1), flush=True)
    if layout["stops"] and not args.accept_s2_stop:
        cm.write_json(args.out / "summary.json", cm.sanitize({**summary, "status": "stopped"}))
        print("Stopped before writing shards: an S2 stop condition holds (see plan.json). Inspect the S2-rejected "
              "windows; --accept-s2-stop writes the tars anyway.", flush=True)
        return 3
    if args.plan_only:
        cm.write_json(args.out / "summary.json", cm.sanitize({**summary, "status": "planned"}))
        return 0
    w0 = {(w["segment_id"], int(w["offset_s"])): w for w in s0_used}

    def entry(sid, offset):
        w = w0[(sid, offset)]
        return (sid, offset, w["spk0"], w["spk1"], by_id[sid]["tracks"][w["spk0"]], by_id[sid]["tracks"][w["spk1"]])

    write_jobs = [("train", i, [entry(s, o) for s, o, _ in contents], args.restored, args.out, digest)
                  for i, contents in enumerate(layout["tars"])]
    write_jobs.append(("dev", 0, [entry(s, o) for s, o in layout["dev"]], args.restored, args.out, digest))
    status, failed = collections.Counter(), []
    with Pool(args.processes) as pool:
        for job, result in zip(write_jobs, pool.imap(_safe_write, write_jobs)):
            status[result.split(":")[0]] += 1
            if result.startswith("failed"):
                failed.append((job[0], job[1], result))
                print(f"tar {job[0]} {job[1]}: {result}", flush=True)
    complete = not failed
    cm.write_json(args.out / "summary.json", cm.sanitize({**summary, "status": "complete" if complete else "incomplete",
                                                          "write_status": dict(status), "write_failures": failed}))
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
