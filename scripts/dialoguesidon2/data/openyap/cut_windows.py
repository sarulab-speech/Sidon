"""Cut of OpenYap-1k: the post-restoration check S2, the speaker-disjoint dev split, blocks of at most 5 consecutive
windows, and the source tars of the producer.

Decodes the restored tracks (CPU only). Phases, each resumable (finished outputs are kept and skipped):

1. s2: per conversation with windows kept after S0 and S1, and per transcript-bleed control, recompute BLR/EXC on
   the restored tracks at S1's lags (s2/<conv>.json). Asserts that both restored tracks are mono 48 kHz PCM_16 and
   exactly as long as the raw track (FLAC header).
2. plan: windows kept = S0 keep, S1 keep (thresholds.json, or --decisions) and S2 keep (BLR >= -30 dB and EXC >= 3 dB
   at either lag rejects). Stops without writing tars (exit 3) when S2 removes more than 3% of the windows that
   passed S1, or catches less than 80% of the controls' S1-flagged windows (at least 10 are needed), unless
   --accept-s2-stop is given. Then the dev split (whole speaker-graph components, smallest first, until >= 64
   windows), blocks of <= 5 consecutive windows per conversation, a seeded shuffle (numpy default_rng(20261012)) and
   assignment of each block to the train tar with the fewest windows (128 tars). Writes manifest.tsv and plan.json.
3. write: one tar per pool job, written to <name>.tmp and renamed; a done/<split>-<n>.json marker holds the plan hash
   and the keys. Records: `flac` (stereo 48 kHz PCM_16, 5,760,000 frames, channel 0 = restored speaker_a, channel
   1 = speaker_b, samples copied losslessly) and `original_path.txt`; keys openyap-<12 hex>-<offset s, 5 digits>.

  python scripts/dialoguesidon2/data/openyap/cut_windows.py --root OPENYAP/conversations --screen WORK/screen \\
      --restored WORK/restored/openyap-1k --out WORK/source --extra-reject WORK/screen/transcript_lags_wide.json
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
from common import core as cm  # noqa: E402
from common import layout as lo  # noqa: E402
from openyap import corpus as oc  # noqa: E402
from openyap import screen_crosstalk as sc  # noqa: E402

PARAMS = {
    "version": 1, "seed": 20261012, "train_tars": 128, "block_max_windows": 5, "dev_min_windows": 64,
    "s2": {"max_removed_share": 0.03, "min_control_hit_rate": 0.80, "min_control_windows": 10},
    "record": {"frames": oc.WINDOW_SAMPLES, "channels": 2, "subtype": "PCM_16", "sample_rate": oc.SR},
}
MANIFEST_COLUMNS = (["key", "conv", "offset_s", "split", "tar", "block", "position", "speaker_a", "speaker_b",
                     "aec_known"]
                    + [f"s1_{f}_{s}" for s in oc.SPEAKERS for f in ("blr_conv", "exc_conv", "blr_blk", "exc_blk")]
                    + [f"s2_{f}_{s}" for s in oc.SPEAKERS for f in ("blr_conv", "exc_conv", "blr_blk", "exc_blk")])


# ---- S2 --------------------------------------------------------------------------------------------------------

def check_restored(path: Path, frames: int) -> None:
    import soundfile as sf
    info = sf.info(str(path))
    if (info.samplerate, info.channels, info.subtype, info.frames) != (oc.SR, 1, "PCM_16", frames):
        raise ValueError(f"{path}: {info.samplerate} Hz, {info.channels} ch, {info.subtype}, {info.frames} frames; "
                         f"expected 48000 Hz mono PCM_16 with the raw length {frames}")


def s2_conversation(job) -> str:
    conv, root, restored, out, fixed, tag = job
    target = out / "s2" / f"{conv}.json"
    if target.exists():
        return "skipped"
    raw = {s: oc.read_streaminfo(root / conv / f"speaker_{s}.flac")["ns"] for s in oc.SPEAKERS}
    if raw["a"] != raw["b"]:
        raise ValueError(f"{conv}: raw tracks differ in length")
    features = {}
    for s in oc.SPEAKERS:
        path = restored / conv / f"speaker_{s}.flac"
        check_restored(path, raw[s])
        features[s] = sc.stream_features(sc.int16_blocks(path), raw[s], None)
    o, g, spans = sc.conversation_masks(root, conv, features["a"]["frames"])
    tracks, windows, _ = sc.analyse(conv, features, o, g, spans, None, fixed_lags=fixed)
    payload = {"params_hash": tag, "conv": conv, "tracks": tracks, "windows": windows}
    oc.write_atomic(target, json.dumps(cm.sanitize(payload), default=cm.nan_to_none))
    return "done"


def _safe(function, job):
    try:
        return function(job)
    except Exception as error:
        return f"failed: {type(error).__name__}: {error}"


def _safe_s2(job):
    return _safe(s2_conversation, job)


# ---- plan ------------------------------------------------------------------------------------------------------

def components(convs: dict[str, tuple[str, str]]) -> list[list[str]]:
    """Connected components of the speaker graph (speakers joined by conversations), as conversation lists."""
    return lo.components(convs)


def dev_split(windows_per_conv: dict[str, int], speakers: dict[str, tuple[str, str]], minimum: int) -> set[str]:
    """Whole components, ordered by (kept windows, smallest conversation id), until >= minimum windows."""
    return lo.dev_split(windows_per_conv, speakers, minimum)


blocks_of = lo.blocks_of
assign = lo.assign
plan_hash = lo.plan_hash
flac_bytes = lo.flac_bytes


def plan(s0_rows, s0_windows, s1_tracks, s1_windows, s2_windows: dict, thresholds: dict, drop_aec_off: bool,
         params: dict = PARAMS, extra_reject=()) -> dict:
    """Decisions, stop conditions, dev split and tar layout. Pure (no IO)."""
    conv_reject, decided = sc.decide(s1_tracks, s1_windows, s0_rows, s0_windows, thresholds, drop_aec_off,
                                     extra_reject)
    s0 = {r["c"]: r for r in s0_rows}
    s1 = {(w["conv"], int(w["offset_s"])): w for w in s1_windows}
    controls = sorted(c for c, r in s0.items() if r.get("transcript_bleed"))
    passed_s1 = [r for r in decided if r["keep"] and r["conv"] not in controls]
    s2_reject = {}
    for r in passed_s1:
        w2 = s2_windows.get((r["conv"], r["offset_s"]))
        if w2 is None:
            raise ValueError(f"No S2 result for {r['conv']} {r['offset_s']}")
        s2_reject[(r["conv"], r["offset_s"])] = sc.window_bleed(w2, thresholds, prefix="s2_")
    removed = sum(s2_reject.values())
    control_flagged = [k for k, w in s1.items() if k[0] in controls and sc.window_bleed(w, thresholds)]
    control_hits = [k for k in control_flagged if k in s2_windows and sc.window_bleed(s2_windows[k], thresholds, "s2_")]
    rule = params["s2"]
    removed_share = removed / max(len(passed_s1), 1)
    hit_rate = len(control_hits) / len(control_flagged) if control_flagged else float("nan")
    stops = []
    if removed_share > rule["max_removed_share"]:
        stops.append(f"S2 removes {removed_share:.2%} of the windows that passed S1 (> {rule['max_removed_share']:.0%})")
    if len(control_flagged) < rule["min_control_windows"] or not hit_rate >= rule["min_control_hit_rate"]:
        stops.append(f"S2 catches {len(control_hits)} of {len(control_flagged)} S1-flagged control windows "
                     f"(needs >= {rule['min_control_hit_rate']:.0%} of >= {rule['min_control_windows']})")
    kept = [(r["conv"], r["offset_s"]) for r in passed_s1 if not s2_reject[(r["conv"], r["offset_s"])]]
    per_conv = collections.Counter(c for c, _ in kept)
    speakers = {c: (r["speaker_id_a"], r["speaker_id_b"]) for c, r in s0.items()}
    dev = dev_split(per_conv, speakers, params["dev_min_windows"])
    train_offsets = collections.defaultdict(list)
    dev_windows = []
    for conv, offset in kept:
        (dev_windows.append((conv, offset)) if conv in dev else train_offsets[conv].append(offset))
    blocks = [(conv, b) for conv in sorted(train_offsets) for b in blocks_of(train_offsets[conv],
                                                                              params["block_max_windows"])]
    tars = assign(blocks, params["train_tars"], params["seed"])
    dev_windows.sort()
    train_speakers = {x for c in train_offsets for x in speakers[c]}
    dev_speakers = {x for c in dev for x in speakers[c]}
    if train_speakers & dev_speakers:
        raise AssertionError("dev and train share a speaker")
    counts = {
        "s0_windows_kept": sum(r["s0_keep"] for r in decided),
        "s1_conversations_rejected": sum(1 for v in conv_reject.values() if v),
        "after_s1": len(passed_s1),
        "s2_removed": removed, "s2_removed_share": removed_share,
        "control_conversations": len(controls), "control_s1_flagged_windows": len(control_flagged),
        "control_s2_hits": len(control_hits), "control_hit_rate": hit_rate,
        "kept": len(kept), "train_windows": sum(len(t) for t in tars), "dev_windows": len(dev_windows),
        "dev_conversations": len(dev), "dev_speakers": len(dev_speakers), "train_speakers": len(train_speakers),
        "train_conversations": len(train_offsets), "blocks": len(blocks),
        "tar_windows_min": min(len(t) for t in tars), "tar_windows_max": max(len(t) for t in tars),
        "hours_train": sum(len(t) for t in tars) * oc.WINDOW_S / 3600,
        "aec_off_windows_train": sum(1 for t in tars for c, _, _ in t if s0[c].get("aec_known")),
    }
    return {"counts": counts, "stops": stops, "tars": tars, "dev": dev_windows, "decided": decided,
            "s2_reject": s2_reject, "conv_reject": conv_reject}


def manifest_rows(layout: dict, s0_rows, s1_windows, s2_windows) -> list[dict]:
    s0 = {r["c"]: r for r in s0_rows}
    s1 = {(w["conv"], int(w["offset_s"])): w for w in s1_windows}
    rows = []

    def row(conv, offset, split, tar, block, position):
        r = {"key": oc.window_key(conv, offset), "conv": conv, "offset_s": offset, "split": split, "tar": tar,
             "block": block, "position": position, "speaker_a": s0[conv]["speaker_id_a"],
             "speaker_b": s0[conv]["speaker_id_b"], "aec_known": bool(s0[conv].get("aec_known"))}
        for s in oc.SPEAKERS:
            for f in ("blr_conv", "exc_conv", "blr_blk", "exc_blk"):
                r[f"s1_{f}_{s}"] = s1[(conv, offset)].get(f"{f}_{s}")
                r[f"s2_{f}_{s}"] = s2_windows[(conv, offset)].get(f"{f}_{s}")
        return r

    for index, contents in enumerate(layout["tars"]):
        for position, (conv, offset, block) in enumerate(contents):
            rows.append(row(conv, offset, "train", index, block, position))
    for position, (conv, offset) in enumerate(layout["dev"]):
        rows.append(row(conv, offset, "dev", 0, -1, position))
    return rows


# ---- write -----------------------------------------------------------------------------------------------------

def read_window(restored: Path, conv: str, offset_s: int) -> np.ndarray:
    """Restored speaker_a and speaker_b samples of one window as (frames, 2) int16, read losslessly."""
    return lo.read_window([restored / conv / f"speaker_{s}.flac" for s in oc.SPEAKERS], offset_s,
                          [f"{conv} {s}" for s in oc.SPEAKERS])


def write_tar(job) -> str:
    split, index, entries, restored, out, digest = job
    keys = [oc.window_key(conv, offset) for conv, offset in entries]

    def records():
        for (conv, offset), key in zip(entries, keys):
            directory = restored / conv
            yield {"__key__": key, "flac": flac_bytes(read_window(restored, conv, offset)),
                   "original_path.txt": f"{directory}/speaker_{{a,b}}.flac offset={offset * oc.SR} "
                                        f"frames={oc.WINDOW_SAMPLES}"}

    return lo.write_tar(out / split / f"data-{index:05d}.tar", out / "done" / f"{split}-{index:05d}.json", keys,
                        records, digest)


def _safe_write(job):
    return _safe(write_tar, job)


# ---- command ---------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the release's conversations/ directory")
    parser.add_argument("--screen", type=Path, required=True, help="the screen directory (S1 outputs)")
    parser.add_argument("--s0", type=Path, default=None, help="directory of the S0 outputs (default: --screen)")
    parser.add_argument("--decisions", type=Path, default=None, help="thresholds JSON (default: thresholds.json)")
    parser.add_argument("--extra-reject", type=Path, default=None,
                        help="conversations to reject as well: the transcript lag scan's JSON or a list")
    parser.add_argument("--restored", type=Path, required=True, help="restored tracks (<work>/restored/openyap-1k)")
    parser.add_argument("--out", type=Path, required=True, help="source directory for the producer")
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--accept-s2-stop", action="store_true",
                        help="write the tars although an S2 stop condition holds (after listening to the windows)")
    parser.add_argument("--plan-only", action="store_true", help="stop after manifest.tsv and plan.json")
    args = parser.parse_args(argv)
    s0_dir = args.s0 or args.screen
    decisions = sc.load_decisions(args.decisions)
    thresholds = decisions["thresholds"]
    drop = bool(decisions.get("fallback_drop_aec_off", False))
    extra = sc.load_extra_reject(args.extra_reject)
    s0_rows = oc.load_json(s0_dir / "s0_conversations.json")
    s0_windows = oc.read_tsv(s0_dir / "s0_windows.tsv")
    s1_tracks, s1_windows = sc.load_screen(args.screen)
    conv_reject, decided = sc.decide(s1_tracks, s1_windows, s0_rows, s0_windows, thresholds, drop, extra)
    s0 = {r["c"]: r for r in s0_rows}
    controls = {c for c, r in s0.items() if r.get("transcript_bleed")}
    candidates = sorted({r["conv"] for r in decided if r["keep"]} | controls)
    tracks_by = {(t["conv"], t["s"]): t for t in s1_tracks}
    tag = oc.params_hash({"cut": PARAMS, "s1": sc.PARAMS})
    jobs = []
    for conv in candidates:
        fixed = {s: {"conv": tracks_by[(conv, s)]["lag"], "blocks": [b["lag"] for b in tracks_by[(conv, s)]["blocks"]]}
                 for s in oc.SPEAKERS}
        jobs.append((conv, args.root, args.restored, args.out, fixed, tag))
    (args.out / "s2").mkdir(parents=True, exist_ok=True)
    oc.write_json(args.out / "cut_start.json", oc.run_info("cut", PARAMS, thresholds=thresholds,
                                                           decisions=str(args.decisions), conversations=len(jobs)))
    failures = []
    with Pool(args.processes) as pool:
        for job, result in zip(jobs, pool.imap(_safe_s2, jobs)):
            if result.startswith("failed"):
                failures.append((job[0], result))
                print(f"S2 {job[0]}: {result}", flush=True)
    if failures:
        oc.write_json(args.out / "s2_failures.json", failures)
        print(f"S2 incomplete: {len(failures)} conversations failed; fix and rerun", flush=True)
        return 1
    s2_windows = {}
    for conv in candidates:
        payload = oc.load_json(args.out / "s2" / f"{conv}.json")
        if payload["params_hash"] != tag:
            raise SystemExit(f"s2/{conv}.json was computed with other parameters")
        for w in payload["windows"]:
            s2_windows[(conv, int(w["offset_s"]))] = w
    layout = plan(s0_rows, s0_windows, s1_tracks, s1_windows, s2_windows, thresholds, drop, extra_reject=extra)
    rows = manifest_rows(layout, s0_rows, s1_windows, s2_windows)
    digest = plan_hash(rows)
    summary = oc.run_info("cut", PARAMS, thresholds=thresholds, fallback_drop_aec_off=drop, extra_reject=extra,
                          decisions_file=str(args.decisions or sc.DEFAULT_DECISIONS), counts=layout["counts"],
                          stops=layout["stops"], s2_stop_accepted=args.accept_s2_stop, plan_hash=digest,
                          restored=str(args.restored))
    existing = args.out / "plan.json"
    if existing.exists() and oc.load_json(existing)["plan_hash"] != digest and \
            any((args.out / split).glob("*.tar") for split in ("train", "dev")):
        raise SystemExit(f"{args.out} holds tars of another plan; write to a new --out")
    oc.write_tsv(args.out / "manifest.tsv", rows, MANIFEST_COLUMNS)
    oc.write_json(args.out / "plan.json", {"plan_hash": digest, "counts": layout["counts"], "stops": layout["stops"]})
    controls_rows = [s2_windows[key] for key in sorted(s2_windows) if key[0] in controls]
    oc.write_tsv(args.out / "controls_s2.tsv", controls_rows,
                 ["conv", "offset_s"] + [f"{f}_{s}" for s in oc.SPEAKERS
                                         for f in ("lag_conv", "blr_conv", "exc_conv", "lag_blk", "blr_blk", "exc_blk")])
    print(json.dumps(cm.sanitize({"counts": layout["counts"], "stops": layout["stops"]}), indent=1), flush=True)
    if layout["stops"] and not args.accept_s2_stop:
        oc.write_json(args.out / "summary.json", cm.sanitize({**summary, "status": "stopped"}))
        print("Stopped before writing tars: an S2 stop condition holds. Listen to the S2-rejected windows; "
              "--accept-s2-stop proceeds.", flush=True)
        return 3
    if args.plan_only:
        oc.write_json(args.out / "summary.json", cm.sanitize({**summary, "status": "planned"}))
        return 0
    write_jobs = [("train", i, [(c, o) for c, o, _ in contents], args.restored, args.out, digest)
                  for i, contents in enumerate(layout["tars"])]
    write_jobs.append(("dev", 0, layout["dev"], args.restored, args.out, digest))
    status = collections.Counter()
    failed = []
    with Pool(args.processes) as pool:
        for job, result in zip(write_jobs, pool.imap(_safe_write, write_jobs)):
            status[result.split(":")[0]] += 1
            if result.startswith("failed"):
                failed.append((job[0], job[1], result))
                print(f"tar {job[0]} {job[1]}: {result}", flush=True)
    complete = not failed
    oc.write_json(args.out / "summary.json", cm.sanitize({**summary, "status": "complete" if complete else "incomplete",
                                                          "write_status": dict(status), "write_failures": failed}))
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
