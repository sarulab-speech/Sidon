"""S0 of the Basis pipeline: windows, one pair per window and the activity rules, from the release metadata only
(segments.jsonl of every language and the Lhotse manifest; no audio is read).

* Track exclusions: T0, the track holds transcript copies of another track (the transcript-copy statistic per
  ordered pair with >= 60 s of common on-call time, over +-3.5 s and +-10 s, the sign of the peak lag naming the
  holder); T1, audio.bandwidth_hz < 3,000 (a missing value passes). T0 tracks are kept as controls of the
  restored-audio check (cut.py), never trained on.
* Window grid t0 = 120 k s for k < floor(num_samples / 5,760,000) on the segment's own clock.
* Eligibility in a window: E1 on call for >= 119.5 s (11,950 frames), E2 echo-word-free speech O_x >= 5%
  (600 frames), E3 not excluded.
* One pair per window: eligible speakers ranked by O_x frames (descending), ties by sha256 of
  "20261013|<segment>|<offset>|<speaker>"; the pair is the top two; channel 0 is the member with the smaller
  sha256 of "20261013|order|<segment>|<offset>|<speaker>". There is no re-pairing later: a window whose pair fails
  a later rule is dropped.
* Window rules: W1 >= 2 eligible speakers, W2 no muted span overlaps the window, W3 the pair's union O_a | O_b
  covers >= 50% of it. Other participants are not constrained.

Outputs (--out): s0_segments.json, s0_windows.tsv, transcript_lags.json and s0_summary.json (counts, an estimate of
the dev split, the run record). These tables are derived from Basis data: keep them private.

  python scripts/dialoguesidon2/data/basis/plan_pairs.py --root BASIS_ROOT --out WORK/screen --processes 16
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basis import corpus as bs  # noqa: E402
from common import core as cm  # noqa: E402
from common import layout as lo  # noqa: E402

PARAMS = {
    "version": 1, "seed": bs.SEED, "window_s": cm.WINDOW_S, "fps": cm.FPS,
    "words": {"normaliser": "NFKC, casefold, categories L/M/N", "min_len": {"default": 4, "ja": 2, "zh": 2, "ko": 3},
              "pad_s": 0.1, "gap_s": 0.3, "echo_window_s": list(bs.ECHO_WINDOW_S)},
    "copies": {"ranges_s": {k: list(v) for k, v in bs.COPY_RANGES.items()}, "rule": bs.COPY_RULE,
               "min_overlap_s": bs.COPY_MIN_OVERLAP_S, "colocated_s": bs.COLOCATED_S},
    "tracks": {"bandwidth_min_hz": 3000},
    "eligibility": {"on_call_min_frames": 11_950, "speech_min_frames": 600},
    "window": {"union_min": 0.5},
    "dev": {"min_windows": 64},
}
WINDOW_COLUMNS = ["segment_id", "conversation_id", "lang", "offset_s", "keep", "reason", "muted", "n_active",
                  "n_eligible", "eligible", "spk0", "spk1", "sp_0", "sp_1", "on_call_0", "on_call_1", "union",
                  "third_only", "n_other_on_call"]


def plan_segment(row: dict, samples: int, params: dict = PARAMS) -> tuple[dict, list[dict]]:
    """S0 of one segment row: the segment summary and one row per full 120 s window."""
    sid, lang = row["segment_id"], row["language"]
    frames = samples // cm.HOP
    words_cfg = params["words"]
    masks = bs.segment_masks(row, frames, pad=words_cfg["pad_s"], gap=words_cfg["gap_s"],
                             window=tuple(words_cfg["echo_window_s"]))
    speakers, clean, on_call, muted = masks["speakers"], masks["o"], masks["c"], masks["m"]
    bandwidth = {s["speaker"]: (s.get("audio") or {}).get("bandwidth_hz") for s in row["speakers"]}
    holders, copy_rows = bs.transcript_copies(masks["words"], speakers, on_call, lang)
    copies = set().union(*holders.values())
    excluded = {}
    for s in speakers:
        reasons = []
        if s in copies:
            reasons.append("T0")
        if bandwidth[s] is not None and bandwidth[s] < params["tracks"]["bandwidth_min_hz"]:
            reasons.append("T1")
        if reasons:
            excluded[s] = reasons
    rule = params["eligibility"]
    union_min = params["window"]["union_min"]
    any_speech = np.zeros(frames, bool)
    for s in speakers:
        any_speech |= clean[s]
    windows = []
    for k in range(samples // cm.WINDOW_SAMPLES):
        r = slice(k * cm.WINDOW_FRAMES, (k + 1) * cm.WINDOW_FRAMES)
        offset = k * cm.WINDOW_S
        on = {s: int(on_call[s][r].sum()) for s in speakers}
        speech = {s: int(clean[s][r].sum()) for s in speakers}
        stats = {s: [float(on[s] / cm.FPS), float(clean[s][r].mean())] for s in speakers}
        rank = lambda s: (-speech[s], bs.order_hash(sid, offset, s))  # noqa: E731
        active = sorted((s for s in speakers if speech[s] >= rule["speech_min_frames"]), key=rank)
        eligible = [s for s in active if on[s] >= rule["on_call_min_frames"] and s not in excluded]
        is_muted = bool(muted[r].any())
        w = {"segment_id": sid, "conversation_id": row["conversation_id"], "lang": lang, "offset_s": offset,
             "muted": is_muted, "active": active, "eligible": eligible, "stats": stats,
             "any_sp": float(any_speech[r].mean()), "n_active": len(active), "n_eligible": len(eligible)}
        for name, ranked in (("r0", active), ("r1", eligible)):
            if len(ranked) >= 2:
                a, b = ranked[0], ranked[1]
                pair = clean[a][r] | clean[b][r]
                third = np.zeros(cm.WINDOW_FRAMES, bool)
                for s in speakers:
                    if s not in (a, b):
                        third |= clean[s][r]
                w[f"{name}_union"] = float(pair.mean())
                w[f"{name}_third"] = float((third & ~pair).mean())
        if len(eligible) >= 2:
            pair = sorted(eligible[:2], key=lambda s: bs.order_hash("order", sid, offset, s))
            w.update(spk0=pair[0], spk1=pair[1], sp_0=stats[pair[0]][1], sp_1=stats[pair[1]][1],
                     on_call_0=stats[pair[0]][0], on_call_1=stats[pair[1]][0], union=w["r1_union"],
                     third_only=w["r1_third"],
                     n_other_on_call=sum(1 for s in speakers if s not in pair and on[s] > 0))
        if len(active) < 2:
            reason = "active<2"
        elif len(eligible) < 2:
            reason = "eligible<2"
        elif is_muted:
            reason = "muted"
        elif w["union"] < union_min:
            reason = "union"
        else:
            reason = ""
        w["reason"] = reason
        w["keep"] = reason == ""
        w["controls"] = [[x, eligible[0]] for x in sorted(copies) if not is_muted and eligible
                         and on[x] >= rule["on_call_min_frames"] and speech[x] >= rule["speech_min_frames"]]
        windows.append(w)
    kept = [w for w in windows if w["keep"]]
    segment = {
        "segment_id": sid, "conversation_id": row["conversation_id"], "lang": lang, "part": row.get("part"),
        "duration": row["duration"], "samples": samples, "frames": frames, "gap_before": row.get("gap_before"),
        "tracks": dict(row["tracks"]), "speakers": speakers,
        "on_call": {s["speaker"]: [[x["start"], x["end"]] for x in s["on_call"]] for s in row["speakers"]},
        "on_call_s": {s: float(on_call[s].sum() / cm.FPS) for s in speakers},
        "muted": [[x["start"], x["end"], x.get("reason")] for x in row["muted_spans"]],
        "muted_s": float(sum(x["end"] - x["start"] for x in row["muted_spans"])),
        "bandwidth_hz": bandwidth, "copies": copy_rows,
        "copy_tracks": {name: sorted(v) for name, v in holders.items()}, "excluded": excluded,
        "windows": len(windows), "kept": len(kept),
        "pair_tracks": sorted({s for w in kept for s in (w["spk0"], w["spk1"])}),
        "controls": sorted(copies),
        # Windows where a control would be eligible (E1, E2, no muted overlap), with the top eligible partner.
        "control_windows": [{"offset_s": w["offset_s"], "x": x, "partner": y} for w in windows
                            for x, y in w["controls"]],
        "audio": {s: "unscanned" for s in speakers},
    }
    return segment, windows


def _one(job):
    row, samples = job
    return plan_segment(row, samples)


def summarize(segments: list[dict], windows: list[dict], params: dict = PARAMS) -> dict:
    hours = lambda n: round(n * cm.WINDOW_S / 3600, 1)  # noqa: E731
    by_id = {s["segment_id"]: s for s in segments}
    r0 = [w for w in windows if w["n_active"] >= 2]
    r1 = [w for w in windows if w["n_eligible"] >= 2]
    r1u = [w for w in r1 if not w["muted"]]
    kept = [w for w in windows if w["keep"]]
    alt = [w for w in r1u if w["union"] >= 0.4]
    speakers = collections.Counter(s for w in kept for s in (w["spk0"], w["spk1"]))
    by_lang = collections.Counter(w["lang"] for w in kept)
    pair_tracks = {(w["segment_id"], s) for w in kept for s in (w["spk0"], w["spk1"])}
    controls = {(s["segment_id"], x) for s in segments for x in s["controls"]}
    track_hours = lambda items: round(sum(by_id[g]["samples"] for g, _ in items) / cm.SR / 3600, 1)  # noqa: E731
    on_call_hours = round(sum(by_id[g]["on_call_s"][s] for g, s in pair_tracks) / 3600, 1)
    bandwidth = np.array([b for b in (by_id[g]["bandwidth_hz"][s] for g, s in pair_tracks) if b is not None], float)
    t1_tracks = [(s["segment_id"], x) for s in segments for x, r in s["excluded"].items() if "T1" in r]
    rule = params["eligibility"]
    t1_windows = sum(1 for w in windows for x in by_id[w["segment_id"]]["excluded"]
                     if "T1" in by_id[w["segment_id"]]["excluded"][x]
                     and w["stats"][x][0] * cm.FPS >= rule["on_call_min_frames"] and x in w["active"])
    copy_pairs = collections.Counter(s["lang"] for s in segments for c in s["copies"] if c["range"] == "wide")
    copy_lags = [c["lstar"] for s in segments for c in s["copies"]]
    pct = lambda values, q: [round(float(v), 3) for v in np.percentile(values, q)] if len(values) else []  # noqa
    windows_per_group = collections.Counter(w["conversation_id"] for w in kept)
    participants = collections.defaultdict(list)
    for s in segments:
        for x in s["speakers"]:
            if x not in participants[s["conversation_id"]]:
                participants[s["conversation_id"]].append(x)
    dev, dev_rows = lo.dev_split_all(windows_per_group, participants, params["dev"]["min_windows"])
    dev_lang = collections.Counter(w["lang"] for w in kept if w["conversation_id"] in dev)
    langs_all = collections.Counter(w["lang"] for w in kept)
    return {
        "segments": len(segments), "conversations": len({s["conversation_id"] for s in segments}),
        "hours": round(sum(s["samples"] for s in segments) / cm.SR / 3600, 1),
        "tracks": sum(len(s["tracks"]) for s in segments),
        "track_hours": round(sum(s["samples"] * len(s["tracks"]) for s in segments) / cm.SR / 3600, 1),
        "muted_spans": sum(len(s["muted"]) for s in segments),
        "muted_hours": round(sum(s["muted_s"] for s in segments) / 3600, 2),
        "full_windows": len(windows), "full_window_hours": hours(len(windows)),
        "windows_two_active": len(r0), "windows_two_active_hours": hours(len(r0)),
        "windows_two_eligible": len(r1), "windows_two_eligible_hours": hours(len(r1)),
        "windows_unmuted": len(r1u), "windows_unmuted_hours": hours(len(r1u)),
        "windows_kept": len(kept), "hours_kept": hours(len(kept)),
        "windows_kept_union_0_4": len(alt), "hours_kept_union_0_4": hours(len(alt)),
        "segments_with_kept": len({w["segment_id"] for w in kept}),
        "conversations_with_kept": len({w["conversation_id"] for w in kept}),
        "speakers_in_pairs": len(speakers),
        "top10_speaker_share": round(sum(c for _, c in speakers.most_common(10)) / max(sum(speakers.values()), 1), 3),
        "by_language": {lang: [n, hours(n)] for lang, n in by_lang.most_common()},
        "eligible_per_kept_window": dict(sorted(collections.Counter(min(w["n_eligible"], 5) for w in kept).items())),
        "other_on_call_per_kept_window": dict(sorted(collections.Counter(min(w["n_other_on_call"], 3)
                                                                         for w in kept).items())),
        "fallback_no_other_on_call": [sum(w["n_other_on_call"] == 0 for w in kept),
                                      hours(sum(w["n_other_on_call"] == 0 for w in kept))],
        "union_p10_p50_p90": pct([w["union"] for w in kept], [10, 50, 90]),
        "third_only_p50_p90_p99": pct([w["third_only"] for w in kept], [50, 90, 99]),
        "copy_tracks": len(controls), "copy_segments": len({g for g, _ in controls}),
        "copy_tracks_by_range": {name: sum(len(s["copy_tracks"][name]) for s in segments) for name in bs.COPY_RANGES},
        "copy_pairs_wide_by_language": dict(copy_pairs.most_common()),
        "copy_lag_abs_max_s": max((abs(x) for x in copy_lags), default=None),
        "t1_tracks": len(t1_tracks), "t1_track_windows": t1_windows,
        "pair_tracks": len(pair_tracks), "pair_track_hours": track_hours(pair_tracks),
        "pair_track_on_call_hours": on_call_hours,
        "controls": len(controls), "control_track_hours": track_hours(controls),
        "restore_tracks": len(pair_tracks | controls), "restore_track_hours": track_hours(pair_tracks | controls),
        "pair_bandwidth_p5_p25_p50_p75_p95": pct(bandwidth, [5, 25, 50, 75, 95]),
        "pair_bandwidth_below_8500": round(float((bandwidth < 8500).mean()), 3) if len(bandwidth) else None,
        "pair_bandwidth_missing": sum(1 for g, s in pair_tracks if by_id[g]["bandwidth_hz"][s] is None),
        "dev_estimate": {"components": len(dev_rows), "conversations": len(dev),
                         "windows": sum(r["windows"] for r in dev_rows),
                         "by_language": dict(dev_lang.most_common()),
                         "languages_moved_out": sorted(l for l, n in dev_lang.items() if n == langs_all[l])},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the downloaded release")
    parser.add_argument("--out", type=Path, required=True, help="the screen directory")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="rewrite existing outputs")
    args = parser.parse_args(argv)
    outputs = [args.out / name for name in ("s0_segments.json", "s0_windows.tsv", "s0_summary.json")]
    if all(p.exists() for p in outputs) and not args.force:
        print(f"S0 outputs exist in {args.out}; nothing to do (--force rewrites them)")
        return 0
    lengths = bs.lhotse_lengths(args.root)
    jobs = ((row, lengths[row["segment_id"]]) for row in bs.iter_segment_rows(args.root))
    if args.processes > 1:
        with Pool(args.processes) as pool:
            results = pool.map(_one, list(jobs), chunksize=4)
    else:
        results = [_one(job) for job in jobs]
    segments = [s for s, _ in results]
    windows = [w for _, ws in results for w in ws]
    summary = summarize(segments, windows)
    lags = {"ranges_s": PARAMS["copies"]["ranges_s"],
            "rows": [{"segment_id": s["segment_id"], "lang": s["lang"], **c} for s in segments for c in s["copies"]]}
    cm.write_json(args.out / "s0_segments.json", segments)
    rows = [{**w, "eligible": ",".join(w["eligible"])} for w in windows]
    cm.write_tsv(args.out / "s0_windows.tsv", rows, WINDOW_COLUMNS)
    cm.write_json(args.out / "transcript_lags.json", lags)
    record = cm.run_info("S0", PARAMS, root=str(args.root), revision=bs.release_revision(args.root),
                         expected_revision=bs.REVISION, counts=summary)
    cm.write_json(args.out / "s0_summary.json", record)
    print(json.dumps({"counts": summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
