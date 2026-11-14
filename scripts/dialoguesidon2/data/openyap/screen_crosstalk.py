"""S1 of OpenYap-1k: crosstalk (bleed) and integrity screen of the raw per-speaker tracks.

`run` decodes audio (CPU only); `rescore` reads only the saved rows and applies the thresholds (thresholds.json next
to this file unless --decisions names another file).

Per track x with partner y (bleed model x(t) ~ g * y(t - lag)):

1. Decode 48 kHz int16 in 60 s blocks; record exact-zero runs >= 960 samples (20 ms) and clipped runs (>= 3 samples
   with |x| >= 32700). Resample to 16 kHz (scipy resample_poly(1, 3), computed blockwise with context, equal to the
   whole-signal call) and band-pass 150-3800 Hz (4th-order Butterworth SOS, state carried across blocks), so
   narrow-band devices compare with full-band ones.
2. Frame energy e[t] in dBFS over 20 ms windows with a 10 ms hop (100 fps), floored at -110 dB.
3. Word masks: O = words +-0.1 s with gaps < 0.3 s merged, without echo-flagged words (a word of x is echo-flagged
   when y says the same word, normalised length >= 4, at t_x - t_y in -0.3..+3.5 s); G = O dilated by 0.4 s. The
   window is one-sided on purpose: transcribed copies of bleed sit later than the partner's original when the tracks
   are aligned, and removing the originals as well would take the partner-only frames away from the statistics.
4. Lag search over -3.5..+3.5 s in 10 ms steps: rho(lag) is the correlation, on frames outside G_x, of
   max(e_x, p10 floor) with e_y(t - lag), per 10-minute block (blocks with < 30 s of frames are skipped); the
   conversation curve is the mean of the block curves. prominence = rho(lag*) - median rho over
   0.25 <= |lag - lag*| <= 0.6 s.
5. At lag*: P = shift(O_y) & ~G_x (partner-only), N = ~G_x & ~dilate(shift(O_y), 0.4 s) (nobody),
   L_act = mean of the top half of e_x on O_x & ~O_y; BLR = p90(e_x on P) - L_act;
   EXC = median(e_x on P) - median(e_x on N) (NaN below 5 s of P or N; a window with less than 5 s of P is NaN and
   passes, one with less than 5 s of N uses the conversation's N median). Diagnostics: ERL (least-squares power fit)
   and the 300-3400 Hz coherence of x(t) with y(t - lag*) on P.
6. Per 120 s window (the metadata screen's grid): BLR/EXC at the conversation lag* and at the window's block lag*,
   zero runs inside own word interiors (start + 30 ms .. end - 30 ms), clipped runs.

Decisions (rescore, and cut_windows.py): a conversation is rejected when either track bleeds (prominence >=
track_prominence and (BLR >= track_blr_db or EXC >= track_exc_db)) or when it is listed by --extra-reject (the
transcript lag scan); a window is rejected when either track at either lag has BLR >= window_blr_db and EXC >=
window_exc_db, or when it fails integrity (in-word zero runs above the limit of the track's silence profile, or more
than window_clip_runs_max clipped runs on a track).

Outputs (--out, the screen directory): rows/<conv>.json and frames/<conv>.npz per conversation (resumable: finished
conversations are skipped), then tracks.jsonl, windows.tsv and s1_run.json; rescore writes
s1_windows_decided_<tag>.tsv and s1_rescore_<tag>.json.

  python scripts/dialoguesidon2/data/openyap/screen_crosstalk.py --root OPENYAP/conversations --out WORK/screen \\
      run --processes 64
  python scripts/dialoguesidon2/data/openyap/screen_crosstalk.py --root OPENYAP/conversations --out WORK/screen \\
      rescore --extra-reject WORK/screen/transcript_lags_wide.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import core as cm  # noqa: E402
from common import crosstalk as xt  # noqa: E402
from common import decisions as dd  # noqa: E402
from openyap import corpus as oc  # noqa: E402
from openyap.scan_transcript_lags import load_extra_reject  # noqa: E402,F401  (used by cut_windows.py too)

PARAMS = {
    "version": 2,
    "decode_block_samples": 2_880_000,  # 60 s at 48 kHz; a multiple of 3 and of 480
    "resample": {"up": 1, "down": 3, "context_samples": 90},
    "bandpass_hz": [150.0, 3800.0], "bandpass_order": 4,
    "frame": {"window_16k": 320, "hop_16k": 160},
    "floor_db": -110.0,
    "words": {"pad_s": 0.1, "gap_s": 0.3, "echo_words_removed": True, "echo_window_s": [-0.3, 3.5]},
    "guard_frames": 40,
    "lag_frames": [-350, 350],
    "lag_block_frames": 60_000, "min_block_frames": 3_000,
    "prominence_band_frames": [25, 60],
    "floor_percentile": 10,
    "min_level_frames": 500, "min_own_frames": 300,
    "zero_run_samples": 960, "clip_level": 32700, "clip_run_samples": 3, "interior_margin_s": 0.03,
    "coherence": {"band_hz": [300.0, 3400.0], "nperseg": 512, "max_seconds": 300.0, "min_seconds": 2.0},
}
DEFAULT_DECISIONS = Path(__file__).resolve().with_name("thresholds.json")
WINDOW_FRAMES = oc.WINDOW_S * oc.FPS  # 12,000
TRACK_WINDOW_FIELDS = ("lag_conv", "blr_conv", "exc_conv", "np_conv", "lag_blk", "blr_blk", "exc_blk", "np_blk",
                       "zr", "clip")
WINDOW_COLUMNS = ["conv", "offset_s"] + [f"{f}_{s}" for s in oc.SPEAKERS for f in TRACK_WINDOW_FIELDS]
DECIDED_COLUMNS = ["conv", "offset_s", "s0_keep", "screened", "s1_conv_reject", "s1_window_bleed", "s1_integrity",
                   "s1_keep", "keep"]


# ---- signal path and statistics (common/crosstalk.py with this module's PARAMS) ----------------------------------

edges_of = xt.edges_of
RunTracker = xt.RunTracker
runs_of = xt.runs_of
count_in_interiors = xt.count_in_interiors


def bandpass_sos():
    return xt.bandpass_sos(PARAMS)


def int16_blocks(path: Path, block: int = PARAMS["decode_block_samples"]):
    return xt.int16_blocks(path, block)


def stream_features(blocks, total: int, bp16_path: Path | None = None) -> dict:
    """Integrity runs at 48 kHz and 100 fps band-passed energies from int16 blocks (each a multiple of 3 samples
    except the last). Optionally writes the 16 kHz band-passed signal (float32) to `bp16_path`."""
    return xt.stream_features(blocks, total, PARAMS, bp16_path)


def lag_curves(e_x: np.ndarray, e_y: np.ndarray, valid: np.ndarray, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return xt.lag_curves(e_x, e_y, valid, lags, PARAMS)


def peak(curve: np.ndarray, lags: np.ndarray) -> tuple[int | None, float, float]:
    return xt.peak(curve, lags, PARAMS)


def own_level(e_x: np.ndarray, own: np.ndarray) -> float:
    return xt.own_level(e_x, own, PARAMS)


def lag_masks(guard_x: np.ndarray, o_y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Partner-only frames P = shift(O_y, lag) & ~G_x and nobody frames N = ~G_x & ~dilate(shift(O_y), 0.4 s)."""
    return xt.lag_masks(guard_x, o_y, lag, PARAMS)


def level_stats(e: np.ndarray, own: np.ndarray, partner: np.ndarray, nobody: np.ndarray,
                l_act: float | None = None, n_med: float | None = None) -> dict:
    return xt.level_stats(e, own, partner, nobody, PARAMS, l_act, n_med)


def levels(e_x, own_x, guard_x, o_y, lag: int, region: slice | None = None, l_act: float | None = None) -> dict:
    """BLR and EXC of track x at `lag` (frames), over all frames or a window `region`."""
    partner, nobody = lag_masks(guard_x, o_y, lag)
    if region is None:
        return level_stats(e_x, own_x, partner, nobody, l_act)
    return level_stats(e_x[region], own_x[region], partner[region], nobody[region], l_act)


def erl(e_x, e_y, partner: np.ndarray, lag: int) -> float:
    return xt.erl(e_x, e_y, partner, lag, PARAMS)


def coherence(bp_x: np.ndarray, bp_y: np.ndarray, partner: np.ndarray, lag: int) -> float:
    return xt.coherence(bp_x, bp_y, partner, lag, PARAMS)


# ---- per conversation ----------------------------------------------------------------------------------------------

def echo_window() -> tuple[float, float]:
    """The echo-word window (s) of this screen, shared by the post-restoration check and the shard QC."""
    lo, hi = PARAMS["words"]["echo_window_s"]
    return float(lo), float(hi)


def conversation_masks(root: Path, conv: str, frames: int) -> tuple[dict, dict, dict]:
    """O (word) masks, G (guard) masks and own interiors (seconds) of both speakers, echo words removed."""
    words = oc.clean_words({s: oc.load_words(root, conv, s) for s in oc.SPEAKERS}, echo_window())
    duration = frames / oc.FPS
    pad, gap = PARAMS["words"]["pad_s"], PARAMS["words"]["gap_s"]
    o = {s: oc.word_mask(words[s], duration + 1.0, pad=pad, gap=gap, frames=frames) for s in oc.SPEAKERS}
    g = {s: oc.dilate(o[s], PARAMS["guard_frames"]) for s in oc.SPEAKERS}
    spans = {s: oc.interiors(words[s], PARAMS["interior_margin_s"]) for s in oc.SPEAKERS}
    return o, g, spans


def analyse(conv: str, features: dict, o: dict, g: dict, spans: dict, bp: dict | None = None,
            fixed_lags: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    """Track rows, window rows and the arrays to save, from both tracks' features and masks.
    `fixed_lags` ({s: {"conv": lag, "blocks": [...]}}) skips the lag search (the post-restoration check).
    The two-track case of common.crosstalk.analyse (every frame active, no other participant, no pooled
    statistic), returned in this screen's row layout."""
    fixed = None if fixed_lags is None else {(s, oc.other(s)): fixed_lags[s] for s in oc.SPEAKERS}
    result = xt.analyse(conv, {s: features[s] for s in oc.SPEAKERS}, o, spans, PARAMS, bp=bp, fixed_lags=fixed,
                        guards=g, pooled=False, extra=False)
    tracks, arrays = {}, {}
    for s in oc.SPEAKERS:
        track, pair = result["tracks"][s], result["pairs"][(s, oc.other(s))]
        if fixed_lags is None:
            arrays[f"rho_{s}"] = result["rho"][(s, oc.other(s))].astype(np.float16)
        lag_star = pair["lag"]
        tracks[s] = {"conv": conv, "s": s, "frames": track["frames"], "samples": track["samples"], "lag": lag_star,
                     "lag_s": None if lag_star is None else lag_star / oc.FPS, "rho": pair["rho"],
                     "prominence": pair["prominence"], "l_act": track["l_act"], "blocks": pair["blocks"],
                     "zero_runs": track["zero_runs"], "zero_run_s": track["zero_run_s"],
                     "clip_runs": track["clip_runs"], "zero_runs_in_words": track["zero_runs_in_words"]}
        tracks[s].update({k: pair[k] for k in ("blr", "exc", "np", "nn", "n_med", "erl", "coherence")})
    stats = {(w["offset_s"], w["x"]): w for w in result["windows"]}
    integrity = {(w["offset_s"], w["x"]): w for w in result["integrity"]}
    windows = []
    for k in range(oc.full_windows(features["a"]["samples"])):
        offset = k * oc.WINDOW_S
        w = {"conv": conv, "offset_s": offset}
        for s in oc.SPEAKERS:
            row = stats[(offset, s)]
            for name in ("conv", "blk"):
                w.update({f"{f}_{name}_{s}": row[f"{f}_{name}"] for f in xt.STAT_FIELDS})
            w[f"zr_{s}"] = integrity[(offset, s)]["zr"]
            w[f"clip_{s}"] = integrity[(offset, s)]["clip"]
        windows.append(w)
    for s in oc.SPEAKERS:
        arrays[f"e_{s}"] = features[s]["energy"].astype(np.float16)
        arrays[f"o_{s}"] = np.packbits(o[s])
        arrays[f"zero_runs_{s}"] = features[s]["zero_runs"]
        arrays[f"clip_runs_{s}"] = features[s]["clip_runs"]
    arrays["frames"] = np.array(result["frames"])
    arrays["lags"] = result["lags"]
    return [tracks["a"], tracks["b"]], windows, arrays


def screen_conversation(args) -> str:
    """The screen of one conversation (worker). Writes frames/<conv>.npz, then rows/<conv>.json last."""
    root, out, conv, scratch, tag = args
    rows_path = out / "rows" / f"{conv}.json"
    if rows_path.exists():
        return "skipped"
    info = {s: oc.read_streaminfo(root / conv / f"speaker_{s}.flac") for s in oc.SPEAKERS}
    if info["a"]["ns"] != info["b"]["ns"] or any(i["sr"] != oc.SR or i["ch"] != 1 or i["bps"] != 16
                                                 for i in info.values()):
        raise ValueError(f"{conv}: unexpected FLAC header {info}")
    total = info["a"]["ns"]
    work = Path(tempfile.mkdtemp(prefix=f"s1-{conv}-", dir=scratch))
    try:
        features, bp = {}, {}
        for s in oc.SPEAKERS:
            path = work / f"bp16_{s}.npy"
            features[s] = stream_features(int16_blocks(root / conv / f"speaker_{s}.flac"), total, path)
            bp[s] = np.load(path, mmap_mode="r")
        o, g, spans = conversation_masks(root, conv, features["a"]["frames"])
        tracks, windows, arrays = analyse(conv, features, o, g, spans, bp)
        del bp
    finally:
        for path in work.glob("*"):
            path.unlink()
        work.rmdir()
    oc.write_atomic(out / "frames" / f"{conv}.npz", cm.npz_bytes(arrays))
    payload = {"params_hash": tag, "conv": conv, "tracks": tracks, "windows": windows}
    oc.write_atomic(rows_path, json.dumps(cm.sanitize(payload), default=cm.nan_to_none))
    return "done"


# ---- decisions -----------------------------------------------------------------------------------------------------

track_bleeds = dd.track_bleeds


def load_decisions(path: Path | None = None) -> dict:
    """The decisions file (thresholds.json next to this module by default), with every threshold key present."""
    defaults = oc.load_json(DEFAULT_DECISIONS)
    decisions = oc.load_json(path) if path is not None else defaults
    return {**decisions, "thresholds": {**defaults["thresholds"], **decisions.get("thresholds", {})}}


def window_bleed(window: dict, th: dict, prefix: str = "") -> bool:
    """Either track at either lag has BLR >= window_blr and EXC >= window_exc (NaN passes)."""
    blr_t = th[f"{prefix}window_blr_db"]
    exc_t = th[f"{prefix}window_exc_db"]
    for s in oc.SPEAKERS:
        for name in ("conv", "blk"):
            if dd.ge(window.get(f"blr_{name}_{s}"), blr_t) and dd.ge(window.get(f"exc_{name}_{s}"), exc_t):
                return True
    return False


def window_integrity(window: dict, th: dict, gated: dict | None = None) -> bool:
    """True if the window fails integrity: zero runs inside own words or too many clipped runs.
    With window_zero_runs_gated_max in th, the zero-run limit is per track, and noise-gated tracks (gated[s], from
    the release's silence_profile) get that limit: their in-word zero runs are mostly gate closures in pauses that
    the word span covers, while on other tracks they are dropouts."""
    if "window_zero_runs_gated_max" in th:
        gated = gated or {}
        zero = any((window.get(f"zr_{s}", 0) or 0) > th["window_zero_runs_gated_max" if gated.get(s)
                                                       else "window_zero_runs_max"] for s in oc.SPEAKERS)
    else:
        zero = (window.get("zr_a", 0) or 0) + (window.get("zr_b", 0) or 0) > th["window_zero_runs_max"]
    return zero or any((window.get(f"clip_{s}", 0) or 0) > th["window_clip_runs_max"] for s in oc.SPEAKERS)


def load_screen(out: Path) -> tuple[list[dict], list[dict]]:
    with open(out / "tracks.jsonl") as stream:
        tracks = [json.loads(line) for line in stream]
    windows = [{k: oc.parse_cell(v) if k != "conv" else v for k, v in row.items()}
               for row in oc.read_tsv(out / "windows.tsv")]
    return tracks, windows


def decide(tracks: list[dict], windows: list[dict], s0_rows: list[dict], s0_windows: list[dict],
           th: dict, drop_aec_off: bool = False, extra_reject=()) -> tuple[dict, list[dict]]:
    """Decisions over the metadata screen's grid: per conversation (bleeding tracks, extra rejections, the optional
    AEC fallback) and per window."""
    by_conv = collections.defaultdict(dict)
    for t in tracks:
        by_conv[t["conv"]][t["s"]] = t
    s0 = {r["c"]: r for r in s0_rows}
    conv_reject = {}
    wide = set(extra_reject)
    for conv, pair in by_conv.items():
        reasons = [f"bleed_{s}" for s in oc.SPEAKERS if s in pair and track_bleeds(pair[s], th)]
        if conv in wide:
            reasons.append("transcript_bleed_wide")
        if drop_aec_off and s0.get(conv, {}).get("aec_known"):
            reasons.append("aec_off_fallback")
        conv_reject[conv] = reasons
    screened = {(w["conv"], int(w["offset_s"])): w for w in windows}
    decided = []
    for w0 in s0_windows:
        key = (w0["conv"], int(w0["offset_s"]))
        w = screened.get(key)
        row = {"conv": key[0], "offset_s": key[1], "s0_keep": bool(int(w0["keep"])),
               "screened": w is not None}
        if w is None:
            row.update(s1_conv_reject=True, s1_window_bleed=False, s1_integrity=False, s1_keep=False)
        else:
            row["s1_conv_reject"] = bool(conv_reject.get(key[0]))
            row["s1_window_bleed"] = window_bleed(w, th)
            gated = {sp: s0.get(key[0], {}).get(f"silence_profile_{sp}") == "noise_gated" for sp in oc.SPEAKERS}
            row["s1_integrity"] = window_integrity(w, th, gated)
            row["s1_keep"] = not (row["s1_conv_reject"] or row["s1_window_bleed"] or row["s1_integrity"])
        row["keep"] = row["s0_keep"] and row["s1_keep"]
        decided.append(row)
    return conv_reject, decided


# ---- commands ------------------------------------------------------------------------------------------------------

def collect(out: Path, convs: list[str], tag: str) -> tuple[list[dict], list[dict]]:
    tracks, windows = [], []
    for conv in convs:
        payload = oc.load_json(out / "rows" / f"{conv}.json")
        if payload["params_hash"] != tag:
            raise SystemExit(f"rows/{conv}.json was written with parameters {payload['params_hash']}, not {tag}")
        tracks.extend(payload["tracks"])
        windows.extend(payload["windows"])
    oc.write_atomic(out / "tracks.jsonl", "".join(json.dumps(t) + "\n" for t in tracks))
    oc.write_tsv(out / "windows.tsv", windows, WINDOW_COLUMNS)
    return tracks, windows


def _safe_screen(job) -> str:
    try:
        return screen_conversation(job)
    except Exception as error:  # reported per conversation; a rerun resumes
        return f"failed: {type(error).__name__}: {error}"


def command_run(args) -> int:
    out = args.out
    tag = oc.params_hash(PARAMS)
    convs = oc.conversation_ids(args.root)
    if args.convs:
        wanted = set(args.convs.split(","))
        convs = [c for c in convs if c in wanted]
    if args.limit:
        convs = convs[:args.limit]
    (out / "rows").mkdir(parents=True, exist_ok=True)
    stale = [c for c in convs if (out / "rows" / f"{c}.json").exists()
             and oc.load_json(out / "rows" / f"{c}.json")["params_hash"] != tag]
    if stale:
        raise SystemExit(f"{len(stale)} conversations were screened with other parameters (e.g. {stale[0]}); "
                         "use a new --out")
    oc.write_json(out / "s1_run_start.json", oc.run_info("S1", PARAMS, params_hash=tag, conversations=len(convs)))
    scratch = Path(args.scratch or tempfile.gettempdir())
    jobs = [(args.root, out, c, scratch, tag) for c in convs]
    # Longest conversations first so the pool's tail is short.
    sizes = {c: (args.root / c / "speaker_a.flac").stat().st_size for c in convs}
    jobs.sort(key=lambda j: -sizes[j[2]])
    status = collections.Counter()
    failures = []
    with Pool(args.processes, maxtasksperchild=8) as pool:
        for conv, result in zip([j[2] for j in jobs], pool.imap(_safe_screen, jobs)):
            status[result.split(":")[0]] += 1
            if result.startswith("failed"):
                failures.append((conv, result))
                print(f"{conv}: {result}", flush=True)
            done = sum(status.values())
            if done % 50 == 0:
                print(f"S1 {done}/{len(jobs)} {dict(status)}", flush=True)
    if failures:
        oc.write_json(out / "s1_failures.json", failures)
        print(f"S1 incomplete: {len(failures)} conversations failed; rerun to resume", flush=True)
        return 1
    tracks, windows = collect(out, convs, tag)
    record = oc.run_info("S1", PARAMS, params_hash=tag, conversations=len(convs), tracks=len(tracks),
                         windows=len(windows), status=dict(status))
    oc.write_json(out / "s1_run.json", record)
    return 0


def command_rescore(args) -> int:
    out = args.out
    tracks, windows = load_screen(out)
    s0_rows = oc.load_json(args.s0 / "s0_conversations.json")
    s0_windows = oc.read_tsv(args.s0 / "s0_windows.tsv")
    decisions = load_decisions(args.decisions)
    th = decisions["thresholds"]
    drop = bool(decisions.get("fallback_drop_aec_off", False))
    extra = load_extra_reject(args.extra_reject)
    conv_reject, decided = decide(tracks, windows, s0_rows, s0_windows, th, drop, extra)
    s0_by_conv = {x["c"]: x for x in s0_rows}
    counts = {
        "s0_windows_kept": sum(r["s0_keep"] for r in decided),
        "s1_conversations_rejected": sum(1 for v in conv_reject.values() if v),
        "after_s1_conversation_rule": sum(r["s0_keep"] and not r["s1_conv_reject"] for r in decided),
        "after_s1_window_rule": sum(r["s0_keep"] and not r["s1_conv_reject"] and not r["s1_window_bleed"]
                                    for r in decided),
        "windows_kept": sum(r["keep"] for r in decided),
        "hours_kept": sum(r["keep"] for r in decided) * oc.WINDOW_S / 3600,
        "aec_off_windows_kept": sum(1 for r in decided if r["keep"] and s0_by_conv[r["conv"]].get("aec_known")),
    }
    tag = args.tag
    oc.write_tsv(out / f"s1_windows_decided_{tag}.tsv", decided, DECIDED_COLUMNS)
    report = oc.run_info("S1-rescore", PARAMS, thresholds=th, fallback_drop_aec_off=drop,
                         decisions_file=str(args.decisions or DEFAULT_DECISIONS), extra_reject=extra, counts=counts,
                         conversations_rejected={c: v for c, v in conv_reject.items() if v})
    oc.write_json(out / f"s1_rescore_{tag}.json", cm.sanitize(report))
    print(json.dumps(cm.sanitize({"counts": counts}), indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the release's conversations/ directory")
    parser.add_argument("--out", type=Path, required=True, help="the screen directory")
    parser.add_argument("--s0", type=Path, default=None, help="directory of the S0 outputs (default: --out)")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--processes", type=int, default=os.cpu_count())
    run.add_argument("--scratch", default=None, help="local scratch for the 16 kHz band-passed signals")
    run.add_argument("--convs", default=None, help="comma-separated subset")
    run.add_argument("--limit", type=int, default=None)
    rescore = sub.add_parser("rescore")
    rescore.add_argument("--decisions", type=Path, default=None, help="thresholds JSON (default: thresholds.json)")
    rescore.add_argument("--extra-reject", type=Path, default=None,
                         help="conversations to reject as well: the transcript lag scan's JSON or a list")
    rescore.add_argument("--tag", default="final")
    args = parser.parse_args(argv)
    args.s0 = args.s0 or args.out
    return {"run": command_run, "rescore": command_rescore}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
