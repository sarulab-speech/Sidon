"""S0 of OpenYap-1k: the metadata screen. Reads JSON and FLAC headers only (no audio is decoded).

Per conversation it computes the transcript-bleed lag histogram, the echo-word-free speech masks and, on the 120 s
grid from t = 0 (full windows only, from the FLAC header length), the speech coverage and the DNSMOS medians of every
window (from the release's speaker_*_dnsmos.json series). Rules:

* reject a conversation with transcript bleed (excess > 0.01, ptb > 3, peak >= 10 of the identical-word lag
  histogram over -1..+2 s), a `dropout` silence profile, a track DNSMOS OVR median below 2.5, or a track noise
  floor above -40 dBFS (a missing value passes);
* keep a window of a kept conversation if speech covers >= 50% of it, each speaker >= 5% (6 s), and each track's
  DNSMOS OVR median there is >= 2.3 where present (at least 3 defined DNSMOS values).

Outputs in --out: s0_conversations.json, s0_windows.tsv and s0_summary.json (counts and the run record).

  python scripts/dialoguesidon2/data/openyap/prescreen.py --root OPENYAP/conversations --out WORK/screen
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
from openyap import corpus as oc  # noqa: E402

PARAMS = {
    "window_s": oc.WINDOW_S, "fps": oc.FPS, "word_pad_s": 0.1, "word_gap_s": 0.3,
    "echo": {"min_len": 4, "window_s": [-0.3, 1.5]},
    "lag_histogram": {"range_s": [-1.0, 2.0], "bin_s": 0.02, "smooth_bins": 3},
    "transcript_bleed": {"excess_gt": 0.01, "ptb_gt": 3, "peak_ge": 10},
    "conversation": {"ovr_median_min": 2.5, "noise_floor_max_dbfs": -40.0, "reject_dropout": True},
    "window": {"union_min": 0.5, "speaker_min": 0.05, "ovr_min": 2.3, "dnsmos_min_finite": 3},
}
WINDOW_COLUMNS = ["conv", "offset_s", "keep", "conv_ok", "union", "both"] + [
    f"{name}_{s}" for name in ("sp", "ponly", "oonly", "ovr", "sig", "bak", "dn_null", "echo", "nw") for s in "ab"]
TRACK_FIELDS = ("speaker_id", "headphones", "echo_cancellation", "device")
METRIC_FIELDS = ("noise_floor_dbfs", "silence_profile", "silent_while_partner_spoke_pct", "effective_bandwidth_hz",
                 "dnsmos_ovr_median", "dnsmos_sig_median", "dnsmos_bak_median")


def lag_flag(words: dict[str, list[dict]]) -> dict | None:
    """Transcript-bleed statistic: lags of identical words (normalised length >= 4) within -1..+2 s, a 20 ms
    histogram smoothed over 3 bins; the direction with the larger excess."""
    best = None
    bins = np.arange(-1.0, 2.0 + 1e-9, 0.02)
    for s, o in (("a", "b"), ("b", "a")):
        index = collections.defaultdict(list)
        for w in words[o]:
            index[oc.norm_word(w["word"])].append(w["start"])
        arrays = {k: np.array(v) for k, v in index.items()}
        lags, n = [], 0
        for w in words[s]:
            key = oc.norm_word(w["word"])
            if len(key) < 4:
                continue
            n += 1
            times = arrays.get(key)
            if times is None:
                continue
            d = w["start"] - times
            d = d[(d > -1.0) & (d < 2.0)]
            if len(d):
                lags.extend(d.round(3).tolist())
        if n == 0:
            continue
        h, _ = np.histogram(np.array(lags), bins=bins)
        smooth = np.convolve(h, np.ones(3), "same")
        k = int(np.argmax(smooth))
        peak = smooth[k]
        background = (h.sum() - peak) / max(len(h) - 3, 1) * 3
        row = {"s": s, "lstar": round(float((bins[k] + bins[k + 1]) / 2), 2), "peak": int(peak),
               "bg": round(float(background), 1), "n": n, "rate": float(peak / n),
               "excess": float((peak - background) / n), "ptb": float(peak / (background + 1))}
        if best is None or row["excess"] > best["excess"]:
            best = row
    return best


def transcript_bleed(flag: dict | None) -> bool:
    rule = PARAMS["transcript_bleed"]
    return flag is not None and flag["excess"] > rule["excess_gt"] and flag["ptb"] > rule["ptb_gt"] \
        and flag["peak"] >= rule["peak_ge"]


def conversation(root: Path, conv: str) -> tuple[dict, list[dict]]:
    """The conversation row and its window rows, with the metadata the rules need."""
    info = {s: oc.read_streaminfo(Path(root) / conv / f"speaker_{s}.flac") for s in oc.SPEAKERS}
    samples = info["a"]["ns"]
    duration = samples / oc.SR
    meta = oc.load_json(Path(root) / conv / "meta.json")
    track_meta = {s: oc.load_json(Path(root) / conv / f"speaker_{s}_meta.json") for s in oc.SPEAKERS}
    words = {s: oc.load_words(root, conv, s) for s in oc.SPEAKERS}
    dnsmos_raw = {s: oc.load_json(Path(root) / conv / f"speaker_{s}_dnsmos.json") for s in oc.SPEAKERS}
    # Echo words: the same normalised token (length >= 4) within -0.3..+1.5 s of the partner's.
    flags, lags, nlong = {}, [], {}
    for s, o in (("a", "b"), ("b", "a")):
        index = collections.defaultdict(list)
        for w in words[o]:
            index[oc.norm_word(w["word"])].append(w["start"])
        index = {k: np.array(sorted(v)) for k, v in index.items()}
        flags[s] = np.zeros(len(words[s]), bool)
        n = 0
        for i, w in enumerate(words[s]):
            key = oc.norm_word(w["word"])
            if len(key) < 4:
                continue
            n += 1
            times = index.get(key)
            if times is None:
                continue
            d = w["start"] - times
            selected = d[(d > -0.3) & (d < 1.5)]
            if len(selected):
                flags[s][i] = True
                lags.append(float(selected[np.argmin(np.abs(selected))]))
        nlong[s] = n
    long_word = {s: np.array([len(oc.norm_word(w["word"])) >= 4 for w in words[s]], bool) for s in oc.SPEAKERS}
    echo = {s: float(flags[s][long_word[s]].sum()) / max(nlong[s], 1) for s in oc.SPEAKERS}
    raw = {s: oc.word_mask(words[s], duration) for s in oc.SPEAKERS}
    clean = {s: oc.word_mask([w for w, f in zip(words[s], flags[s]) if not f], duration) for s in oc.SPEAKERS}
    last_word = max([w["end"] for s in oc.SPEAKERS for w in words[s]] or [0])
    flag = lag_flag(words)
    row = dict(c=conv, T=duration, ns=samples, echo_a=echo["a"], echo_b=echo["b"], nlong_a=nlong["a"],
               nlong_b=nlong["b"], lag_med=float(np.median(lags)) if len(lags) >= 20 else None, nlags=len(lags),
               lag_iqr=(float(np.percentile(lags, 25)), float(np.percentile(lags, 75))) if len(lags) >= 20 else None,
               sp_a=float(raw["a"].mean()), sp_b=float(raw["b"].mean()), both=float((raw["a"] & raw["b"]).mean()),
               last_word=last_word, lag_flag=flag, transcript_bleed=transcript_bleed(flag),
               recorded_at=meta.get("recorded_at"), streaminfo=info,
               ns_equal=info["a"]["ns"] == info["b"]["ns"])
    for s in oc.SPEAKERS:
        o = oc.other(s)
        row[f"ponly_frac_{s}"] = float((clean[o] & ~clean[s]).mean())  # partner speaks, this one does not
        row[f"clean_sp_{s}"] = float(clean[s].mean())
        for field in TRACK_FIELDS:
            row[f"{field}_{s}"] = track_meta[s].get(field)
        for field in METRIC_FIELDS:
            row[f"{field}_{s}"] = (track_meta[s].get("audio_metrics") or {}).get(field)
        row[f"true_peak_dbtp_{s}"] = (track_meta[s].get("recording") or {}).get("true_peak_dbtp")
    dnsmos = {s: {k: np.array([np.nan if v is None else v for v in dnsmos_raw[s][k]], float)
                  for k in ("sig", "bak", "ovr")} for s in oc.SPEAKERS}
    windows = []
    window = oc.WINDOW_S
    for k in range(int(duration // window)):
        a0 = k * window
        f0, f1 = int(a0 * oc.FPS), int((a0 + window) * oc.FPS)
        w = dict(c=conv, off=a0)
        for s in oc.SPEAKERS:
            o = oc.other(s)
            m, p = clean[s][f0:f1], clean[o][f0:f1]
            w[f"sp_{s}"] = float(m.mean())
            w[f"ponly_{s}"] = float((p & ~m).mean()) * window
            w[f"oonly_{s}"] = float((m & ~p).mean()) * window
            # The DNSMOS series holds one value per 9.01 s clip starting every 3 s.
            i0, i1 = int(np.ceil(a0 / 3)), int((a0 + window - 9.01) // 3) + 1
            for q in ("sig", "ovr", "bak"):
                segment = dnsmos[s][q][i0:i1]
                w[f"{q}_{s}"] = float(np.nanmedian(segment)) if np.isfinite(segment).sum() >= 3 else None
            w[f"dn_null_{s}"] = float(np.isnan(dnsmos[s]["ovr"][i0:i1]).mean()) if i1 > i0 else None
            w[f"echo_{s}"] = int(sum(1 for x, f in zip(words[s], flags[s]) if f and a0 <= x["start"] < a0 + window))
            w[f"nw_{s}"] = int(sum(1 for x in words[s] if a0 <= x["start"] < a0 + window))
        union = clean["a"][f0:f1] | clean["b"][f0:f1]
        w["union"] = float(union.mean())
        w["both"] = float((clean["a"][f0:f1] & clean["b"][f0:f1]).mean())
        windows.append(w)
    return row, windows


def conversation_reasons(row: dict) -> list[str]:
    """The conversation rules on one conversation row (with the track metadata fields)."""
    rule = PARAMS["conversation"]
    reasons = []
    if row["transcript_bleed"]:
        reasons.append("transcript_bleed")
    if any(row[f"silence_profile_{s}"] == "dropout" for s in oc.SPEAKERS):
        reasons.append("dropout")
    for s in oc.SPEAKERS:
        if row[f"dnsmos_ovr_median_{s}"] < rule["ovr_median_min"]:
            reasons.append(f"ovr_median_{s}")
        floor = row[f"noise_floor_dbfs_{s}"]
        if (floor if floor is not None else -99) > rule["noise_floor_max_dbfs"]:
            reasons.append(f"noise_floor_{s}")
    return reasons


def window_ok(w: dict) -> bool:
    rule = PARAMS["window"]
    if w["union"] < rule["union_min"]:
        return False
    if min(w["sp_a"], w["sp_b"]) < rule["speaker_min"]:
        return False
    for s in oc.SPEAKERS:
        ovr = w[f"ovr_{s}"]
        if ovr is not None and ovr < rule["ovr_min"]:
            return False
    return True


def select(rows: list[dict], windows: list[dict]) -> dict:
    """Apply the rules; sets row['conv_ok'], row['reasons'], window['conv_ok'], window['keep'] and returns the
    summary counts."""
    by_conv = {r["c"]: r for r in rows}
    for r in rows:
        r["reasons"] = conversation_reasons(r)
        r["conv_ok"] = not r["reasons"]
        r["aec_known"] = any(r[f"headphones_{s}"] is not None for s in oc.SPEAKERS)
        r["n_windows"] = 0
        r["n_keep"] = 0
    for w in windows:
        r = by_conv[w["c"]]
        w["conv_ok"] = r["conv_ok"]
        w["keep"] = r["conv_ok"] and window_ok(w)
        r["n_windows"] += 1
        r["n_keep"] += int(w["keep"])
    kept = [w for w in windows if w["keep"]]
    speakers = collections.Counter()
    for w in kept:
        for s in oc.SPEAKERS:
            speakers[by_conv[w["c"]][f"speaker_id_{s}"]] += 1
    ok = [r for r in rows if r["conv_ok"]]
    return {
        "conversations": len(rows), "conversations_kept": len(ok),
        "conversation_hours_kept": sum(r["T"] for r in ok) / 3600,
        "conversations_with_kept_windows": sum(1 for r in ok if r["n_keep"]),
        "transcript_bleed": sum(r["transcript_bleed"] for r in rows),
        "transcript_bleed_hours": sum(r["T"] for r in rows if r["transcript_bleed"]) / 3600,
        "dropout": sum("dropout" in r["reasons"] for r in rows),
        "full_windows": len(windows), "windows_kept": len(kept), "window_hours_kept": len(kept) * oc.WINDOW_S / 3600,
        "speakers": len(speakers),
        "top10_speaker_share": sum(x for _, x in speakers.most_common(10)) / max(sum(speakers.values()), 1),
        "aec_known_conversations": sum(r["aec_known"] for r in rows),
        "aec_known_windows_kept": sum(1 for w in kept if by_conv[w["c"]]["aec_known"]),
        "reasons": dict(collections.Counter(x for r in rows for x in r["reasons"])),
    }


def _one(args):
    root, conv = args
    return conversation(root, conv)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the release's conversations/ directory")
    parser.add_argument("--out", type=Path, required=True, help="screen directory (shared with the later stages)")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="first N conversations (tests)")
    parser.add_argument("--force", action="store_true", help="rewrite existing outputs")
    args = parser.parse_args(argv)
    outputs = [args.out / name for name in ("s0_conversations.json", "s0_windows.tsv", "s0_summary.json")]
    if all(p.exists() for p in outputs) and not args.force:
        print(f"S0 outputs exist in {args.out}; nothing to do (--force rewrites them)")
        return 0
    convs = oc.conversation_ids(args.root)[:args.limit]
    jobs = [(args.root, c) for c in convs]
    if args.processes > 1:
        with Pool(args.processes) as pool:
            results = pool.map(_one, jobs, chunksize=4)
    else:
        results = [_one(job) for job in jobs]
    rows = [r for r, _ in results]
    windows = [w for _, ws in results for w in ws]
    summary = select(rows, windows)
    if not all(r["ns_equal"] for r in rows):
        raise SystemExit("Tracks a and b differ in length in: "
                         + ", ".join(r["c"] for r in rows if not r["ns_equal"]))
    table = []
    for w in windows:
        cell = {"conv": w["c"], "offset_s": w["off"], "keep": w["keep"], "conv_ok": w["conv_ok"],
                "union": w["union"], "both": w["both"]}
        for name in ("sp", "ponly", "oonly", "ovr", "sig", "bak", "dn_null", "echo", "nw"):
            for s in oc.SPEAKERS:
                cell[f"{name}_{s}"] = w[f"{name}_{s}"]
        table.append(cell)
    oc.write_json(args.out / "s0_conversations.json", rows)
    oc.write_tsv(args.out / "s0_windows.tsv", table, WINDOW_COLUMNS)
    record = oc.run_info("S0", PARAMS, root=str(args.root), counts=summary)
    oc.write_json(args.out / "s0_summary.json", record)
    print(json.dumps({"counts": summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
