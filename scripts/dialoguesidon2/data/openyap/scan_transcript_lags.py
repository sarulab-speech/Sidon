"""Transcript copy-lag scan of OpenYap-1k over a wider range than the metadata screen's. Reads transcript JSON only.

The metadata screen (prescreen.lag_flag) histograms the start-time differences d = t_x - t_y of identical words
(normalised length >= 4) over -1..+2 s in 20 ms bins smoothed over 3 bins, keeps the direction with the larger
excess and flags transcript bleed if excess > 0.01, ptb > 3 and peak >= 10. This scan computes the same statistic for
both directions over -10..+10 s and lists the conversations that the same rule flags in either direction but the
metadata screen does not (tracks misaligned by more than the screen's range). The crosstalk screen rejects the
listed conversations (screen_crosstalk.py --extra-reject, reading summary.flagged_not_s0 of this file).

Output (--out): per conversation and direction the peak lag, counts and rule statistics, the flagged
conversations, and those not flagged by the metadata screen.

  python scripts/dialoguesidon2/data/openyap/scan_transcript_lags.py --root OPENYAP/conversations \\
      --s0 WORK/screen --out WORK/screen/transcript_lags_wide.json --processes 4
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
from openyap import prescreen as pm  # noqa: E402

PARAMS = {"range_s": [-10.0, 10.0], "bin_s": 0.02, "smooth_bins": 3, "min_len": 4,
          "rule": pm.PARAMS["transcript_bleed"]}


def direction_peak(words: list[dict], partner: list[dict], lo: float, hi: float, bin_s: float = 0.02) -> dict | None:
    """The metadata screen's lag statistic of `words` against `partner` (d = own start - partner start in the open
    interval (lo, hi)); with (lo, hi) = (-1, 2) it equals the direction row of prescreen.lag_flag."""
    index = collections.defaultdict(list)
    for w in partner:
        index[oc.norm_word(w["word"])].append(w["start"])
    arrays = {k: np.sort(np.array(v)) for k, v in index.items()}
    bins = np.arange(lo, hi + 1e-9, bin_s)
    lags, n = [], 0
    for w in words:
        key = oc.norm_word(w["word"])
        if len(key) < PARAMS["min_len"]:
            continue
        n += 1
        times = arrays.get(key)
        if times is None:
            continue
        i0 = int(np.searchsorted(times, w["start"] - hi, "left"))
        i1 = int(np.searchsorted(times, w["start"] - lo, "right"))
        d = w["start"] - times[i0:i1]
        d = d[(d > lo) & (d < hi)]
        if len(d):
            lags.extend(d.round(3).tolist())
    if n == 0:
        return None
    h, _ = np.histogram(np.array(lags), bins=bins)
    smooth = np.convolve(h, np.ones(PARAMS["smooth_bins"]), "same")
    k = int(np.argmax(smooth))
    peak = smooth[k]
    background = (h.sum() - peak) / max(len(h) - 3, 1) * 3
    return {"lstar": round(float((bins[k] + bins[k + 1]) / 2), 2), "peak": int(peak), "bg": round(float(background), 1),
            "n": n, "rate": float(peak / n), "excess": float((peak - background) / n),
            "ptb": float(peak / (background + 1))}


def flagged(row: dict | None) -> bool:
    rule = PARAMS["rule"]
    return row is not None and row["excess"] > rule["excess_gt"] and row["ptb"] > rule["ptb_gt"] \
        and row["peak"] >= rule["peak_ge"]


def scan(words: dict[str, list[dict]], lo: float, hi: float) -> dict:
    rows = {s: direction_peak(words[s], words[oc.other(s)], lo, hi, PARAMS["bin_s"]) for s in oc.SPEAKERS}
    return {"directions": rows, "flagged": [s for s in oc.SPEAKERS if flagged(rows[s])]}


def _one(args) -> dict:
    root, conv = args
    lo, hi = PARAMS["range_s"]
    return {"c": conv, **scan({s: oc.load_words(root, conv, s) for s in oc.SPEAKERS}, lo, hi)}


def summarize(rows: list[dict], s0_rows: list[dict]) -> dict:
    s0 = {r["c"]: r for r in s0_rows}
    hits = [r for r in rows if r["flagged"]]
    beyond = sorted(r["c"] for r in hits if not s0.get(r["c"], {}).get("transcript_bleed"))
    missed = sorted(c for c, r in s0.items() if r.get("transcript_bleed") and c not in {h["c"] for h in hits})
    return {"conversations": len(rows), "flagged": len(hits), "flagged_not_s0": beyond, "s0_not_flagged": missed,
            "flagged_not_s0_detail": {c: {"rows": next(r for r in hits if r["c"] == c)["directions"],
                                          "s0_conv_ok": s0.get(c, {}).get("conv_ok"),
                                          "s0_reasons": s0.get(c, {}).get("reasons"),
                                          "s0_n_keep": s0.get(c, {}).get("n_keep")} for c in beyond}}


def load_extra_reject(path: Path | None) -> list[str]:
    """The conversations to reject in addition to the screens' rules: summary.flagged_not_s0 of this scan's JSON
    output, or a text file with one conversation id per line."""
    if path is None:
        return []
    path = Path(path)
    if path.suffix == ".json":
        return list(oc.load_json(path)["summary"]["flagged_not_s0"])
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the release's conversations/ directory")
    parser.add_argument("--s0", type=Path, required=True, help="directory of s0_conversations.json")
    parser.add_argument("--out", type=Path, required=True, help="output JSON (transcript_lags_wide.json)")
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args(argv)
    jobs = [(args.root, c) for c in oc.conversation_ids(args.root)]
    with Pool(args.processes) as pool:
        rows = pool.map(_one, jobs, chunksize=8)
    summary = summarize(rows, oc.load_json(args.s0 / "s0_conversations.json"))
    oc.write_json(args.out, {**oc.run_info("transcript-lag-scan", PARAMS, root=str(args.root)), "summary": summary,
                             "rows": rows})
    print(json.dumps({k: v for k, v in summary.items() if k != "flagged_not_s0_detail"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
