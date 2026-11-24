"""Screen decisions of the data pipelines: the track and window bleed rules and the integrity rule with a per-track
gate profile. Thresholds come in as arguments (each corpus's `thresholds` in its decisions JSON).

The two-track rules (ge, track_bleeds) are used by the OpenYap-1k screen. The N-track rules act on the rows of
crosstalk.analyse as the Basis screen writes them: pair rows (x, y), window rows (x, y with y = "*" for the pooled
statistic) and track rows.
"""

from __future__ import annotations

import collections

import numpy as np


def ge(value, threshold) -> bool:
    return value is not None and np.isfinite(value) and value >= threshold


def track_bleeds(track: dict, th: dict) -> bool:
    """prominence >= track_prominence and (BLR >= track_blr_db or EXC >= track_exc_db)."""
    return ge(track.get("prominence"), th["track_prominence"]) and (
        ge(track.get("blr"), th["track_blr_db"]) or ge(track.get("exc"), th["track_exc_db"]))


# ---- N-track rules (Basis) -----------------------------------------------------------------------------------------
#
# Optional threshold keys (absent: the plain two-track rules above):
# * track_lag_s [lo, hi]: the track rule counts only peaks at lags (s, x hears y later when > 0) inside [lo, hi], where
#   playback echo and co-located bleed lie; chance peaks spread over the whole +-3.5 s search range.
# * track_level_mode "or" (default) | "and": BLR and EXC alternatively or jointly.
# * track_colocated_prominence (with track_colocated_lag_s, default 0.1): a peak this prominent within +-lag of 0
#   flags the pair without level statistics (co-located tracks whose y-only frames are too few for BLR/EXC).
# * window_lag_s [lo, hi] and window_prominence: a window row counts only at a lag inside [lo, hi] whose curve
#   (the conversation curve for the conversation lag, the window's block curve for the block lag) has at least this
#   prominence; a pooled row then counts only when another row of the same window passes this gate.
# * window_clip_runs_max None disables the run count; window_clip_share_max limits the clipped-sample share of the
#   track-window (the clip table of the Basis screen).

FPS = 100
BLOCK_FRAMES = 60_000  # the 10-minute lag blocks of the screen (crosstalk PARAMS lag_block_frames)


def in_lag_window(lag_frames, window_s) -> bool:
    return lag_frames is not None and window_s[0] <= lag_frames / FPS <= window_s[1]


def pair_bleeds(pair: dict, th: dict) -> bool:
    """The track rule on one ordered pair (x, y) with a searched lag (is_pair)."""
    if not pair.get("is_pair", True):
        return False
    colocated = th.get("track_colocated_prominence")
    lag = pair.get("lag")
    if colocated is not None and lag is not None and abs(lag) / FPS <= th.get("track_colocated_lag_s", 0.1) \
            and ge(pair.get("prominence"), colocated):
        return True
    window = th.get("track_lag_s")
    if window is not None and not in_lag_window(lag, window):
        return False
    if th.get("track_level_mode", "or") == "and":
        return ge(pair.get("prominence"), th["track_prominence"]) and ge(pair.get("blr"), th["track_blr_db"]) \
            and ge(pair.get("exc"), th["track_exc_db"])
    return track_bleeds(pair, th)


def window_gate(pairs: dict, th: dict, offset_s: int):
    """None without window_lag_s; else allowed(row, kind) for the window rows of x at offset_s: the row's lag lies
    in window_lag_s and its curve has prominence >= window_prominence (pairs: x's pair rows by y)."""
    lags = th.get("window_lag_s")
    if lags is None:
        return None
    minimum = th["window_prominence"]
    block = offset_s * FPS // BLOCK_FRAMES

    def allowed(row: dict, kind: str) -> bool:
        lag = row.get(f"lag_{kind}")
        pair = pairs.get(row["y"])
        if pair is None or not in_lag_window(lag, lags):
            return False
        if kind == "blk":
            blocks = pair.get("blocks") or []
            if block < len(blocks) and blocks[block].get("lag") == lag:
                return ge(blocks[block].get("prominence"), minimum)
        return pair.get("lag") == lag and ge(pair.get("prominence"), minimum)

    return allowed


def gated(track: dict, th: dict) -> bool:
    """Gate profile: at least gate_runs_per_hour_min exact-zero runs of >= 20 ms per hour of A_x (a noise-gated
    track, whose in-word zero runs are gate closures rather than dropouts)."""
    rate = track.get("gate_runs_per_hour")
    return rate is not None and np.isfinite(rate) and rate >= th["gate_runs_per_hour_min"]


def window_bleed_rows(rows: list[dict], th: dict, prefix: str = "", pooled: bool = True, allowed=None) -> list[str]:
    """The y (or '*') at which a window row of x fails: BLR >= {prefix}window_blr_db and EXC >=
    {prefix}window_exc_db at the conversation or the block lag (NaN passes). `allowed(row, kind)` (window_gate)
    restricts the rows and lags that count; a pooled row then counts only if some other row passes the gate."""
    blr_t, exc_t = th[f"{prefix}window_blr_db"], th[f"{prefix}window_exc_db"]
    gate_open = allowed is None or any(allowed(row, kind) for row in rows if row["y"] != "*"
                                       for kind in ("conv", "blk"))
    hits = []
    for row in rows:
        if row["y"] == "*" and (not pooled or not gate_open):
            continue
        for kind in ("conv", "blk"):
            if allowed is not None and row["y"] != "*" and not allowed(row, kind):
                continue
            if ge(row.get(f"blr_{kind}"), blr_t) and ge(row.get(f"exc_{kind}"), exc_t):
                hits.append(row["y"])
                break
    return hits


def integrity_fails(zero_runs: int | None, clip_runs: int | None, is_gated: bool, th: dict,
                    clip_share: float | None = None) -> list[str]:
    reasons = []
    limit = th["window_zero_runs_gated_max"] if is_gated else th["window_zero_runs_max"]
    if (zero_runs or 0) > limit:
        reasons.append("zero_runs")
    if th["window_clip_runs_max"] is not None and (clip_runs or 0) > th["window_clip_runs_max"]:
        reasons.append("clip_runs")
    share_max = th.get("window_clip_share_max")
    if share_max is not None:
        if clip_share is None:
            raise ValueError("window_clip_share_max needs the clipped-sample share of every track-window "
                             "(basis/screen.py clip-table)")
        if clip_share > share_max:
            reasons.append("clip_share")
    return reasons


def counter(values) -> dict:
    return dict(sorted(collections.Counter(values).items(), key=lambda kv: str(kv[0])))
