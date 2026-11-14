"""Crosstalk (bleed) statistics of per-speaker tracks: the signal path shared by the screens and the N-track
analysis.

Bleed model: a tested track x holds a delayed, attenuated copy of another track y, x(t) ~ g * y(t - lag). The
statistics are computed on 100 fps band-passed frame energies (`stream_features`) and transcript word masks.

Per segment, x is a tested track and y any other track:

* A_x: x on call and not muted (`active`; None = every frame). O_x: echo-word-free word mask. G_x = dilate(O_x).
  own_x = O_x & ~(O_y for every other y); L_act,x = mean of the top half of e_x on own_x.
* Lag search per pair (x, y) with at least `pair_min_frames` of A_x & A_y: rho over valid = ~G_x & A_x & A_y,
  per 10-minute block; the conversation curve is the nanmean of the defined blocks. A pair without a defined
  curve (or under the minimum) has no lag; with `lag_fallback` it takes the lower median of x's defined pair
  lags, else 0 (used for Basis), otherwise it stays undefined and its statistics are NaN (used for OpenYap).
* O^others_xy = OR of shift(O_z, lag_xz) over z not in {x, y}; P_xy = shift(O_y, lag) & ~G_x & A_x &
  ~dilate(O^others_xy); N = A_x & ~G_x & ~dilate(shift(O_y, lag) | O^others_xy). BLR = p90(e_x on P) - L_act,
  EXC = median(e_x on P) - median(e_x on N) (level_stats).
* Pooled: P_x* = OR_y shift(O_y, lag_xy) & ~G_x & A_x, N_x* = A_x & ~G_x & ~dilate(OR_y shift(O_y, lag_xy));
  zshare_x* = share of P_x* frames that lie inside exact-zero runs of x.
* Windows: for each listed window of x, BLR/EXC against every y with at least `window_min_frames` of A_y in the
  window and pooled, at the conversation lags and at the window's block lags (other tracks at the same kind of
  lag), plus the integrity counts (zero runs inside own word interiors, clipped runs; runs starting outside A_x
  are not counted).

With two tracks, every frame active and no fallback, the N-track analysis reduces to the two-track screen used for
OpenYap-1k.
"""

from __future__ import annotations

import numpy as np

from . import core as cm

STAT_FIELDS = ("lag", "blr", "exc", "np")


# ---- signal path ---------------------------------------------------------------------------------------------------

def edges_of(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starts and ends (exclusive) of the runs of True in a boolean array (int8 arithmetic)."""
    padded = np.zeros(len(mask) + 2, np.int8)
    padded[1:-1] = mask
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


class RunTracker:
    """Runs of True samples of at least `min_length`, fed block by block: (start, length) in samples."""

    def __init__(self, min_length: int):
        self.min_length = min_length
        self.open_start: int | None = None
        self.runs: list[tuple[int, int]] = []

    def update(self, mask: np.ndarray, offset: int) -> None:
        if len(mask) == 0:
            return
        if self.open_start is not None and not mask[0]:
            self._close(self.open_start, offset)  # the open run ended at the block boundary
        starts, ends = edges_of(mask)
        for start, end in zip(starts, ends):
            start_abs = self.open_start if (start == 0 and self.open_start is not None) else offset + int(start)
            if end == len(mask):
                self.open_start = start_abs  # may continue into the next block
            else:
                self._close(start_abs, offset + int(end))

    def _close(self, start: int, end: int) -> None:
        self.open_start = None
        if end - start >= self.min_length:
            self.runs.append((start, end - start))

    def finish(self, total: int) -> np.ndarray:
        if self.open_start is not None:
            self._close(self.open_start, total)
        return np.array(self.runs, np.int64).reshape(-1, 2)


def bandpass_sos(params: dict):
    from scipy.signal import butter
    return butter(params["bandpass_order"], params["bandpass_hz"], btype="bandpass", fs=16_000, output="sos")


def int16_blocks(path, block: int):
    import soundfile as sf
    with sf.SoundFile(str(path)) as stream:
        if stream.samplerate != cm.SR or stream.channels != 1:
            raise ValueError(f"{path}: expected 48 kHz mono, got {stream.samplerate} Hz x {stream.channels}")
        while True:
            data = stream.read(block, dtype="int16", always_2d=False)
            if len(data) == 0:
                break
            yield data


def stream_features(blocks, total: int, params: dict, bp16_path=None) -> dict:
    """Integrity runs at 48 kHz and 100 fps band-passed energies from int16 blocks (each a multiple of 3
    samples except the last). Optionally writes the 16 kHz band-passed signal (float32) to `bp16_path`."""
    from scipy.signal import resample_poly, sosfilt
    context = params["resample"]["context_samples"]
    hop = params["frame"]["hop_16k"]
    sos = bandpass_sos(params)
    zi = np.zeros((sos.shape[0], 2))
    zeros = RunTracker(params["zero_run_samples"])
    clips = RunTracker(params["clip_run_samples"])
    n16 = -(-total // 3)
    hops = np.zeros(n16 // hop + 1, np.float64)
    bp = np.lib.format.open_memmap(str(bp16_path), mode="w+", dtype=np.float32, shape=(n16,)) \
        if bp16_path is not None else None
    previous_tail = np.zeros(context)
    position = 0  # samples at 48 kHz consumed
    out16 = 0
    pending = None

    def process(block: np.ndarray, head: np.ndarray):
        nonlocal previous_tail, position, out16, zi
        x = block.astype(np.float64) / 32768.0
        extended = np.concatenate([previous_tail, x, head])
        y = resample_poly(extended, 1, 3)
        count = -(-len(block) // 3)
        y = y[context // 3: context // 3 + count]
        y, zi = sosfilt(sos, y, zi=zi)
        if bp is not None:
            bp[out16:out16 + count] = y
        full = (count // hop) * hop
        start_hop = out16 // hop
        hops[start_hop:start_hop + count // hop] = (y[:full].reshape(-1, hop) ** 2).sum(1)
        wide = block.astype(np.int32)
        zeros.update(wide == 0, position)
        clips.update(np.abs(wide) >= params["clip_level"], position)
        tail = x[-context:]
        previous_tail = np.concatenate([previous_tail, tail])[-context:]
        position += len(block)
        out16 += count

    for block in blocks:
        if pending is not None:
            if len(pending) % 480:
                raise ValueError("Every block but the last must hold a multiple of 480 samples (one 10 ms hop)")
            head = (block[:context].astype(np.float64) / 32768.0)
            if len(head) < context:
                head = np.concatenate([head, np.zeros(context - len(head))])
            process(pending, head)
        pending = block
    if pending is not None:
        process(pending, np.zeros(context))
    if position != total:
        raise ValueError(f"Decoded {position} samples, the FLAC header says {total}")
    if bp is not None:
        bp.flush()
        del bp
    frames = total // 480
    s = hops[:frames + 1]
    if len(s) < frames + 1:
        s = np.concatenate([s, np.zeros(frames + 1 - len(s))])
    ms = (s[:frames] + s[1:frames + 1]) / params["frame"]["window_16k"]
    energy = 10 * np.log10(np.maximum(ms, 10 ** (params["floor_db"] / 10)))
    return {"energy": energy, "zero_runs": zeros.finish(total), "clip_runs": clips.finish(total),
            "samples": total, "frames": frames}


# ---- statistics ----------------------------------------------------------------------------------------------------

def lag_curves(e_x: np.ndarray, e_y: np.ndarray, valid: np.ndarray, lags: np.ndarray,
               params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-block correlation rho[b, j] of max(e_x, floor) with e_y(t - lags[j]) over valid frames t, and the
    frame counts. NaN where a block has fewer than min_block_frames; 0 where a series is constant."""
    frames = len(e_x)
    block_frames = params["lag_block_frames"]
    blocks = max(1, -(-frames // block_frames))
    floor = np.percentile(e_x[valid], params["floor_percentile"]) if valid.any() else params["floor_db"]
    a = np.maximum(e_x, floor)
    a = a - (a[valid].mean() if valid.any() else 0.0)
    y_all = e_y - e_y.mean()
    index = np.arange(frames) // block_frames
    rho = np.full((blocks, len(lags)), np.nan)
    counts = np.zeros((blocks, len(lags)), np.int64)
    for j, lag in enumerate(lags):
        lag = int(lag)
        if lag >= 0:
            tx, ty = slice(lag, frames), slice(0, frames - lag)
        else:
            tx, ty = slice(0, frames + lag), slice(-lag, frames)
        m = valid[tx]
        av, yv, bv = a[tx][m], y_all[ty][m], index[tx][m]
        n = np.bincount(bv, minlength=blocks).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            ma = np.bincount(bv, av, blocks) / n
            my = np.bincount(bv, yv, blocks) / n
            va = np.bincount(bv, av * av, blocks) / n - ma * ma
            vy = np.bincount(bv, yv * yv, blocks) / n - my * my
            cov = np.bincount(bv, av * yv, blocks) / n - ma * my
            r = cov / np.sqrt(va * vy)
        constant = (va < 1e-6) | (vy < 1e-6)
        r = np.where(constant, 0.0, r)
        r = np.where(n >= params["min_block_frames"], r, np.nan)
        rho[:, j] = r
        counts[:, j] = n.astype(np.int64)
    return rho, counts


def peak(curve: np.ndarray, lags: np.ndarray, params: dict) -> tuple[int | None, float, float]:
    """lag* (frames), rho(lag*) and the prominence of a correlation curve; (None, nan, nan) if undefined."""
    if not np.isfinite(curve).any():
        return None, float("nan"), float("nan")
    j = int(np.nanargmax(curve))
    lo, hi = params["prominence_band_frames"]
    distance = np.abs(lags - lags[j])
    ring = curve[(distance >= lo) & (distance <= hi)]
    ring = ring[np.isfinite(ring)]
    prominence = float(curve[j] - np.median(ring)) if len(ring) else float("nan")
    return int(lags[j]), float(curve[j]), prominence


def own_level(e_x: np.ndarray, own: np.ndarray, params: dict) -> float:
    values = np.sort(e_x[own])
    if len(values) < params["min_own_frames"]:
        return float("nan")
    return float(values[len(values) // 2:].mean())


def lag_masks(guard_x: np.ndarray, o_y: np.ndarray, lag: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Partner-only frames P = shift(O_y, lag) & ~G_x and nobody frames N = ~G_x & ~dilate(shift(O_y), 0.4 s)."""
    shifted = cm.shift(o_y, lag)
    return shifted & ~guard_x, ~guard_x & ~cm.dilate(shifted, params["guard_frames"])


def level_stats(e: np.ndarray, own: np.ndarray, partner: np.ndarray, nobody: np.ndarray, params: dict,
                l_act: float | None = None, n_med: float | None = None) -> dict:
    """BLR = p90(e on P) - L_act and EXC = median(e on P) - median(e on N), NaN below min_level_frames of P.
    L_act is the own level of these frames when there are >= min_own_frames own-only frames, else `l_act`;
    median(e on N) is these frames' when there are >= min_level_frames N frames, else `n_med` (a window
    falls back to its conversation's values; without a fallback the statistic is NaN, which passes)."""
    level = own_level(e, own, params)
    if not np.isfinite(level) and l_act is not None:
        level = l_act
    n_p, n_n = int(partner.sum()), int(nobody.sum())
    minimum = params["min_level_frames"]
    reference = float(np.median(e[nobody])) if n_n >= minimum else (n_med if n_med is not None else float("nan"))
    blr = exc = float("nan")
    if n_p >= minimum:
        blr = float(np.percentile(e[partner], 90) - level)
        exc = float(np.median(e[partner]) - reference)
    return {"blr": blr, "exc": exc, "np": n_p, "nn": n_n, "l_act": level, "n_med": reference}


def erl(e_x, e_y, partner: np.ndarray, lag: int, params: dict) -> float:
    if partner.sum() < params["min_level_frames"]:
        return float("nan")
    px = 10 ** (e_x[partner] / 10)
    py = 10 ** (cm.shift(e_y, lag, fill=params["floor_db"])[partner] / 10)
    gain = (px * py).sum() / max((py * py).sum(), 1e-300)
    return float(-10 * np.log10(gain)) if gain > 0 else float("nan")


def runs_of(mask: np.ndarray) -> np.ndarray:
    """(start, end) of every run of True frames."""
    return np.stack(edges_of(mask), 1)


def coherence(bp_x: np.ndarray, bp_y: np.ndarray, partner: np.ndarray, lag: int, params: dict) -> float:
    """Mean 300-3400 Hz magnitude-squared coherence of x(t) with y(t - lag) over partner-only runs."""
    from scipy.signal import coherence as msc
    cfg = params["coherence"]
    hop = params["frame"]["hop_16k"]
    pieces_x, pieces_y, total = [], [], 0
    limit = int(cfg["max_seconds"] * 16_000)
    for t0, t1 in runs_of(partner):
        a0, a1 = t0 * hop, t1 * hop
        b0, b1 = a0 - lag * hop, a1 - lag * hop
        if a1 - a0 < cfg["nperseg"] or b0 < 0 or b1 > len(bp_y) or a1 > len(bp_x):
            continue
        pieces_x.append(np.asarray(bp_x[a0:a1], np.float64))
        pieces_y.append(np.asarray(bp_y[b0:b1], np.float64))
        total += a1 - a0
        if total >= limit:
            break
    if total < cfg["min_seconds"] * 16_000:
        return float("nan")
    f, c = msc(np.concatenate(pieces_x), np.concatenate(pieces_y), fs=16_000, nperseg=cfg["nperseg"])
    band = (f >= cfg["band_hz"][0]) & (f <= cfg["band_hz"][1])
    return float(np.mean(c[band]))


def count_in_interiors(runs: np.ndarray, spans: np.ndarray, lo: int, hi: int) -> int:
    """Runs (start, length; samples) starting in [lo, hi) that overlap any interior span (seconds; sorted
    by start)."""
    if len(runs) == 0 or len(spans) == 0:
        return 0
    selected = runs[(runs[:, 0] >= lo) & (runs[:, 0] < hi)]
    if len(selected) == 0:
        return 0
    starts = spans[:, 0] * cm.SR
    max_end = np.maximum.accumulate(spans[:, 1] * cm.SR)
    run_start = selected[:, 0].astype(np.float64)
    run_end = run_start + selected[:, 1]
    k = np.searchsorted(starts, run_end, side="left")  # spans starting before the run ends
    hit = (k > 0) & (max_end[np.maximum(k - 1, 0)] > run_start)
    return int(hit.sum())


def runs_in(runs: np.ndarray, active: np.ndarray | None) -> np.ndarray:
    """Runs whose first sample lies in an active frame (all runs when `active` is None)."""
    if active is None or len(runs) == 0:
        return runs
    frames = np.minimum(runs[:, 0] // cm.HOP, len(active) - 1)
    return runs[active[frames]]


def lower_median(values: list[int]) -> int:
    ordered = sorted(values)
    return int(ordered[(len(ordered) - 1) // 2])


# ---- N-track analysis ----------------------------------------------------------------------------------------------

def _and(mask: np.ndarray, other: np.ndarray | None) -> np.ndarray:
    return mask if other is None else mask & other


def analyse(segment: str, features: dict, o: dict, spans: dict, params: dict, *, tested=None,
            active: dict | None = None, windows: dict | None = None, bp: dict | None = None,
            fixed_lags: dict | None = None, guards: dict | None = None, pair_min_frames: int = 0,
            window_min_frames: int = 0, lag_fallback: bool = False, pooled: bool = True, extra: bool = True) -> dict:
    """Track, pair and window statistics of one segment (module docstring).

    features: {track: {"energy", "zero_runs", "clip_runs", "samples", "frames"}} of every track of the segment
    (the order of this dict is the order of the others); o: {track: O mask}; spans: {x: word interiors (s)} of
    the tested tracks; active: {track: A mask} or None; tested: the tracks x (default all); windows: {x: [window
    index k]} (default every full window); bp: {track: 16 kHz band-passed signal} for the coherence diagnostic;
    fixed_lags: {(x, y): {"conv": lag, "blocks": [lag per block]}} replaces the lag search (used on restored
    tracks, where the lags of the raw-audio screen are reused); guards: {x: G_x}
    (default dilate(O_x, guard_frames)).
    `extra` adds the N-track fields (floor, gate rate, active counts) that the two-track rows do not have.
    Returns {"tracks": {x: row}, "pairs": {(x, y): row}, "windows": [rows], "integrity": [rows], "rho": {...}}.
    """
    order = list(features)
    tested = order if tested is None else list(tested)
    first = features[order[0]]
    frames, samples = first["frames"], first["samples"]
    lags = np.arange(params["lag_frames"][0], params["lag_frames"][1] + 1)
    guard = params["guard_frames"]
    block_frames = params["lag_block_frames"]
    n_windows = cm.full_windows(samples)
    window_frames = cm.WINDOW_S * cm.FPS
    track_rows, pair_rows, rho_arrays = {}, {}, {}
    window_rows, integrity_rows = [], []
    for x in tested:
        others = [y for y in order if y != x]
        e_x = features[x]["energy"]
        a_x = None if active is None else active[x]
        g_x = guards[x] if guards is not None else cm.dilate(o[x], guard)
        union = np.zeros(frames, bool)
        for y in others:
            union = union | o[y]
        own = o[x] & ~union
        l_act = own_level(e_x, own, params)
        # ---- lags
        lag_of, blocks_of, source = {}, {}, {}
        for y in others:
            overlap = frames if active is None else int((active[x] & active[y]).sum())
            row = {"x": x, "y": y, "overlap_frames": overlap, "is_pair": overlap >= pair_min_frames}
            if fixed_lags is not None:
                fixed = fixed_lags.get((x, y), {"conv": None, "blocks": []})
                lag_star, rho_peak, prominence = fixed["conv"], float("nan"), float("nan")
                blocks = [{"lag": lag, "rho": float("nan"), "prominence": float("nan"), "frames": 0}
                          for lag in fixed["blocks"]]
                source[y] = "fixed" if lag_star is not None else None
            elif row["is_pair"]:
                valid = _and(_and(~g_x, a_x), None if active is None else active[y])
                rho, counts = lag_curves(e_x, features[y]["energy"], valid, lags, params)
                with np.errstate(invalid="ignore"):
                    valid_blocks = np.isfinite(rho).any(1)
                    curve = np.nanmean(rho[valid_blocks], 0) if valid_blocks.any() else np.full(len(lags), np.nan)
                lag_star, rho_peak, prominence = peak(curve, lags, params)
                blocks = []
                for b in range(rho.shape[0]):
                    block_lag, block_peak, block_prominence = peak(rho[b], lags, params)
                    blocks.append({"lag": block_lag, "rho": block_peak, "prominence": block_prominence,
                                   "frames": int(counts[b, int(np.argmin(np.abs(lags)))])})
                rho_arrays[(x, y)] = rho
                source[y] = "search" if lag_star is not None else None
            else:
                lag_star, rho_peak, prominence, blocks = None, float("nan"), float("nan"), []
                source[y] = None
            row.update(lag=lag_star, rho=rho_peak, prominence=prominence, blocks=blocks)
            pair_rows[(x, y)] = row
            lag_of[y] = lag_star
            blocks_of[y] = blocks
        if lag_fallback:
            defined = [lag for y, lag in lag_of.items() if lag is not None and source[y] in ("search", "fixed")]
            fallback = lower_median(defined) if defined else 0
            for y in others:
                if lag_of[y] is None:
                    lag_of[y] = fallback
                    source[y] = "fallback"
        for y in others:
            pair_rows[(x, y)]["lag_used"] = lag_of[y]
            pair_rows[(x, y)]["lag_source"] = source[y]

        def block_lag(y: str, block: int):
            blocks = blocks_of[y]
            lag = blocks[block]["lag"] if block < len(blocks) else None
            return lag_of[y] if lag is None else lag

        cache: dict = {}

        def masks(y: str | None, kind: str, block: int):
            """(P, N) of x against y (None: pooled) at the conversation lags or the block lags of `block`."""
            key = (y, kind, block if kind == "blk" else -1)
            if key in cache:
                return cache[key]
            lag = (lambda z: lag_of[z]) if kind == "conv" else (lambda z: block_lag(z, block))
            if y is None:
                shifted = np.zeros(frames, bool)
                for z in others:
                    if lag(z) is not None:
                        shifted = shifted | cm.shift(o[z], lag(z))
                partner = _and(shifted & ~g_x, a_x)
                nobody = _and(~g_x, a_x) & ~cm.dilate(shifted, guard)
            else:
                lag_y = lag(y)
                if lag_y is None:
                    cache[key] = None
                    return None
                third = None
                for z in others:
                    if z != y and lag(z) is not None:
                        shifted_z = cm.shift(o[z], lag(z))
                        third = shifted_z if third is None else third | shifted_z
                shifted = cm.shift(o[y], lag_y)
                if third is None:
                    partner = _and(shifted & ~g_x, a_x)
                    nobody = _and(~g_x, a_x) & ~cm.dilate(shifted, guard)
                else:
                    partner = _and(shifted & ~g_x, a_x) & ~cm.dilate(third, guard)
                    nobody = _and(~g_x, a_x) & ~cm.dilate(shifted | third, guard)
            cache[key] = (partner, nobody)
            return cache[key]

        # ---- pair statistics at the conversation lags
        for y in others:
            row = pair_rows[(x, y)]
            lag = lag_of[y]
            if lag is not None:
                partner, nobody = masks(y, "conv", 0)
                stats = level_stats(e_x, own, partner, nobody, params)
                e_y = features[y].get("energy")  # None on restored tracks: the other tracks are not restored
                row.update(blr=stats["blr"], exc=stats["exc"], np=stats["np"], nn=stats["nn"], n_med=stats["n_med"],
                           erl=erl(e_x, e_y, partner, lag, params) if e_y is not None else float("nan"),
                           coherence=coherence(bp[x], bp[y], partner, lag, params) if bp is not None
                           else float("nan"))
            else:
                row.update(blr=float("nan"), exc=float("nan"), np=0, nn=0, n_med=float("nan"), erl=float("nan"),
                           coherence=float("nan"))
        # ---- track row
        zero_runs, clip_runs = features[x]["zero_runs"], features[x]["clip_runs"]
        track = {"x": x, "frames": frames, "samples": samples, "l_act": l_act,
                 "zero_runs": int(len(zero_runs)),
                 "zero_run_s": float(zero_runs[:, 1].sum() / cm.SR) if len(zero_runs) else 0.0,
                 "clip_runs": int(len(clip_runs)),
                 "zero_runs_in_words": count_in_interiors(zero_runs, spans[x], 0, samples)}
        zero_active = runs_in(zero_runs, a_x)
        clip_active = runs_in(clip_runs, a_x)
        zero_frames = cm.run_frames(zero_runs, frames) if (extra or pooled) else None
        if extra:
            active_frames = frames if a_x is None else int(a_x.sum())
            floor_mask = _and(~g_x, a_x) & ~zero_frames
            track.update(active_frames=active_frames, zero_runs_active=int(len(zero_active)),
                         clip_runs_active=int(len(clip_active)),
                         gate_runs_per_hour=len(zero_active) / (active_frames / 360_000) if active_frames
                         else float("nan"),
                         floor=float(np.percentile(e_x[floor_mask], 10)) if floor_mask.any() else float("nan"),
                         others=others, pairs=[y for y in others if pair_rows[(x, y)]["is_pair"]])
        if pooled:
            partner, nobody = masks(None, "conv", 0)
            stats = level_stats(e_x, own, partner, nobody, params)
            n_partner = int(partner.sum())
            track["pooled"] = {"blr": stats["blr"], "exc": stats["exc"], "np": stats["np"], "nn": stats["nn"],
                               "n_med": stats["n_med"],
                               "zshare": float((partner & zero_frames).sum() / n_partner) if n_partner
                               else float("nan")}
        track_rows[x] = track
        # ---- windows
        listed = range(n_windows) if windows is None else windows.get(x, [])
        for k in listed:
            region = slice(k * window_frames, (k + 1) * window_frames)
            block = (k * window_frames) // block_frames
            targets = [y for y in others if active is None or int(active[y][region].sum()) >= window_min_frames]
            if pooled:
                targets.append(None)
            for y in targets:
                w = {"offset_s": k * cm.WINDOW_S, "x": x, "y": "*" if y is None else y}
                if y is None:
                    n_med = track["pooled"]["n_med"]
                else:
                    n_med = pair_rows[(x, y)]["n_med"]
                    if not pair_rows[(x, y)]["is_pair"] and pooled:
                        n_med = track["pooled"]["n_med"]
                for kind in ("conv", "blk"):
                    if y is None:
                        lag = None
                    else:
                        lag = lag_of[y] if kind == "conv" else block_lag(y, block)
                    pair_masks = masks(y, kind, block)
                    if pair_masks is None:
                        w.update({f"lag_{kind}": None, f"blr_{kind}": float("nan"), f"exc_{kind}": float("nan"),
                                  f"np_{kind}": 0})
                        continue
                    partner, nobody = pair_masks
                    stats = level_stats(e_x[region], own[region], partner[region], nobody[region], params,
                                        l_act=l_act, n_med=n_med)
                    w.update({f"lag_{kind}": lag, f"blr_{kind}": stats["blr"], f"exc_{kind}": stats["exc"],
                              f"np_{kind}": stats["np"]})
                window_rows.append(w)
            lo, hi = k * cm.WINDOW_SAMPLES, (k + 1) * cm.WINDOW_SAMPLES
            integrity = {"offset_s": k * cm.WINDOW_S, "x": x,
                         "zr": count_in_interiors(zero_active, spans[x], lo, hi),
                         "clip": int(((clip_active[:, 0] >= lo) & (clip_active[:, 0] < hi)).sum())
                         if len(clip_active) else 0}
            if extra:
                inside = (zero_active[:, 0] >= lo) & (zero_active[:, 0] < hi) if len(zero_active) else None
                integrity["zero_s"] = float(zero_active[inside, 1].sum() / cm.SR) if inside is not None else 0.0
            integrity_rows.append(integrity)
    return {"tracks": track_rows, "pairs": pair_rows, "windows": window_rows, "integrity": integrity_rows,
            "rho": rho_arrays, "lags": lags, "frames": frames, "samples": samples}
