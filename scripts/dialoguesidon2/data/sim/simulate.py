"""Plan and render the clean 24 kHz tracks of the simulated conversations.

* plan (deterministic, no audio; `--dry-run` writes WORK/dry_run/plan_dry.{tsv,json} instead of
  WORK/clean/plan.{tsv,json}, so the record counts and the speech budget are seen before anything is rendered):
  1. Dev speakers: a seeded draw over the speakers of train-clean-100 and train-clean-360 reserves speakers until
     their usable speech is 2x what 72 records need (36 per style), and more, one at a time, until all 72 dev
     records can be built. Train records never use them.
  2. Records of each split are numbered n = 0, 1, ... in generation order; even n is overlap-style, odd n is
     conversation-style. Two different speakers of the split per record, drawn with probability proportional to
     the uses of their speech left in the record's style (no gender information is used: pairing is random).
     Reuse (PARAMS["reuse"], --max-uses, default 4): an utterance is used at most max_uses times in all and
     ceil(max_uses / 2) times per style, never twice in one record and never twice with the same partner
     (placements are drawn afresh); max_uses 1 is single use. Per style, a speaker's utterances are taken in
     (chapter, paragraph, sentence) order, continuing into the next chapter and after the last one from the first
     again (the next use). A draw that cannot be completed (a speaker runs out, or the planned masks give a
     channel below 5% activity or a silent 30 s quarter) is discarded and redrawn; 64 failed draws in a row end
     the split.
  3. Overlap-style: channel A from 0 s and channel B from U(0, 1) s, each a continuous run of its speaker's
     utterances with U(0.2, 1.0) s pauses between trimmed utterances, until 120 s.
  4. Conversation-style (the turn-taking table, --fit, default sim/turn_fit.json): the floor alternates; a turn
     takes utterances in order with within-turn pauses from the fitted table until its mask-domain length is
     closest to a turn length drawn from the table; the next turn starts at the turn end plus a drawn FTO (an
     overlap is limited so that neither the last spurt nor the next first spurt lies inside the other). Placement
     is in the mask domain, from each utterance's indexed VAD runs (pad 0.1 s), so the drawn values are what the
     S1 check measures. Interjections: Poisson with the fitted rate per second of turn, each the listener's
     shortest unused utterance whose mask is <= 1.45 s (unused in this record and style, as above), placed
     strictly inside one of the turn's spurts and >= 0.35 s from the listener's other speech; when none is left or
     none fits, it is skipped and counted (the rate is lowered). The record starts at a uniform offset into its
     first turn; utterances are cut at both 120 s edges.
  5. Levels: each utterance is scaled to -26 dBFS active level + U(-3, +3) dB; a track whose peak would exceed
     0.99 is scaled down as a whole (at render time).
  Train tars: record n goes to tar (n // 2) % 128, so every tar is 50/50 by style. Keys: base.record_key.
  plan.json and plan_dry.json carry `s1_preview`: the S1 check's conversation-style medians (union activity,
  overlap share, FTO, turn length) on the planned masks against the fit with the S1 tolerance, so a plan the S1
  check would fail is seen before rendering. It is a preview only: the planned masks are the indexed VAD runs, not
  the energy VAD of the rendered tracks.
* render (CPU): per record, both channels as mono 24 kHz PCM_16 FLACs of exactly 2,880,000 samples
  (WORK/clean/<split>/t<tar>/<key>_{a,b}.flac, silence is digital zero) from the index's PCM cache, then
  WORK/clean/done/<key>.json. Resumable per record; refuses a plan other than the one already rendered.

All randomness comes from numpy generators seeded with base.SEED, so the same index and fit give the same plan.

  python sim/simulate.py --work SIM_WORK plan --dry-run [--max-uses 4]
  python sim/simulate.py --work SIM_WORK plan [--max-uses 4]
  python sim/simulate.py --work SIM_WORK render --processes 64
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import base as sc  # noqa: E402
from sim import fit_turns  # noqa: E402
from sim import index_libritts as il  # noqa: E402
from sim.base import cm  # noqa: E402

PLAN_COLUMNS = (["key", "split", "n", "style", "tar", "speaker_a", "speaker_b", "chapters_a", "chapters_b"]
                + [f"{f}_{c}" for c in ("a", "b") for f in ("utterances", "offsets", "gains", "kinds")]
                + ["plan_activity_a", "plan_activity_b", "plan_activity_union", "plan_overlap_share",
                   "interjections", "shortfall_utterance", "shortfall_room", "start_s"])


class Utterance:
    """One usable utterance of the index with its placement geometry (seconds)."""

    def __init__(self, i: int, row: dict):
        pad = sc.PARAMS["vad"]["pad_s"]
        self.i = i
        self.utt, self.speaker, self.chapter, self.subset = row["utt"], row["speaker"], row["chapter"], row["subset"]
        self.tar, self.pcm_offset, self.length = row["tar"], row["pcm_offset"], row["length"]
        self.active_db = row["active_db"]
        self.runs = sc.parse_runs(row["runs"])
        self.dur = self.length / sc.SR
        self.first_on = self.runs[0, 0] / sc.FPS - pad
        self.mask_len = (self.runs[-1, 1] - self.runs[0, 0]) / sc.FPS + 2 * pad
        self.first_run = (self.runs[0, 1] - self.runs[0, 0]) / sc.FPS + 2 * pad
        self.last_run = (self.runs[-1, 1] - self.runs[-1, 0]) / sc.FPS + 2 * pad

    def spurts(self, t: float) -> list[tuple[float, float]]:
        """Mask intervals (padded runs) of the utterance with its trimmed start at t."""
        pad = sc.PARAMS["vad"]["pad_s"]
        return [(t + a / sc.FPS - pad, t + b / sc.FPS + pad) for a, b in self.runs]

    def spans(self, t: float) -> list[tuple[float, float]]:
        return [(t + a / sc.FPS, t + b / sc.FPS) for a, b in self.runs]


def load_utterances(index: Path) -> list[Utterance]:
    """Usable utterances in index order (speaker, chapter, paragraph, sentence)."""
    rows = [r for r in il.load_index(index) if r["usable"]]
    rows.sort(key=il.sort_key)
    return [Utterance(i, r) for i, r in enumerate(rows)]


class SpeakerPool:
    """Utterances per speaker and their uses: the ordered stream, the short ones by mask length, and per style a
    stream cursor, the speech a record can still take (`distinct`: utterances with a use left in the style) and the
    uses of speech left (`capacity`: duration x uses left). With max_uses 1 both are the unused speech."""

    def __init__(self, utts: list[Utterance], speakers, max_uses: int | None = None):
        limit = sc.PARAMS["conversation"]["interjection_max_s"] - 0.05
        self.utts = utts
        self.max_uses = int(sc.PARAMS["reuse"]["max_uses"] if max_uses is None else max_uses)
        if self.max_uses < 1:
            raise ValueError(f"max_uses must be at least 1, not {self.max_uses}")
        self.per_style = -(-self.max_uses // 2)
        self.speakers = sorted(speakers, key=lambda s: (int(s) if s.isdigit() else s))
        chosen = set(self.speakers)
        self.stream = collections.defaultdict(list)
        for u in utts:
            if u.speaker in chosen:
                self.stream[u.speaker].append(u.i)
        self.short = {s: sorted((i for i in self.stream[s] if utts[i].mask_len <= limit),
                                key=lambda i: (utts[i].mask_len, i)) for s in self.speakers}
        self.cursor = {style: {s: 0 for s in self.speakers} for style in sc.STYLES}
        self.total = bytearray(len(utts))
        self.uses = {style: bytearray(len(utts)) for style in sc.STYLES}
        self.partners: dict[int, set[str]] = collections.defaultdict(set)
        self.capacity = {style: {s: sum(utts[i].dur * self.left(i, style) for i in self.stream[s])
                                 for s in self.speakers} for style in sc.STYLES}
        self.distinct = {style: {s: sum(utts[i].dur for i in self.stream[s]) for s in self.speakers}
                         for style in sc.STYLES}

    def left(self, i: int, style: str) -> int:
        """Further uses of utterance i in a style."""
        return max(0, min(self.per_style - self.uses[style][i], self.max_uses - self.total[i]))

    def free(self, i: int, style: str, partner: str) -> bool:
        return self.left(i, style) > 0 and partner not in self.partners.get(i, ())

    def use(self, i: int, style: str, partner: str) -> None:
        before = {s: self.left(i, s) for s in sc.STYLES}
        self.uses[style][i] += 1
        self.total[i] += 1
        self.partners[i].add(partner)
        u = self.utts[i]
        for s in sc.STYLES:
            after = self.left(i, s)
            self.capacity[s][u.speaker] += u.dur * (after - before[s])
            if before[s] and not after:
                self.distinct[s][u.speaker] -= u.dur

    def eligible(self, minimum: float, style: str) -> list[str]:
        return [s for s in self.speakers if self.distinct[style][s] >= minimum]

    def speech_left(self) -> float:
        """Speech of the utterances with a use left (unused speech with max_uses 1)."""
        return sum(self.utts[i].dur for s in self.speakers for i in self.stream[s] if self.total[i] < self.max_uses)


class Session:
    """Tentative use of a pool while one record of a style and speaker pair is built; commit() makes it final."""

    def __init__(self, pool: SpeakerPool, style: str, pair):
        self.pool = pool
        self.style = style
        self.partner = {pair[0]: pair[1], pair[1]: pair[0]}
        self.taken: list[int] = []
        self.taken_set: set[int] = set()
        self.cursor: dict[str, int] = {}

    def used(self, i: int) -> bool:
        """Not available to this record: no use left in its style, used with this partner, or taken already."""
        return i in self.taken_set or not self.pool.free(i, self.style, self.partner[self.pool.utts[i].speaker])

    def peek(self, speaker: str) -> int | None:
        """The next available utterance in stream order from the style's cursor, once around the stream."""
        stream = self.pool.stream[speaker]
        start = self.cursor.get(speaker, self.pool.cursor[self.style][speaker])
        for step in range(len(stream)):
            c = (start + step) % len(stream)
            if not self.used(stream[c]):
                self.cursor[speaker] = c
                return stream[c]
        self.cursor[speaker] = start
        return None

    def take(self, i: int) -> None:
        self.taken.append(i)
        self.taken_set.add(i)

    def untake(self, i: int) -> None:
        self.taken.remove(i)
        self.taken_set.discard(i)

    def shortest(self, speaker: str, exclude: int | None = None) -> int | None:
        return next((i for i in self.pool.short[speaker] if not self.used(i) and i != exclude), None)

    def commit(self) -> None:
        pool = self.pool
        for i in self.taken:
            pool.use(i, self.style, self.partner[pool.utts[i].speaker])
        pool.cursor[self.style].update(self.cursor)
        for speaker in {pool.utts[i].speaker for i in self.taken}:
            pool.short[speaker] = [i for i in pool.short[speaker] if pool.total[i] < pool.max_uses]


# ---- record builders ----------------------------------------------------------------------------------------------

def build_overlap(session: Session, pair, rng) -> dict | None:
    cfg = sc.PARAMS["overlap"]
    utts = session.pool.utts
    placements = {0: [], 1: []}
    for ch, speaker in enumerate(pair):
        t = 0.0 if ch == 0 else float(rng.uniform(*cfg["b_start_s"]))
        while t < sc.TRACK_S:
            i = session.peek(speaker)
            if i is None:
                return None
            session.take(i)
            placements[ch].append((i, t, "o"))
            t += utts[i].dur + float(rng.uniform(*cfg["pause_s"]))
    return {"placements": placements, "start_s": 0.0, "interjections": 0, "shortfall_utterance": 0,
            "shortfall_room": 0}


def _clear(interval, spurts, guard) -> bool:
    lo, hi = interval
    return all(hi <= a - guard or lo >= b + guard for a, b in spurts)


def build_conversation(session: Session, pair, rng, fit: dict) -> dict | None:
    cfg = sc.PARAMS["conversation"]
    tables, rate = fit["tables"], float(fit["rates"]["interjections_per_turn_s"])
    utts = session.pool.utts
    placements = {0: [], 1: []}
    spurts = {0: [], 1: []}
    stats = {"interjections": 0, "shortfall_utterance": 0, "shortfall_room": 0}
    cur = int(rng.integers(2))
    m_start, start, end = 0.0, None, None
    while True:
        speaker = pair[cur]
        target = max(cfg["min_turn_s"], sc.draw(tables["turn_s"], rng))
        turn, m_end = [], m_start
        while True:
            i = session.peek(speaker)
            if i is None:
                return None
            u = utts[i]
            s_mask = m_start if not turn else m_end + sc.draw(tables["pause_s"], rng)
            if turn and end is not None and s_mask >= end:
                break
            e_mask = s_mask + u.mask_len
            if turn and abs(e_mask - m_start - target) > abs(m_end - m_start - target):
                break
            session.take(i)
            turn.append((i, s_mask - u.first_on))
            m_end = e_mask
            if e_mask - m_start >= target:
                break
        for i, t in turn:
            placements[cur].append((i, t, "t"))
            spurts[cur] += utts[i].spurts(t)
        if start is None:
            start = m_start + float(rng.uniform(0, m_end - m_start))
            end = start + sc.TRACK_S
        done = m_end >= end
        listener = 1 - cur
        next_start, j = None, None
        if not done:
            j = session.peek(pair[listener])
            if j is None:
                return None
            fto = sc.draw(tables["fto_s"], rng)
            fto = max(fto, -(min(utts[turn[-1][0]].last_run, utts[j].first_run) - cfg["fto_margin_s"]))
            next_start = m_end + fto
            if spurts[listener]:
                next_start = max(next_start, max(b for _, b in spurts[listener]) + cfg["guard_s"])
        # Interjections of the listener inside this turn's spurts.
        count = int(rng.poisson(rate * (m_end - m_start))) if rate > 0 else 0
        own = [(a + cfg["inside_margin_s"], b - cfg["inside_margin_s"]) for i, t in turn for a, b in utts[i].spurts(t)]
        for _ in range(count):
            y = session.shortest(pair[listener], exclude=j)  # j opens the listener's next turn
            if y is None:
                stats["shortfall_utterance"] += 1
                continue
            length = utts[y].mask_len
            room = [(a, b - length) for a, b in own if b - length > a]
            placed = False
            for _ in range(8 if room else 0):
                weights = np.array([b - a for a, b in room])
                a, b = room[int(rng.choice(len(room), p=weights / weights.sum()))]
                p = float(rng.uniform(a, b))
                if next_start is not None and p + length > next_start - cfg["guard_s"]:
                    continue
                if not _clear((p, p + length), spurts[listener], cfg["guard_s"]):
                    continue
                session.take(y)
                t = p - utts[y].first_on
                placements[listener].append((y, t, "i"))
                spurts[listener] += utts[y].spurts(t)
                stats["interjections"] += 1
                placed = True
                break
            if not placed:
                stats["shortfall_room"] += 1
        if done:
            break
        m_start, cur = next_start, listener
    return {"placements": placements, "start_s": start, **stats}


def planned_masks(utts: list[Utterance], placements: dict, start: float) -> tuple[np.ndarray, np.ndarray]:
    masks = []
    for ch in (0, 1):
        spans = [(a - start, b - start) for i, t, _ in placements[ch] for a, b in utts[i].spans(t)]
        masks.append(sc.mask_from_spans(spans))
    return masks[0], masks[1]


def acceptable(a: np.ndarray, b: np.ndarray) -> bool:
    minimum = sc.PARAMS["record"]["min_activity"]
    return a.mean() >= minimum and b.mean() >= minimum and all(sc.quarters_active(a | b))


def record_row(split: str, n: int, pair, built: dict, utts: list[Utterance], rng, tars: int) -> dict:
    level = sc.PARAMS["level"]
    start = built["start_s"]
    row = {"key": sc.record_key(split, n, pair[0], pair[1]), "split": split, "n": n, "style": sc.style_of(n),
           "tar": sc.tar_of(n, tars) if split == "train" else 0, "speaker_a": pair[0], "speaker_b": pair[1],
           "start_s": round(start, 6)}
    for ch, c in ((0, "a"), (1, "b")):
        chosen = []
        for i, t, kind in sorted(built["placements"][ch], key=lambda p: p[1]):
            offset = int(round((t - start) * sc.SR))
            if offset >= sc.TRACK_SAMPLES or offset + utts[i].length <= 0:
                continue
            gain = level["target_dbfs"] + float(rng.uniform(-level["jitter_db"], level["jitter_db"])) - utts[i].active_db
            chosen.append((i, offset, round(gain, 3), kind))
        row[f"utterances_{c}"] = ",".join(utts[i].utt for i, *_ in chosen)
        row[f"offsets_{c}"] = ",".join(str(o) for _, o, _, _ in chosen)
        row[f"gains_{c}"] = ",".join(f"{g:.3f}" for _, _, g, _ in chosen)
        row[f"kinds_{c}"] = "".join(k for *_, k in chosen)
        row[f"chapters_{c}"] = ",".join(dict.fromkeys(utts[i].chapter for i, *_ in chosen))
    a, b = planned_masks(utts, built["placements"], start)
    act = sc.activity(a, b)
    row.update({f"plan_{k}": v for k, v in act.items()})
    row["interjections"] = (row["kinds_a"] + row["kinds_b"]).count("i")  # inside the record
    row.update({k: built[k] for k in ("shortfall_utterance", "shortfall_room")})
    return row


def generate(split: str, pool: SpeakerPool, target: int, rng, fit: dict, tars: int, utts) -> tuple[list[dict], dict]:
    pairing = sc.PARAMS["pairing"]
    rows, failures, stop = [], collections.Counter(), None
    builders = {"overlap": build_overlap, "conversation": lambda s, p, r: build_conversation(s, p, r, fit)}
    while len(rows) < target and stop is None:
        n = len(rows)
        style = sc.style_of(n)
        for _ in range(pairing["attempts"]):
            speakers = pool.eligible(pairing["min_remaining_s"][style], style)
            if len(speakers) < 2:
                stop = f"fewer than 2 speakers with >= {pairing['min_remaining_s'][style]} s left for {style}"
                break
            weights = np.array([pool.capacity[style][s] for s in speakers])
            first, second = rng.choice(len(speakers), size=2, replace=False, p=weights / weights.sum())
            pair = (speakers[int(first)], speakers[int(second)])
            session = Session(pool, style, pair)
            built = builders[style](session, pair, rng)
            if built is None:
                failures[f"{style}: speaker ran out"] += 1
                continue
            if not acceptable(*planned_masks(utts, built["placements"], built["start_s"])):
                failures[f"{style}: activity or quarter rule"] += 1
                continue
            session.commit()
            rows.append(record_row(split, n, pair, built, utts, rng, tars))
            break
        else:
            stop = f"{pairing['attempts']} failed draws in a row for {style} record {n}"
    return rows, {"records": len(rows), "target": target, "stop": stop or "target reached",
                  "failed_draws": dict(failures)}


def dev_candidates(utts: list[Utterance], rng) -> tuple[list[str], int, dict]:
    """Speakers of train-clean-100/360 only, in a seeded random order, and how many of them hold 2x the speech of
    the dev records (at least 2)."""
    split = sc.PARAMS["split"]
    subsets, speech = collections.defaultdict(set), collections.Counter()
    for u in utts:
        subsets[u.speaker].add(u.subset)
        speech[u.speaker] += u.dur
    candidates = sorted((s for s, v in subsets.items() if v <= set(split["dev_subsets"])),
                        key=lambda s: (int(s) if s.isdigit() else s))
    order = [candidates[int(k)] for k in rng.permutation(len(candidates))]
    per_record = split["speech_per_record_s"]
    need = split["dev_records"] / 2 * (per_record["overlap"] + per_record["conversation"]) * split["dev_reserve_factor"]
    count, total = 0, 0.0
    while count < len(order) and (total < need or count < 2):
        total += speech[order[count]]
        count += 1
    return order, count, {"candidates": len(candidates), "need_s": need, "speech": speech}


def plan(utts: list[Utterance], fit: dict, seed: int = sc.SEED) -> tuple[list[dict], dict]:
    """Dev first: the first k speakers of the seeded order, k growing from the speech estimate until the dev records
    can all be built (or the candidates run out); then train on every other speaker."""
    split_cfg = sc.PARAMS["split"]
    order, count, reservation = dev_candidates(utts, np.random.default_rng([seed, 0]))
    speech = reservation.pop("speech")
    while True:
        dev = order[:count]
        pool = SpeakerPool(utts, dev)
        dev_rows, dev_report = generate("dev", pool, split_cfg["dev_records"], np.random.default_rng([seed, 1]), fit,
                                        split_cfg["train_tars"], utts)
        if len(dev_rows) >= split_cfg["dev_records"] or count >= len(order):
            break
        count += 1
    dev_report["speech_available_s"] = sum(speech[s] for s in dev)
    dev_report["speech_left_s"] = pool.speech_left()
    reservation.update(speakers=len(dev), speech_s=sum(speech[s] for s in dev))
    dev_set = set(dev)
    train = [s for s in sorted({u.speaker for u in utts}, key=lambda s: (int(s) if s.isdigit() else s))
             if s not in dev_set]
    pool = SpeakerPool(utts, train)
    train_rows, train_report = generate("train", pool, split_cfg["train_records"], np.random.default_rng([seed, 2]),
                                        fit, split_cfg["train_tars"], utts)
    train_report["speech_available_s"] = sum(speech[s] for s in train)
    train_report["speech_left_s"] = pool.speech_left()
    return dev_rows + train_rows, {"dev_reservation": reservation, "dev": dev_report, "train": train_report}


def summarize(rows: list[dict], info: dict, utts: list[Utterance]) -> dict:
    by_id = {u.utt: u for u in utts}
    summary = {"splits": {}}
    for split in ("train", "dev"):
        part = [r for r in rows if r["split"] == split]
        styles = collections.Counter(r["style"] for r in part)
        used = [u for r in part for c in ("a", "b") for u in r[f"utterances_{c}"].split(",") if u]
        uses = collections.Counter(collections.Counter(used).values())
        summary["splits"][split] = {
            "records": len(part), "hours": len(part) * sc.TRACK_S / 3600, "styles": dict(styles),
            "speakers": len({r[f"speaker_{c}"] for r in part for c in ("a", "b")}),
            "speech_used_h": sum(by_id[u].dur for u in used) / 3600,  # every use counted
            "speech_distinct_h": sum(by_id[u].dur for u in set(used)) / 3600,
            "utterances_by_uses": {str(k): uses[k] for k in sorted(uses)},
            "interjections": sum(r["interjections"] for r in part),
            "shortfall_utterance": sum(r["shortfall_utterance"] for r in part),
            "shortfall_room": sum(r["shortfall_room"] for r in part),
            "plan_activity": {style: {k: sc.pct([r[f"plan_{k}"] for r in part if r["style"] == style])
                                      for k in ("activity_a", "activity_b", "activity_union", "overlap_share")}
                              for style in sc.STYLES},
            **info.get(split, {})}
    summary["dev_reservation"] = info["dev_reservation"]
    summary["speech_total_h"] = sum(u.dur for u in utts) / 3600
    summary["max_uses"] = sc.PARAMS["reuse"]["max_uses"]
    return summary


def s1_preview(rows: list[dict], utts: list[Utterance], fit: dict) -> dict:
    """The S1 check's conversation-style statistics (qc.s1_checks) on the planned masks of every split, segmented
    as the check segments the rendered tracks, against the fit's medians with the check's tolerance."""
    cfg = sc.PARAMS["s1"]
    by_id = {u.utt: u for u in utts}
    measured = {"activity_union": [], "overlap_share": [], "fto_s": [], "turn_s": []}
    for r in rows:
        if r["style"] != "conversation":
            continue
        masks = [sc.mask_from_spans([span for u, o, _, _ in channel(r, c) for span in by_id[u].spans(o / sc.SR)])
                 for c in ("a", "b")]
        act, seg = sc.activity(*masks), sc.segment(*masks)
        measured["activity_union"].append(act["activity_union"])
        measured["overlap_share"].append(act["overlap_share"])
        measured["fto_s"] += seg["fto_s"]
        measured["turn_s"] += seg["turn_s"]
    preview = {"source": "planned masks (indexed VAD runs), before rendering and the energy VAD of the S1 check"}
    for name, values in measured.items():
        plan_pct, fit_pct = sc.pct(values), {p: fit["summary"][name][p] for p in ("p10", "p50", "p90")}
        floor = cfg["frame_floor_s"] if name.endswith("_s") else 0.0
        preview[name] = {"plan": plan_pct, "fit": fit_pct, "n": len(values),
                         "pass": sc.within(plan_pct["p50"], fit_pct["p50"], cfg["tolerance"], floor),
                         "report_only": name in cfg.get("report_only", ())}
    preview["pass"] = all(preview[name]["pass"] for name in measured if not preview[name]["report_only"])
    return preview


def plan_text(rows: list[dict]) -> str:
    lines = ["\t".join(PLAN_COLUMNS)] + ["\t".join(cm.format_cell(r.get(c)) for c in PLAN_COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


def read_plan(path: Path) -> list[dict]:
    rows = cm.read_tsv(path)
    for r in rows:
        for k in ("n", "tar", "interjections", "shortfall_utterance", "shortfall_room"):
            r[k] = int(r[k])
    return rows


def channel(row: dict, c: str) -> list[tuple[str, int, float, str]]:
    """(utterance id, start sample, gain dB, kind) of one channel of a plan row."""
    ids = [u for u in row[f"utterances_{c}"].split(",") if u]
    offsets = [int(v) for v in row[f"offsets_{c}"].split(",") if v]
    gains = [float(v) for v in row[f"gains_{c}"].split(",") if v]
    kinds = row[f"kinds_{c}"] if row[f"kinds_{c}"] != "NA" else ""
    return list(zip(ids, offsets, gains, kinds))


def plan_max_uses(record: dict) -> int:
    """The reuse cap a plan record (plan.json, plan_dry.json) was planned with."""
    return int(record["params"]["reuse"]["max_uses"])


def plan_command(index: Path, fit_path: Path, out: Path, dry_run: bool, max_uses: int) -> int:
    if max_uses < 1:
        raise SystemExit("--max-uses must be at least 1")
    sc.PARAMS["reuse"]["max_uses"] = max_uses  # recorded with the parameters in plan.json
    utts = load_utterances(index)
    fit = fit_turns.load_fit(fit_path)
    rows, info = plan(utts, fit)
    text = plan_text(rows)
    digest = hashlib.sha256(text.encode()).hexdigest()
    summary = summarize(rows, info, utts)
    preview = s1_preview(rows, utts, fit)
    record = cm.run_info("plan" + ("-dry" if dry_run else ""), sc.PARAMS, plan_hash=digest, dry_run=dry_run,
                         max_uses=max_uses,
                         index={"path": str(index), "sha256": cm.file_sha256(index)},
                         fit={"path": str(fit_path), "sha256": cm.file_sha256(fit_path)}, summary=summary,
                         s1_preview=preview)
    name = "plan_dry" if dry_run else "plan"
    if not dry_run and (out / "plan.json").exists():
        old = cm.load_json(out / "plan.json")["plan_hash"]
        if old != digest and any((out / "done").glob("*.json")):
            raise SystemExit(f"{out} holds tracks rendered from plan {old[:12]}; write the new plan elsewhere")
    cm.write_atomic(out / f"{name}.tsv", text)
    cm.write_json(out / f"{name}.json", record)
    print(json.dumps({"plan_hash": digest, "splits": {s: {k: v for k, v in d.items() if k != "plan_activity"}
                                                      for s, d in summary["splits"].items()},
                      "s1_preview": {k: (v if k in ("pass", "source") else {"plan_p50": v["plan"]["p50"],
                                                                          "fit_p50": v["fit"]["p50"],
                                                                          "pass": v["pass"]})
                                     for k, v in preview.items()}}, indent=1), flush=True)
    return 0


# ---- render -----------------------------------------------------------------------------------------------------

_PCM: dict = {}


def pcm(index_dir: Path, subset: str, tar: int) -> np.ndarray:
    key = (str(index_dir), subset, tar)
    if key not in _PCM:
        _PCM[key] = np.load(index_dir / "pcm" / subset / f"data-{tar:05d}.npy", mmap_mode="r")
    return _PCM[key]


def render_track(entries, sources: dict, index_dir: Path) -> tuple[np.ndarray, float, float]:
    """int16 track, the whole-track scale and the peak before it. entries: (utt, start, gain dB, kind)."""
    x = np.zeros(sc.TRACK_SAMPLES, np.float32)
    for utt, start, gain, _ in entries:
        subset, tar, offset, length = sources[utt]
        data = np.asarray(pcm(index_dir, subset, tar)[offset:offset + length], np.float32)
        data *= np.float32(10 ** (gain / 20) / 32768.0)
        lo, hi = max(0, start), min(sc.TRACK_SAMPLES, start + length)
        if hi > lo:
            x[lo:hi] += data[lo - start:hi - start]
    peak = float(np.abs(x).max())
    scale = 1.0 if peak <= sc.PARAMS["level"]["peak_max"] else sc.PARAMS["level"]["peak_max"] / peak
    if scale != 1.0:
        x *= np.float32(scale)
    return np.round(x * 32768.0).clip(-32768, 32767).astype(np.int16), scale, peak


def flac_bytes_24k(x: np.ndarray) -> bytes:
    import soundfile as sf
    stream = io.BytesIO()
    sf.write(stream, x, sc.SR, format="FLAC", subtype="PCM_16")
    return stream.getvalue()


def render_record(job) -> str:
    row, sources, index_dir, clean, digest = job
    marker = clean / "done" / f"{row['key']}.json"
    paths = {c: clean / sc.track_relpath(row["split"], row["tar"], row["key"], c) for c in ("a", "b")}
    if marker.exists() and cm.load_json(marker).get("plan_hash") == digest \
            and not any(sc.clean_problem(p) for p in paths.values()):
        return "skipped"
    done = {"key": row["key"], "plan_hash": digest}
    for c in ("a", "b"):
        x, scale, peak = render_track(channel(row, c), sources, index_dir)
        cm.write_atomic(paths[c], flac_bytes_24k(x))
        done[f"scale_{c}"], done[f"peak_{c}"] = scale, peak
    cm.write_json(marker, done)
    return "done"


def _safe_render(job):
    try:
        return render_record(job)
    except Exception as error:
        return f"failed: {type(error).__name__}: {error}"


def render_command(index: Path, clean: Path, processes: int) -> int:
    if not (clean / "plan.json").exists():
        raise SystemExit(f"No plan in {clean}; run `simulate.py plan` first")
    digest = cm.load_json(clean / "plan.json")["plan_hash"]
    if hashlib.sha256((clean / "plan.tsv").read_bytes()).hexdigest() != digest:
        raise SystemExit(f"{clean / 'plan.tsv'} does not match plan.json")
    rows = read_plan(clean / "plan.tsv")
    needed = {u for r in rows for c in ("a", "b") for u in r[f"utterances_{c}"].split(",") if u}
    sources = {r["utt"]: (r["subset"], r["tar"], r["pcm_offset"], r["length"])
               for r in il.load_index(index / "index.tsv") if r["usable"] and r["utt"] in needed}
    missing = needed - set(sources)
    if missing:
        raise SystemExit(f"{len(missing)} planned utterances are not in the index (e.g. {sorted(missing)[:3]})")
    jobs = [(r, {u: sources[u] for c in ("a", "b") for u in r[f"utterances_{c}"].split(",") if u}, index, clean,
             digest) for r in rows]
    status, failed = collections.Counter(), []
    with Pool(processes) as pool:
        for job, result in zip(jobs, pool.imap(_safe_render, jobs, chunksize=4)):
            status[result.split(":")[0]] += 1
            if result.startswith("failed"):
                failed.append((job[0]["key"], result))
                print(f"{job[0]['key']}: {result}", flush=True)
    scales = [cm.load_json(clean / "done" / f"{r['key']}.json") for r in rows
              if (clean / "done" / f"{r['key']}.json").exists()]
    record = cm.run_info("render", sc.PARAMS, plan_hash=digest, records=len(rows), status=dict(status),
                         failures=failed[:50], scaled_tracks=sum(d.get(f"scale_{c}", 1.0) < 1.0 for d in scales
                                                                  for c in ("a", "b")))
    cm.write_json(clean / "render.json", {**record, "complete": not failed})
    print(json.dumps({"records": len(rows), **status, "scaled_tracks": record["scaled_tracks"]}), flush=True)
    return 0 if not failed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, required=True, help="the simulation's work directory (SIM_WORK)")
    parser.add_argument("--index", type=Path, default=None, help="default: WORK/index")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--dry-run", action="store_true", help="write WORK/dry_run/plan_dry.{tsv,json} and render nothing")
    p.add_argument("--fit", type=Path, default=fit_turns.SHIPPED, help="turn-taking table (default: the shipped one)")
    p.add_argument("--max-uses", type=int, default=sc.PARAMS["reuse"]["max_uses"],
                   help="uses per utterance in all (ceil(N / 2) per style; the released data used 4)")
    r = sub.add_parser("render")
    r.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)
    index = args.index or args.work / "index"
    if args.command == "plan":
        out = args.work / "dry_run" if args.dry_run else args.work / "clean"
        return plan_command(index / "index.tsv", args.fit, out, args.dry_run, args.max_uses)
    return render_command(index, args.work / "clean", args.processes)


if __name__ == "__main__":
    sys.exit(main())
