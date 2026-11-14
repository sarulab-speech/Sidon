"""DNSMOS P.835 OVR of the raw Basis pair tracks in their S0 windows (optional stage).

The released thresholds (basis/thresholds.json) switch both DNSMOS rules off (dnsmos_track_min and
dnsmos_window_min are null), so these scores did not select any window of the released data. The stage is kept so
that the rules can be switched on; cut.py requires its output only when one of the two rules is set.

Scoring: the primary DNSMOS P.835 model (sig_bak_ovr.onnx from Microsoft's DNS Challenge repository, DNSMOS/DNSMOS/)
run with onnxruntime on raw 9.01 s clips, followed by the non-personalised polynomial mapping of the OVR score
(the P.808 model is not used). Per window of a pair track (S0-kept windows, and the controls' windows): the 120 s
window of the raw 48 kHz track resampled to 16 kHz with librosa.resample (default resampler), and the 37 clips of
9.01 s starting at t0 + 3j (j = 0..36). Clips whose echo-word-free speech mask O_x covers less than 10% are not
scored; the coverage of every clip is stored, so the coverage rule (0.10, 0.25 or 0.50) is applied in `summarize`.

* run (resumable per segment): dnsmos/<segment>.json. Needs onnxruntime and librosa.
* summarize (tables only): q_windows.tsv (window median over the clips that pass the coverage rule; NaN below 3
  clips) and q_tracks.tsv (a track's median over the scored clips of its windows).

  python scripts/dialoguesidon2/data/basis/dnsmos_windows.py --root BASIS_ROOT --screen WORK/screen run \\
      --model DNSMOS/sig_bak_ovr.onnx --processes 64
  python scripts/dialoguesidon2/data/basis/dnsmos_windows.py --root BASIS_ROOT --screen WORK/screen summarize \\
      --coverage 0.25
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

# sha256 of the sig_bak_ovr.onnx file that scored the released data (pass it to --sha256 to check a download).
MODEL_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
PARAMS = {"version": 1, "model": "DNSMOS P.835 sig_bak_ovr.onnx", "model_sha256": MODEL_SHA256,
          "personalized": False, "sr": 16_000, "clip_s": 9.01, "step_s": 3, "clips_per_window": 37,
          "resample": "librosa.resample(orig_sr=48000, target_sr=16000) of the 120 s window",
          "min_coverage_scored": 0.10, "coverage_rules": [0.10, 0.25, 0.50], "min_clips": 3,
          "thresholds": {"track": 2.5, "window": 2.3}}
CLIP_SAMPLES = int(PARAMS["clip_s"] * PARAMS["sr"])  # 144,160 samples
CLIP_FRAMES = int(round(PARAMS["clip_s"] * cm.FPS))  # 901 mask frames
# The non-personalised polynomial mapping of the raw OVR output.
OVR_POLY = (-0.06766283, 1.11546468, 0.04602535)

MODEL: Path | None = None  # set by `run` and in every worker process (set_model)
_session = None


def set_model(path) -> None:
    global MODEL, _session
    MODEL, _session = Path(path), None


def session():
    """The ONNX session of the primary model (one intra-op thread per process)."""
    global _session
    if _session is None:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        _session = ort.InferenceSession(str(MODEL), sess_options=options)
    return _session


def clip_starts(offset_s: int) -> list[float]:
    return [offset_s + PARAMS["step_s"] * j for j in range(PARAMS["clips_per_window"])]


def clip_coverage(o: np.ndarray, offset_s: int) -> list[float]:
    """Share of each clip's 901 frames covered by the speech mask."""
    result = []
    for start in clip_starts(offset_s):
        first = int(round(start * cm.FPS))
        segment = o[first:first + CLIP_FRAMES]
        result.append(float(segment.mean()) if len(segment) else 0.0)
    return result


def score_clips(audio16: np.ndarray, coverage: list[float], minimum: float, score=None) -> list[float | None]:
    """OVR of every clip whose coverage is >= minimum (None otherwise), from a 16 kHz window."""
    score = score or primary_ovr
    result = []
    for j, cov in enumerate(coverage):
        start = PARAMS["step_s"] * j * PARAMS["sr"]
        clip = audio16[start:start + CLIP_SAMPLES]
        result.append(score(clip) if cov >= minimum and len(clip) == CLIP_SAMPLES else None)
    return result


def primary_ovr(clip: np.ndarray) -> float:
    features = np.asarray(clip, np.float32)[np.newaxis, :]
    _, _, ovr_raw = session().run(None, {"input_1": features})[0][0]
    return float(np.poly1d(OVR_POLY)(ovr_raw))


def window_audio16(path: Path, offset_s: int) -> np.ndarray:
    import librosa
    import soundfile as sf
    audio, sr = sf.read(str(path), start=offset_s * cm.SR, frames=cm.WINDOW_SAMPLES, dtype="float64")
    if sr != cm.SR or len(audio) != cm.WINDOW_SAMPLES:
        raise ValueError(f"{path} at {offset_s} s: {len(audio)} frames at {sr} Hz")
    return librosa.resample(audio, orig_sr=cm.SR, target_sr=PARAMS["sr"])


def q_segment(job) -> str:
    root, screen, segment, task, location, tag = job
    sid = segment["segment_id"]
    target = screen / "dnsmos" / f"{sid}.json"
    if target.exists():
        return "skipped"
    row = bs.read_segment(root, *location)
    masks = bs.segment_masks(row, segment["samples"] // cm.HOP)
    result = {}
    for x in task["tested"]:
        result[x] = {}
        for offset in task["windows"].get(x, {}):
            offset = int(offset)
            coverage = clip_coverage(masks["o"][x], offset)
            audio16 = window_audio16(root / segment["tracks"][x], offset)
            result[x][str(offset)] = {"coverage": coverage,
                                      "ovr": score_clips(audio16, coverage, PARAMS["min_coverage_scored"])}
    cm.write_atomic(target, json.dumps({"params_hash": tag, "segment_id": sid, "tracks": result}))
    return "done"


def _safe(job):
    try:
        return q_segment(job)
    except Exception as error:
        return f"failed: {type(error).__name__}: {error}"


def command_run(args) -> int:
    set_model(args.model)
    if args.sha256 and cm.file_sha256(MODEL) != args.sha256:
        raise SystemExit(f"{MODEL} has another sha256 than {args.sha256}")
    segments = tb.load_segments(args.screen)
    tasks = tb.segment_tasks(segments, tb.load_windows(args.screen))
    ready = tb.ready_segments(args.screen)
    index = bs.segment_index(args.root)
    tag = cm.params_hash(PARAMS)
    jobs = [(args.root, args.screen, s, tasks[s["segment_id"]], index[s["segment_id"]], tag) for s in segments
            if s["segment_id"] in tasks and s["segment_id"] in ready]
    jobs.sort(key=lambda j: -sum(len(v) for v in j[3]["windows"].values()))
    cm.write_json(args.screen / "q_run_start.json", cm.run_info("Q", PARAMS, params_hash=tag, segments=len(jobs),
                                                                model=str(MODEL)))
    status, failures = collections.Counter(), []
    with Pool(args.processes, maxtasksperchild=8, initializer=set_model, initargs=(str(MODEL),)) as pool:
        for job, result in zip(jobs, pool.imap(_safe, jobs)):
            status[result.split(":")[0]] += 1
            if result.startswith("failed"):
                failures.append((job[2]["segment_id"], result))
                print(f"{job[2]['segment_id']}: {result}", flush=True)
            if sum(status.values()) % 50 == 0:
                print(f"DNSMOS {sum(status.values())}/{len(jobs)} {dict(status)}", flush=True)
    cm.write_json(args.screen / "q_failures.json", failures)
    cm.write_json(args.screen / "q_run.json", cm.run_info("Q", PARAMS, params_hash=tag, status=dict(status),
                                                          failures=len(failures),
                                                          not_ready=sorted(set(tasks) - ready)))
    return 1 if failures else 0


def summarize(screen: Path, coverage: float, min_clips: int = PARAMS["min_clips"]) -> dict:
    windows, tracks = [], []
    for path in sorted((screen / "dnsmos").glob("*.json")):
        payload = cm.load_json(path)
        sid = payload["segment_id"]
        for x, per_window in payload["tracks"].items():
            pooled = []
            for offset, item in sorted(per_window.items(), key=lambda kv: int(kv[0])):
                scored = [v for v, c in zip(item["ovr"], item["coverage"]) if v is not None and c >= coverage]
                pooled.extend(scored)
                windows.append({"seg": sid, "x": x, "offset_s": int(offset), "scored": len(scored),
                                "median": float(np.median(scored)) if len(scored) >= min_clips else None})
            tracks.append({"seg": sid, "x": x, "scored": len(pooled),
                           "median": float(np.median(pooled)) if pooled else None})
    cm.write_tsv(screen / "q_windows.tsv", windows, ["seg", "x", "offset_s", "scored", "median"])
    cm.write_tsv(screen / "q_tracks.tsv", tracks, ["seg", "x", "scored", "median"])
    summary = {"coverage": coverage, "windows": len(windows), "tracks": len(tracks),
               "windows_with_median": sum(w["median"] is not None for w in windows),
               "windows_below_2_3": sum(w["median"] is not None and w["median"] < 2.3 for w in windows),
               "tracks_below_2_5": sum(t["median"] is not None and t["median"] < 2.5 for t in tracks)}
    cm.write_json(screen / "q_summary.json", cm.run_info("Q-summary", PARAMS, counts=summary))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the downloaded release")
    parser.add_argument("--screen", type=Path, required=True, help="the screen directory")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True, help="DNSMOS P.835 sig_bak_ovr.onnx")
    r.add_argument("--sha256", default=None, help=f"expected sha256 of the model (the released data: {MODEL_SHA256})")
    r.add_argument("--processes", type=int, default=os.cpu_count())
    s = sub.add_parser("summarize")
    s.add_argument("--coverage", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.command == "run":
        return command_run(args)
    print(json.dumps(summarize(args.screen, args.coverage)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
