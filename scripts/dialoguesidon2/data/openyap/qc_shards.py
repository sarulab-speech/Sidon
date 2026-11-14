"""Checks of the producer output for OpenYap-1k (common/shards_qc.py).

* headers (tar headers only): every record has the 11 members with the expected byte sizes; keys are unique; per
  producer task the record count equals the windows of its source tars. Writes qc_headers.json.
* finalize: requires a done marker for each of the 16 tasks, runs the header check, compares the totals with the
  source manifest (a task may fall short by at most --max-shortfall windows), makes train/ and valid/ read-only and
  writes N_SHARDS.txt (the number of training shards).
* audio: decodes the first record of --count random training shards and reports per channel of input_wav the
  active level, silence fraction, 99% bandwidth and the count of exact-zero runs of 20-200 ms (packet-loss traces in
  the target), plus the BLR of the final 24 kHz targets at the crosstalk screen's lag (transcript masks). Exports
  --exports records as WAV for listening.

  python scripts/dialoguesidon2/data/openyap/qc_shards.py --shards OUT/openyap_24k_120s --source WORK/source headers
  python scripts/dialoguesidon2/data/openyap/qc_shards.py --shards OUT/openyap_24k_120s --source WORK/source finalize
  python scripts/dialoguesidon2/data/openyap/qc_shards.py --shards OUT/openyap_24k_120s --source WORK/source audio \\
      --out WORK/qc --screen WORK/screen --root OPENYAP/conversations
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import shards_qc as sq  # noqa: E402
from openyap import corpus as oc  # noqa: E402
from openyap import screen_crosstalk as sc  # noqa: E402


def target_blr(record: dict, screen: dict, root: Path) -> list[float | None]:
    """BLR of each final 24 kHz target channel at the screen's conversation lag, on the window's transcript masks."""
    names = record["names.pickle"]
    conv, offset = oc.parse_window_key(names[0])  # the producer's key: <source key>_0
    words = oc.clean_words({s: oc.load_words(root, conv, s) for s in oc.SPEAKERS}, sc.echo_window())
    shifted = {s: [{**w, "start": w["start"] - offset, "end": w["end"] - offset} for w in words[s]
                   if offset - 1 <= w["start"] < offset + oc.WINDOW_S + 1] for s in oc.SPEAKERS}
    frames = oc.WINDOW_S * oc.FPS
    o = {s: oc.word_mask(shifted[s], oc.WINDOW_S + 1.0, frames=frames) for s in oc.SPEAKERS}
    g = {s: oc.dilate(o[s], sc.PARAMS["guard_frames"]) for s in oc.SPEAKERS}
    wav = record["input_wav.pth"][0].numpy()
    result = []
    for i, s in enumerate(oc.SPEAKERS):
        e = sq.frame_energy(wav[i], 24_000, bandpass=True)[:frames]
        lag = screen.get((conv, s), {}).get("lag") or 0
        stats = sc.levels(e, o[s] & ~o[oc.other(s)], g[s], o[oc.other(s)], int(lag))
        result.append(stats["blr"])
    return result


def audio(out: Path, shards: Path, screen_dir: Path, root: Path, count: int, exports: int, seed: int) -> int:
    import soundfile as sf
    rng = random.Random(seed)
    tars = sorted((shards / "train").rglob("*.tar"))
    screen = {}
    if (screen_dir / "tracks.jsonl").exists():
        screen = {(t["conv"], t["s"]): t for t in sc.load_screen(screen_dir)[0]}
    (out / "exports").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, path in enumerate(rng.sample(tars, min(count, len(tars)))):
        record = sq.first_record(path)
        wav = record["input_wav.pth"][0].numpy()
        row = {"shard": str(path), "names": record["names.pickle"]}
        for i in range(2):
            row[f"ch{i}"] = sq.channel_stats(wav[i])
        try:
            row["blr_target"] = target_blr(record, screen, root)
        except Exception as error:
            row["blr_target_error"] = f"{type(error).__name__}: {error}"
        if n < exports:
            stem = out / "exports" / f"{n:02d}_{record['names.pickle'][0]}"
            for i in range(2):
                sf.write(f"{stem}_target{i}.wav", wav[i], 24_000, subtype="PCM_16")
            sf.write(f"{stem}_noisy_mixture.wav", record["noisy_mixture.pth"][0].numpy(), 24_000, subtype="PCM_16")
        rows.append(row)
    report = {"records": rows, "summary": sq.summarize(rows)}
    oc.write_json(out / "report.json", oc.run_info("audio-qc", {"count": count, "exports": exports, "seed": seed},
                                                    report=report))
    print(json.dumps(report["summary"], indent=1), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", type=Path, required=True, help="producer output (<root>/<writer name>)")
    parser.add_argument("--source", type=Path, required=True, help="the source directory (manifest.tsv)")
    parser.add_argument("--max-shortfall", type=int, default=0,
                        help="windows a producer task may drop by itself (all-quiet chunks, NaN records)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("headers")
    sub.add_parser("finalize")
    a = sub.add_parser("audio")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--screen", type=Path, required=True, help="the screen directory (tracks.jsonl)")
    a.add_argument("--root", type=Path, required=True, help="the release's conversations/ directory")
    a.add_argument("--count", type=int, default=300)
    a.add_argument("--exports", type=int, default=20)
    a.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args(argv)
    if args.command == "headers":
        source = args.source if (args.source / "manifest.tsv").exists() else None
        report = sq.headers(args.shards, source, sq.TASKS, args.max_shortfall)
        oc.write_json(args.shards / "qc_headers.json", oc.run_info("qc-headers", {}, report=report))
        print(json.dumps({k: report.get(k) for k in ("pass", "train_shards", "train_records", "task_mismatches",
                                                      "task_shortfalls", "valid_records", "duplicate_keys")}),
              flush=True)
        return 0 if report["pass"] else 1
    if args.command == "finalize":
        return sq.finalize(args.shards, args.source, sq.TASKS, count_file="N_SHARDS.txt",
                           max_shortfall=args.max_shortfall)
    return audio(args.out, args.shards, args.screen, args.root, args.count, args.exports, args.seed)


if __name__ == "__main__":
    sys.exit(main())
