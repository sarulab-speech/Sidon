"""Pack a LibriTTS-R release into the tar.gz layout that sim/index_libritts.py reads (CPU stage, no decoding).

Input: --release DIR, the extracted LibriTTS-R release with DIR/<subset>/<spk>/<chapter>/<spk>_<chapter>_<para>_<sent>
.wav (24 kHz PCM_16 mono). Output: --out DIR/<subset>/data-NNNNN.tar.gz, each holding up to --per-tar utterances
as members <spk>/<chapter>/<utt>.wav.audio.wav (the WAV bytes copied unchanged), in sorted path order. Transcripts
and other files are not copied. The plan depends only on the utterances, not on how they are spread over tars.

  python sim/pack_libritts.py --release /data/LibriTTS_R --out WORK/libritts_tars
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import base as sc  # noqa: E402

UTTERANCE = re.compile(r"(\d+)_(\d+)_(\d+)_(\d+)\.wav")


def utterances(release: Path, subset: str) -> list[Path]:
    """The release's utterance WAVs of one subset, sorted by path."""
    found = []
    for path in sorted((release / subset).glob("*/*/*.wav")):
        match = UTTERANCE.fullmatch(path.name)
        if match and (match.group(1), match.group(2)) == (path.parent.parent.name, path.parent.name):
            found.append(path)
    return found


def write_tar(job) -> str:
    target, paths = job
    if target.exists():
        return "skipped"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for path in paths:
            data = path.read_bytes()
            info = tarfile.TarInfo(f"{path.parent.parent.name}/{path.parent.name}/{path.name}.audio.wav")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tmp.replace(target)
    return "done"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release", type=Path, required=True, help="the extracted LibriTTS-R release")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-tar", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args(argv)
    jobs, counts = [], {}
    for subset in sc.SUBSETS:
        paths = utterances(args.release, subset)
        if not paths:
            raise SystemExit(f"No utterances under {args.release / subset}")
        counts[subset] = len(paths)
        for number, start in enumerate(range(0, len(paths), args.per_tar)):
            jobs.append((args.out / subset / f"data-{number:05d}.tar.gz", paths[start:start + args.per_tar]))
    with Pool(args.processes) as pool:
        status = pool.map(write_tar, jobs)
    print(json.dumps({"utterances": counts, "tars": len(jobs), "written": status.count("done")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
