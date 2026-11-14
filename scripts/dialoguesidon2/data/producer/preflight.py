"""Checks before a producer run.

1. `producer` and `sidon` import from this repository (an installed copy of `sidon` elsewhere would be used
   silently otherwise), and pyroomacoustics loads;
2. the five mp3 effectors (qscale 10, 8, 4, 2, 1) encode 1 s of noise. The producer's codec step returns the sample
   unchanged when the encoder fails, so without a working FFmpeg the mp3 degradation would silently disappear;
3. with --source, one record of task --shard-idx goes through the whole pipeline and process_item, and its members
   and byte sizes are compared with the expected record layout (common/shards_qc.MEMBER_SIZES); with
   --key-pattern, its name must be <source key>_0 with the source key matching the pattern.

  python scripts/dialoguesidon2/data/producer/preflight.py --out preflight.json --source SOURCE --noise NOISE \\
      [--key-pattern 'openyap-[0-9a-f]{12}-[0-9]{5}']
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import tarfile
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(DATA_DIR))

QSCALES = (10, 8, 4, 2, 1)


def check_imports() -> dict:
    import producer.datamodule as datamodule
    import sidon
    files = {"sidon": sidon.__file__, "producer.datamodule": datamodule.__file__}
    try:
        import pyroomacoustics  # noqa: F401
        pra = "ok"
    except Exception as error:  # e.g. a C++ runtime that is too old for its compiled extension
        pra = f"{type(error).__name__}: {error}"
    inside = all(Path(f).resolve().is_relative_to(REPO) for f in files.values())
    return {"files": files, "pyroomacoustics": pra, "pass": inside and pra == "ok"}


def check_codecs() -> dict:
    import torch
    import torchaudio
    noise = 0.1 * torch.randn(1, 48_000, generator=torch.Generator().manual_seed(0))
    results = {}
    for q in QSCALES:
        effector = torchaudio.io.AudioEffector(format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=q))
        try:
            out = effector.apply(noise.T, 48_000).T
            results[str(q)] = {"ok": bool(torch.isfinite(out).all()) and out.numel() > 0, "shape": list(out.shape)}
        except Exception as error:
            results[str(q)] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return {"torchaudio": torchaudio.__version__, "effectors": results, "pass": all(r["ok"] for r in results.values())}


def overrides(source: Path, noise: Path, shard_idx: int, output_root: str, writer_name: str) -> list[str]:
    return ["data=dialoguesidon2_producer", f"data.datamodule.train_wds_patterns=[{source}/train]",
            f"data.datamodule.val_wds_patterns=[{source}/dev]", f"data.datamodule.noise_path=[{noise}]",
            f"data.datamodule.shard_idx={shard_idx}", f"preprocess.output_root={output_root}",
            f"preprocess.writer_name={writer_name}"]


def tar_members(path: Path) -> list[tuple[str, int]]:
    with tarfile.open(path) as stream:
        return [(member.name.split(".", 1)[1], member.size) for member in stream]


def dry_run(source: Path, noise: Path, shard_idx: int, key_pattern: str | None) -> dict:
    """One record through the task's pipeline (no DataLoader workers) and process_item, written to a tar."""
    import hydra
    import webdataset as wds
    from hydra import compose, initialize_config_dir

    from common.shards_qc import MEMBER_SIZES
    from sidon.preprocess import process_item
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose("preprocess", overrides=overrides(source, noise, shard_idx, "/nonexistent", "dry_run"))
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup()
    record = process_item(next(iter(datamodule.train_dataset)))
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
        path = Path(directory) / "dry.tar"
        with wds.TarWriter(str(path)) as writer:
            writer.write(record)
        members = tar_members(path)
    names = record["names.pickle"]
    names = pickle.loads(names) if isinstance(names, bytes) else names
    sizes = dict(members)
    layout_ok = set(sizes) == set(MEMBER_SIZES) and all(size is None or sizes[name] == size
                                                         for name, size in MEMBER_SIZES.items())
    key_ok = isinstance(names, list) and len(names) == 1 and (
        key_pattern is None or re.fullmatch(key_pattern + "_0", names[0]) is not None)
    return {"members": members, "names": names, "train_shards": len(datamodule.train_wds_patterns),
            "pass": bool(layout_ok and key_ok)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None, help="source tars ({train,dev}/data-*.tar): dry run")
    parser.add_argument("--noise", type=Path, default=None, help="noise shard directory (needed for the dry run)")
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--key-pattern", default=None, help="regex of a source key (without the producer's _0)")
    args = parser.parse_args(argv)
    report = {"imports": check_imports(), "codecs": check_codecs()}
    if args.source is not None:
        try:
            if args.noise is None:
                raise ValueError("--noise is required with --source")
            report["dry_run"] = dry_run(args.source, args.noise, args.shard_idx, args.key_pattern)
        except Exception as error:
            report["dry_run"] = {"pass": False, "error": f"{type(error).__name__}: {error}"}
    report["pass"] = all(part["pass"] for part in report.values() if isinstance(part, dict))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({k: v.get("pass") if isinstance(v, dict) else v for k, v in report.items()}), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
