"""Pack restored conversations into source tars for the producer (the base root: CallHome and Fisher).

Every audio file below --input-dir becomes one record: the file's bytes under its extension (`flac` for the
restored files) and `original_path.txt`, with a random uuid4 key. The producer then cuts each conversation into
120 s windows itself. Shards are written with webdataset's ShardWriter (default limits unless --maxcount is given)
to <output-root>/<input-dir relative to --restored-root>/data-%05d.tar.

  python scripts/dialoguesidon2/data/common/pack_sources.py --restored-root RESTORED \\
      --input-dir RESTORED/callhome-eng --output-root SOURCE
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--restored-root", type=Path, required=True,
                        help="root used to name the output directory (input-dir relative to it)")
    parser.add_argument("--input-dir", type=Path, required=True, help="directory scanned recursively")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", default="data-%05d.tar", help="shard name pattern")
    parser.add_argument("--exts", nargs="*", default=AUDIO_EXTENSIONS, help="audio file extensions to include")
    parser.add_argument("--maxcount", type=int, default=None, help="records per shard (default: webdataset's)")
    args = parser.parse_args(argv)
    import webdataset as wds
    from tqdm import tqdm
    restored_root, input_dir, output_root = (p.resolve() for p in (args.restored_root, args.input_dir,
                                                                   args.output_root))
    try:
        relative = input_dir.relative_to(restored_root)
    except ValueError:
        relative = Path(input_dir.name)
    pattern = output_root / relative / args.name
    pattern.parent.mkdir(parents=True, exist_ok=True)
    extensions = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.exts}
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    kwargs = {"maxcount": args.maxcount} if args.maxcount is not None else {}
    with wds.ShardWriter(str(pattern), **kwargs) as writer:
        for path in tqdm(files, desc="Writing shards"):
            writer.write({"__key__": uuid.uuid4().hex, path.suffix.lower()[1:]: path.read_bytes(),
                          "original_path.txt": str(path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
