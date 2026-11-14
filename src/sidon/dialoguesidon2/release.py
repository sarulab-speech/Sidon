"""Release checkpoints of the separation model: a directory with model.safetensors (every weight and buffer of
DialogueSidon2) and config.json (the model config, without local paths). Loading is strict.

  python -m sidon.dialoguesidon2.release --checkpoint <training final.ckpt> --out <release dir>
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch

FORMAT = "dialoguesidon2"
VERSION = 1
# Local paths used only to build a fresh model; a release does not need them.
PATH_KEYS = ("vae_checkpoint", "statistics_path")


def release_config(cfg: dict, global_step: int | None = None) -> dict:
    cfg = deepcopy(cfg)
    for key in PATH_KEYS:
        cfg[key] = None
    return {"format": FORMAT, "version": VERSION, "global_step": global_step, "cfg": cfg}


def save_release(model, directory, global_step: int | None = None) -> Path:
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
    config = release_config(model.cfg, global_step)
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    return directory


def load_release(directory, map_location: str | torch.device = "cpu"):
    """A DialogueSidon2 in eval mode from a release directory; every key must match exactly."""
    from safetensors.torch import load_file

    from sidon.model.dialoguesidon2.lightning_module import DialogueSidon2

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if config.get("format") != FORMAT or config.get("version") != VERSION:
        raise ValueError(f"{directory} is not a DialogueSidon2 release (format {config.get('format')!r})")
    model = DialogueSidon2(config["cfg"], initialize_pretrained=False)
    state = load_file(str(directory / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(map_location).eval()


def load_model(path, map_location: str | torch.device = "cpu"):
    """A separation model in eval mode from a release directory or a training checkpoint (.ckpt)."""
    from sidon.model.dialoguesidon2.lightning_module import DialogueSidon2

    path = Path(path)
    if path.is_dir():
        return load_release(path, map_location)
    model = DialogueSidon2.load_from_checkpoint(str(path), map_location="cpu", initialize_pretrained=False)
    return model.to(map_location).eval()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="separation training checkpoint (.ckpt)")
    parser.add_argument("--out", type=Path, required=True, help="release directory to write")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty")
    model = load_model(args.checkpoint)
    step = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False).get("global_step")
    save_release(model, args.out, step)
    reloaded = load_release(args.out)
    expected = model.state_dict()
    if any(not torch.equal(value, expected[key]) for key, value in reloaded.state_dict().items()):
        raise RuntimeError("The release does not reload to the checkpoint's weights")
    print(f"Wrote {args.out} (global step {step})")


if __name__ == "__main__":
    main()
