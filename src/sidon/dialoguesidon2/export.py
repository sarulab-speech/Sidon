"""Export the separation model as two torch.export programs plus metadata, for inference without this package
(spaces/dialoguesidon2/ds2_infer.py):

- separator.pt2: input features [1, T, 160] -> decoder-ready (de-normalized) latents [1, 2, T, D]: the
  conditioner, the readout head and the de-normalization with the latent statistics;
- vae_decoder.pt2: latents [2, D, T] -> waveforms [2, N] at 24 kHz (the SSL-VAE decoder);
- metadata.json: sample rates, hop, latent size, maximum frames and the long-input window.

Node metadata that holds source locations (stack traces) is removed before saving, so the files contain no paths
of the exporting machine. Before saving, the wrapper is compared with the model's own inference path and each
program with its eager module at lengths other than the export example's.

  python -m sidon.dialoguesidon2.export --model <release dir or .ckpt> --out <dir>
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

META_KEYS_DROPPED = ("stack_trace", "source_fn_stack", "from_node", "original_aten", "seq_nr")
MAX_FRAMES = 6100  # 122 s at 50 frames per second


class Separator(nn.Module):
    """Input features [1, T, 160] -> latents [1, 2, T, D] in the decoder's input space."""

    def __init__(self, model):
        super().__init__()
        self.conditioner = model.conditioner
        self.projection = model.readout.projection
        self.head = model.readout.head
        self.register_buffer("mean", model.vae.mean.detach().clone().float())
        self.register_buffer("scale", torch.sqrt(model.vae.var.detach().clone().float() + 1e-5))

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        condition = self.conditioner(input_features=input_features, attention_mask=None).last_hidden_state
        joint = self.head(self.projection(condition))
        b, frames, channels = joint.shape
        latent = joint.reshape(b, frames, 2, channels // 2).transpose(1, 2)
        return latent * self.scale + self.mean


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.vae.decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent).squeeze(1)  # [S, D, T] -> [S, N]


def strip(program: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    for module in program.graph_module.modules():
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                for key in META_KEYS_DROPPED:
                    node.meta.pop(key, None)
    return program


def max_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def example(seconds: float, seed: int) -> torch.Tensor:
    """Input features of a random 24 kHz waveform."""
    from sidon.dialoguesidon2.audio import extract_features

    generator = torch.Generator().manual_seed(seed)
    wave = torch.randn(1, int(seconds * 24000), generator=generator) * 0.1
    inputs, _ = extract_features(wave, torch.tensor([wave.size(-1)]), 24000)
    frames = int(inputs["attention_mask"].sum())
    return inputs["input_features"][:, :frames]


def export(model, out: Path, window_seconds: float = 30.0, overlap_seconds: float = 5.0,
           global_step: int | None = None, check_seconds=(3.1, 17.9, 30.0), example_seconds: float = 10.0,
           tolerance: float = 1e-4) -> dict:
    model = model.float().eval()
    separator, decoder = Separator(model).eval(), Decoder(model).eval()
    report = {"hop": model.vae.hop, "latent_dim": model.vae.dim,
              "separator_parameters": sum(p.numel() for p in separator.parameters()),
              "decoder_parameters": sum(p.numel() for p in decoder.parameters())}

    # The wrapper against the model's own inference path.
    with torch.inference_mode():
        x = example(min(7.3, max(check_seconds)), 1)
        frames = torch.tensor([x.size(1)])
        expected, _ = model.separate_features({"input_features": x,
                                               "attention_mask": torch.ones(1, x.size(1), dtype=torch.long)},
                                              frames * model.vae.hop)
        got = decoder(separator(x)[0].transpose(1, 2))
        n = min(got.size(-1), expected.size(-1))
        report["wrapper_vs_model_max_error"] = max_error(got[:, :n], expected[0, :, :n])
        if report["wrapper_vs_model_max_error"] > 1e-5:
            raise RuntimeError(f"The export wrapper differs from the model: {report['wrapper_vs_model_max_error']}")

    dim = torch.export.Dim("frames", min=2, max=MAX_FRAMES)
    with torch.no_grad():
        x = example(example_seconds, 2)
        separator_program = strip(torch.export.export(separator, (x,), dynamic_shapes={"input_features": {1: dim}}))
        latent = separator(x)[0].transpose(1, 2).contiguous()
        decoder_program = strip(torch.export.export(decoder, (latent,), dynamic_shapes={"latent": {2: dim}}))
        checks = {}
        for index, seconds in enumerate(check_seconds):
            x = example(seconds, 3 + index)
            latent = separator(x)
            checks[f"separator_{seconds}s"] = max_error(separator_program.module()(x), latent)
            z = latent[0].transpose(1, 2).contiguous()
            checks[f"decoder_{seconds}s"] = max_error(decoder_program.module()(z), decoder(z))
        if max(checks.values()) > tolerance:
            raise RuntimeError(f"An exported program differs from its eager module: {checks}")
    report["exported_vs_eager_max_error"] = checks

    out.mkdir(parents=True, exist_ok=True)
    torch.export.save(separator_program, str(out / "separator.pt2"))
    torch.export.save(decoder_program, str(out / "vae_decoder.pt2"))
    meta = {"model": "DialogueSidon2: per-speaker readout from a fine-tuned w2v-BERT 2.0, SSL-VAE decoder",
            "sample_rate": 24000, "feature_rate": 16000, "hop": model.vae.hop, "latent_dim": model.vae.dim,
            "speakers": 2, "max_frames": MAX_FRAMES, "window_seconds": window_seconds,
            "overlap_seconds": overlap_seconds, "training_steps": global_step, "torch": torch.__version__}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    from sidon.dialoguesidon2.release import load_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="release directory or training checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--overlap-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    model = load_model(args.model)
    step = None
    if args.model.is_dir():
        step = json.loads((args.model / "config.json").read_text()).get("global_step")
    else:
        step = torch.load(args.model, map_location="cpu", mmap=True, weights_only=False).get("global_step")
    report = export(model, args.out, args.window_seconds, args.overlap_seconds, step)
    print(json.dumps(report, indent=1))
    print("saved", sorted(p.name for p in args.out.iterdir()))


if __name__ == "__main__":
    main()
