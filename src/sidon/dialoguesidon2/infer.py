"""Separate two-speaker recordings into one 24 kHz file per speaker.

  python -m sidon.dialoguesidon2.infer --model <release dir or .ckpt> --output-dir out/ a.wav b.flac ...

Each input is mixed to mono, scaled to a peak of 0.9 and resampled to 24 kHz. Inputs up to --window-seconds (the
30 s training length by default) are separated in one pass. Longer inputs are separated in windows of that length
with --overlap-seconds of overlap; the speaker order of each window is matched to the previous window by the
correlation of the two outputs in the overlap, and the overlap is cross-faded. Writes <stem>_speaker1.wav and
<stem>_speaker2.wav.
"""

import argparse
from pathlib import Path

import torch
import torchaudio

SAMPLE_RATE = 24000


def prepare(wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """[C, N] or [N] audio -> [1, N24] mono at 24 kHz with a peak of 0.9 (scaled before resampling)."""
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if wave.ndim != 2:
        raise ValueError("Expected audio of shape (channels, samples) or (samples,)")
    wave = wave.float().mean(dim=0, keepdim=True)
    peak = wave.abs().max()
    if peak > 0:
        wave = wave / peak * 0.9
    if sample_rate != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sample_rate, SAMPLE_RATE)
    return wave


@torch.inference_mode()
def separate_window(model, wave: torch.Tensor) -> torch.Tensor:
    """[1, N] at 24 kHz (at most 120 s) -> [2, N] separated speakers at 24 kHz."""
    output, _ = model(wave, torch.tensor([wave.size(-1)]), SAMPLE_RATE)
    return output[0]


def _similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1) - a.mean(), b.reshape(-1) - b.mean()
    norm = torch.linalg.norm(a) * torch.linalg.norm(b)
    return float(torch.dot(a, b) / norm) if float(norm) > 1e-8 else 0.0


def window_starts(total: int, window: int, overlap: int) -> list[int]:
    """Window starts covering [0, total): a hop of window - overlap, and a last window ending at total."""
    if total <= window:
        return [0]
    starts = list(range(0, total - window, window - overlap))
    return starts + [total - window]


@torch.inference_mode()
def separate(model, wave: torch.Tensor, sample_rate: int, window_seconds: float = 30.0,
             overlap_seconds: float = 5.0, log: list | None = None) -> torch.Tensor:
    """Audio [C, N] or [N] at sample_rate -> [2, N24] separated speakers at 24 kHz (on the CPU).
    log, if given, receives one record per window boundary: the overlap and both alignment scores."""
    wave = prepare(wave, sample_rate)
    window = int(window_seconds * SAMPLE_RATE)
    overlap = int(overlap_seconds * SAMPLE_RATE)
    if not 0 <= overlap < window:
        raise ValueError("The overlap must be shorter than the window")
    total = wave.size(-1)
    if total <= window:
        return separate_window(model, wave).float().cpu()
    output = None
    end_before = 0
    for start in window_starts(total, window, overlap):
        chunk = wave[:, start:start + window]
        # Each window is separated at peak 0.9 and scaled back, so windows keep their relative level.
        peak = float(chunk.abs().max())
        if peak <= 0:
            out = torch.zeros(2, chunk.size(-1))
        else:
            out = separate_window(model, chunk / peak * 0.9).float().cpu() * (peak / 0.9)
        if output is None:
            output, end_before = out, start + chunk.size(-1)
            continue
        shared = end_before - start
        before, now = output[:, -shared:], out[:, :shared]
        direct = _similarity(before[0], now[0]) + _similarity(before[1], now[1])
        swapped = _similarity(before[0], now[1]) + _similarity(before[1], now[0])
        if swapped > direct:
            out = out[[1, 0]]
        if log is not None:
            log.append({"start": start / SAMPLE_RATE, "overlap": shared / SAMPLE_RATE, "direct": direct,
                        "swapped": swapped, "swap": swapped > direct})
        fade = torch.linspace(0.0, 1.0, shared).unsqueeze(0)
        blended = output[:, -shared:] * (1 - fade) + out[:, :shared] * fade
        output = torch.cat([output[:, :-shared], blended, out[:, shared:]], dim=-1)
        end_before = start + chunk.size(-1)
    return output


def main(argv: list[str] | None = None) -> None:
    from sidon.dialoguesidon2.release import load_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="release directory or training checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--overlap-seconds", type=float, default=5.0)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if not 0 < args.window_seconds <= 120:
        parser.error("--window-seconds must be in (0, 120]")
    model = load_model(args.model, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        wave, rate = torchaudio.load(str(path))
        if wave.numel() == 0:
            parser.error(f"{path} is empty")
        output = separate(model, wave, rate, args.window_seconds, args.overlap_seconds)
        for speaker in range(2):
            destination = args.output_dir / f"{path.stem}_speaker{speaker + 1}.wav"
            torchaudio.save(str(destination), output[speaker:speaker + 1], SAMPLE_RATE)
        print(f"{path} -> {args.output_dir}/{path.stem}_speaker{{1,2}}.wav ({output.size(-1) / SAMPLE_RATE:.2f} s)")


if __name__ == "__main__":
    main()
