#!/usr/bin/env python3
"""DialogueSidon2 — two-speaker dialogue separation and restoration demo.

Loads the torch.export release of sarulab-speech/DialogueSidon2 from the Hugging Face Hub (ds2_infer.py). The model
reads each speaker's restored latents directly from the mixture in one forward pass: there is no sampling, and the
output is the same on every run. Inputs up to 30 s, the training length, are separated in one shot; longer inputs
in 30 s windows with 5 s of overlap, cross-faded, with the speakers re-aligned across windows.
The interface follows the DialogueSidon demo (sarulab-speech/DialogueSidon-demo).
"""
from __future__ import annotations

import os
import subprocess
import tempfile

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

import gradio as gr
import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download

import ds2_infer

REPO_ID = "sarulab-speech/DialogueSidon2"
MODEL_FILES = ["separator.pt2", "vae_decoder.pt2", "metadata.json"]
HF_TOKEN = os.environ.get("HF_TOKEN")
EXAMPLE = "LDC2026S02_mono.wav"  # the DialogueSidon demo's example, uploaded next to app.py on the Space

MODEL_DIR = snapshot_download(REPO_ID, allow_patterns=MODEL_FILES, token=HF_TOKEN)
_cache: dict = {}


def get_models(device: torch.device) -> dict:
    key = str(device)
    if key not in _cache:
        _cache[key] = ds2_infer.load(MODEL_DIR, device)
    return _cache[key]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def separate(wav: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
    return ds2_infer.separate(wav, sample_rate, get_models(get_device())), ds2_infer.SAMPLE_RATE


# ---------------------------------------------------------------------------
# Video helpers (as in the DialogueSidon demo; audio is extracted at 24 kHz)
# ---------------------------------------------------------------------------

def extract_audio_from_video(video_path: str) -> tuple[torch.Tensor, int]:
    """Extract mono audio from a video file using ffmpeg. Returns (wav [1, T], sr)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(ds2_infer.SAMPLE_RATE), "-vn", tmp_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wav, sr = torchaudio.load(tmp_path)
    os.unlink(tmp_path)
    return wav, sr


def create_stereo_video(video_path: str, spk1: np.ndarray, spk2: np.ndarray, out_sr: int) -> str:
    """Mux the separated speakers (L = speaker 1, R = speaker 2) back into the video."""
    stereo = torch.from_numpy(np.stack([spk1, spk2], axis=0)).float()
    stereo = stereo / stereo.abs().max().clamp_min(1e-6) * 0.9
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        audio_path = tmp_audio.name
    torchaudio.save(audio_path, stereo, out_sr)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        out_path = tmp_video.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-map", "0:v:0", "-map", "1:a:0", "-shortest", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.unlink(audio_path)
    return out_path


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def _wav_to_numpy_output(wav_tensor: torch.Tensor, sr: int) -> tuple[int, np.ndarray]:
    arr = wav_tensor.cpu().numpy()
    arr = np.clip(arr / max(np.abs(arr).max(), 1e-6) * 0.9, -1.0, 1.0)
    return sr, (arr * 32767).astype(np.int16)


def run_separation_audio(input_audio: tuple[int, np.ndarray] | None):
    if input_audio is None:
        raise gr.Error("Please upload an audio file.")
    sr, audio_np = input_audio
    wav = torch.from_numpy(audio_np.copy()).float()
    if wav.ndim == 2 and wav.shape[1] <= 8:
        wav = wav.T  # Gradio gives (samples, channels)
    if audio_np.dtype in (np.int16, np.int32):
        wav = wav / float(np.iinfo(audio_np.dtype).max)
    separated, out_sr = separate(wav, sr)
    return _wav_to_numpy_output(separated[0], out_sr), _wav_to_numpy_output(separated[1], out_sr)


def run_separation_video(video_path: str | None):
    if video_path is None:
        raise gr.Error("Please upload a video file.")
    wav, sr = extract_audio_from_video(video_path)
    separated, out_sr = separate(wav, sr)
    spk1, spk2 = separated[0].numpy(), separated[1].numpy()
    out_video = create_stereo_video(video_path, spk1, spk2, out_sr)
    return out_video, _wav_to_numpy_output(separated[0], out_sr), _wav_to_numpy_output(separated[1], out_sr)


if HAS_SPACES:
    run_separation_audio = spaces.GPU(run_separation_audio)
    run_separation_video = spaces.GPU(run_separation_video)


with gr.Blocks(title="DialogueSidon2 — Dialogue Separation") as demo:
    gr.Markdown(
        """
        # DialogueSidon2 — Two-Speaker Dialogue Separation & Restoration
        Upload a degraded or noisy audio/video recording of a two-speaker conversation.
        DialogueSidon2 **separates** the two speakers and **restores** clean 24 kHz speech from the mixture in one
        forward pass, without sampling: the same input always gives the same output.
        Inputs up to 30 s are processed in one shot. Longer inputs are processed in 30 s windows with 5 s overlap
        crossfade and automatic speaker re-alignment across windows.
        **Model**: [sarulab-speech/DialogueSidon2](https://huggingface.co/sarulab-speech/DialogueSidon2) ·
        previous version: [DialogueSidon](https://huggingface.co/spaces/sarulab-speech/DialogueSidon-demo)
        """
    )

    with gr.Tabs():
        with gr.Tab("Audio"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(label="Input mixture (mono or stereo, any sample rate)", type="numpy")
                    audio_btn = gr.Button("Separate", variant="primary")
                with gr.Column():
                    audio_spk1 = gr.Audio(label="Speaker 1", type="numpy")
                    audio_spk2 = gr.Audio(label="Speaker 2", type="numpy")
            audio_btn.click(fn=run_separation_audio, inputs=[audio_input], outputs=[audio_spk1, audio_spk2])
            if os.path.exists(EXAMPLE):
                gr.Examples(examples=[[EXAMPLE]], inputs=[audio_input], label="Ex1")

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Input video (audio will be extracted and separated)")
                    video_btn = gr.Button("Separate", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Output video (Speaker 1 = Left, Speaker 2 = Right)")
                    video_spk1 = gr.Audio(label="Speaker 1 (audio only)", type="numpy")
                    video_spk2 = gr.Audio(label="Speaker 2 (audio only)", type="numpy")
            video_btn.click(fn=run_separation_video, inputs=[video_input],
                            outputs=[video_output, video_spk1, video_spk2])

    gr.Markdown("---\n**License**: CC-BY-NC 4.0 — non-commercial use only. Not for voice cloning or speaker "
                "identification.")


if __name__ == "__main__":
    demo.launch()
