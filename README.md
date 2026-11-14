# Sidon
[![Sidon arXiv](https://img.shields.io/badge/Sidon-arXiv%202509.17052-b31b1b.svg)](https://arxiv.org/abs/2509.17052)
[![DialogueSidon arXiv](https://img.shields.io/badge/DialogueSidon-arXiv%202604.09344-b31b1b.svg)](https://arxiv.org/abs/2604.09344)
[![Sidon Gradio Demo](https://img.shields.io/badge/Sidon-Gradio%20demo-orange.svg)](https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta)
[![DialogueSidon Gradio Demo](https://img.shields.io/badge/DialogueSidon-Gradio%20demo-orange.svg)](https://huggingface.co/spaces/sarulab-speech/DialogueSidon-demo)
[![DialogueSidon2 Gradio Demo](https://img.shields.io/badge/DialogueSidon2-Gradio%20demo-orange.svg)](https://huggingface.co/spaces/sarulab-speech/DialogueSidon2-demo)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-demo)](https://huggingface.co/spaces/Wataru/SidonSamples)


Large-scale text-to-speech (TTS) systems are bottlenecked by the scarcity of clean, multilingual recordings. Sidon tackles this by pairing a fast, open-source speech restoration model with reproducible tooling so researchers can turn noisy in-the-wild corpora into studio-quality datasets that scale across dozens of languages.

Sidon consists of two stages: a w2v-BERT 2.0 feature predictor finetuned to cleanse representations from degraded speech, and a vocoder trained to synthesise restored waveforms from those features. The stack achieves restoration quality comparable to Miipher—Google's internal speech restoration pipeline—while running up to 500× faster than real time on a single GPU. We also observe that training downstream TTS models on Sidon-cleansed automatic speech recognition corpora improves zero-shot synthesis quality. This repository releases the code, configs, and models needed to reproduce Sidon's dataset cleansing workflow for the community.

This repository ships three models:

- **Sidon** ([arXiv:2509.17052](https://arxiv.org/abs/2509.17052)) — the
  single-speaker speech restoration pipeline described above.
- **DialogueSidon** ([arXiv:2604.09344](https://arxiv.org/abs/2604.09344)) — a
  diffusion-based two-speaker dialogue separator that reuses the Sidon
  backbone. See the [DialogueSidon section](#dialoguesidon--diffusion-based-dialogue-separation).
- **DialogueSidon2** — a two-speaker dialogue separator and restorer that reads
  both speakers' SSL-VAE latents directly from the mixture in one forward pass.
  See the [DialogueSidon2 section](#dialoguesidon2--direct-readout-dialogue-separation).

## Requirements

- Python 3.10+
- Recent PyTorch / CUDA stack (tested with `torch>=2.8`, `torchaudio>=2.8`)
- `uv` for dependency management (or an equivalent toolchain you are
  comfortable with)

Install the default project dependencies for Sidon and DialogueSidon:

```bash
uv sync
```

The experimental GENESES separator uses additional packages. Install them only
when running `infer_geneses.py` or `model=geneses_dialogue`:

```bash
uv sync --extra geneses
```

If you rely on a different environment manager, replicate the dependencies
listed in `pyproject.toml` (including the `geneses` extra only if needed).

## License scope

The repository's [MIT license](LICENSE) covers the project code. Dependencies
have their own licenses; installing Sidon does not change those terms. In
particular, the optional `flow-matching` package used by GENESES is published
under [CC BY-NC 4.0](https://github.com/facebookresearch/flow_matching/blob/main/LICENSE).
For version 1.0.10, its [PyPI metadata](https://pypi.org/project/flow-matching/1.0.10/)
also lists an MIT classifier, which conflicts with the license included in
the source distribution. Do not infer its license from that classifier alone.

Published checkpoints have separate model-card licenses:
[Sidon v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1) is marked MIT,
while [DialogueSidon](https://huggingface.co/sarulab-speech/DialogueSidon) and
[DialogueSidon2](https://huggingface.co/sarulab-speech/DialogueSidon2) are
marked CC BY-NC 4.0. Check the license of the specific checkpoint and all
dependencies used in a deployment.

## Repository layout

- `src/sidon/model/sidon/lightning_module.py` — Feature predictor, decoder, and
  discriminator Lightning modules.
- `src/sidon/data` — WebDataset helpers, preprocessing augmentations, and the
  `PreprocessedDataModule` used for training.
- `src/sidon/preprocess.py` — Parallel writer that turns augmented samples into
  on-disk shards.
- `config/` — Hydra configuration tree with defaults for preprocessing, data,
  models, and trainer settings.
- `scripts/` — Utility scripts plus PBS job templates for batch processing.

## Preparing data

Training consumes WebDataset shards that contain tensors expected by the
`PreprocessedDataModule`:

- `input_wav.pth` and `noisy_input_wav.pth` — paired clean / degraded waveforms
  stored as 1D float tensors.
- Optional SSL features (`ssl_inputs.pickle`, `noisy_ssl_inputs.pickle`) that
  provide contextual embeddings for the model.
- `sr.index` and other metadata entries produced by the preprocessing pipeline.

Update `config/data/preprocessed.yaml` with the locations of your prepared
shards. You can point the `train_urls` and `val_urls` entries at directories of
`.tar` / `.tar.gz` files, or text manifests containing S3 URIs. Set `is_s3=true` to stream from object storage via the AWS CLI.

## Generating preprocessed shards

Use the Hydra-driven preprocessing entrypoint to convert raw WebDataset
collections into the tensorised format described above.

1. Choose the base configuration in `config/preprocess.yaml` (e.g.
   `webdataset_preprocess_24k` or `webdataset_preprocess_48k`). These configs
   reference the augmentation pipeline, SSL encoders, and noise sources defined
   in `config/data/webdataset_preprocess_*.yaml`.
2. Set output parameters in `config/preprocess/default.yaml` (target directory,
   shard size, number of writer processes).
3. Launch preprocessing locally:

   ```bash
   uv run python -m sidon.preprocess \
     data=webdataset_preprocess_24k \
     preprocess.writer_name=my_preprocessed_run
   ```

   Hydra creates run-specific subdirectories under `outputs/` and writes shards
   into `${preprocess.output_root}/{writer_name}/{split}/{job_id}`.
4. On PBS-based clusters, adapt the templates in `scripts/pbs/` (e.g.
   `preprocess_24k.sh`) to submit distributed jobs. The scripts activate a local
   virtual environment, set MPI-friendly environment variables, and forward
   Hydra overrides to the preprocessing entrypoint.

Utilities such as `scripts/summarise_shard_durations.py` can help audit the
duration distribution of generated shards before training.

## Training pipeline

Sidon training runs in three sequential stages. Every invocation of
`python -m sidon.train` resolves a Hydra config and writes artefacts under
`outputs/<timestamped_run>/`.

1. **Feature predictor pretraining** — LoRA-adapts the SSL encoder to denoise
   representations before they are fed to the vocoder.

   ```bash
   uv run python -m sidon.train \
     model=sidon_feature_predictor \
     data=preprocessed
   ```

   The resulting checkpoint (e.g. `outputs/<run>/checkpoints/last.ckpt`) becomes
   the `model.cfg.ssl_model_name` input for the finetuning stage.

2. **Vocoder pretraining** — Trains the decoder and discriminator while the SSL
   encoder remains frozen on clean features.

   ```bash
   uv run python -m sidon.train \
     model=sidon_vocoder_pretrain \
     data=preprocessed \
     train=gan
   ```

   Capture the checkpoint path; it will be referenced as `model.cfg.pretrain_path`
   during finetuning.

3. **Vocoder finetuning** — Warm-starts from the pretraining weights and swaps
   in the denoised SSL features predicted by the feature predictor.

   ```bash
   uv run python -m sidon.train \
     model=sidon_vocoder_finetune \
     data=preprocessed_48k \
     train=gan \
     model.cfg.ssl_model_name=/path/to/feature_predictor.ckpt \
     model.cfg.pretrain_path=/path/to/vocoder_pretrain.ckpt
   ```

Adjust optimiser, scheduler, or trainer parameters via the files in
`config/model/` and `config/train/`, and use `train.ckpt_path` to resume a run.
The GAN stages (vocoder pretraining/finetuning, SSL-VAE) use manual
optimization and must run with `train=gan`: under DDP, the `static_graph: true`
of `train=default` would leave their gradients unsynchronized across GPUs.

## DialogueSidon — diffusion-based dialogue separation

Full-duplex dialogue audio, in which each speaker is recorded on a separate track, is an important resource for spoken dialogue research, but is difficult to collect at scale. Most in-the-wild two-speaker dialogue is available only as degraded monaural mixtures, making it unsuitable for systems requiring clean speaker-wise signals. We propose DialogueSidon, a model for joint restoration and separation of degraded monaural two-speaker dialogue audio. DialogueSidon combines a variational autoencoder (VAE) operates on the speech self-supervised learning (SSL) model feature, which compresses SSL model features into a compact latent space, with a diffusion-based latent predictor that recovers speaker-wise latent representations from the degraded mixture. Experiments on English, multilingual, and in-the-wild dialogue datasets show that DialogueSidon substantially improves intelligibility and separation quality over a baseline, while also achieving much faster inference.

This repository implements DialogueSidon on top of the Sidon feature backbone:
a diffusion transformer head predicts per-speaker latents over a frozen
SSL-VAE, conditioned on features from a LoRA-adapted w2v-BERT encoder.

### Architecture

- **SSL encoder** — a LoRA-adapted `facebook/w2v-bert-2.0` student encodes the
  noisy mixture into frame-level features.
- **SSL-VAE** — a pretrained `SSLVAE` (loaded from `cfg.vae_checkpoint_path`)
  provides the target latents; its weights are frozen during training.
- **Conditioning heads** — two linear projections (`output_linear1`,
  `output_linear2`) map SSL features to per-speaker VAE latents used as a
  conditioning signal.
- **Diffusion transformer head** — a DiT with AdaLN conditioning, RoPE
  attention, and sinusoidal timestep embeddings predicts the noise (or `v`
  target) for the concatenated two-speaker latents.
- **DDPM training** — noise is sampled with a `DDPMScheduler`
  (`prediction_type=v_prediction` by default, 1000 training timesteps). Speaker
  assignment is resolved with Permutation-Invariant Training on the
  conditioning heads.
- **Latent normalisation** — running mean/std buffers are initialised from the
  first training batch and re-used at inference to stabilise diffusion.

The matching inference script is `infer.py` (not `infer_geneses.py`, which is
reserved for the flow-matching GENESES separator).

### Model variants

Available under `config/model/`:

| Config | Head hidden | Head layers | Heads | Notes |
|---|---|---|---|---|
| `diffusion_dialogue_sidon` | 768 | 8 | 12 | default |
| `diffusion_dialogue_sidon_small` | 384 | 12 | 6 | small |
| `diffusion_dialogue_sidon_xsmall` | 384 | 6 | 6 | xsmall |
| `diffusion_dialogue_sidon_ac` | 768 | 8 | 12 | activation checkpointing |
| `diffusion_dialogue_sidon_wo_diffusion_head` | — | — | — | baseline without diffusion head |
| `diffusion_dialogue_sidon_wo_vae_latent` | 768 | 8 | 12 | ablation without VAE latent conditioning |
| `diffusion_dialogue_sidon_decoder_finetune` | — | — | — | decoder finetuning stage |

### Training

DialogueSidon requires a pretrained **SSL-VAE** checkpoint. Train it first
with `model=ssl_vae`, then pass the resulting checkpoint into the diffusion
run via `model.cfg.vae_checkpoint_path`.

1. **SSL-VAE pretraining** — learns the latent space that the diffusion head
   will predict over.

   ```bash
   uv run python -m sidon.train \
     model=ssl_vae \
     data=dialogue_preprocessed \
     train=gan
   ```

2. **Diffusion training** — point `model.cfg.vae_checkpoint_path` at the
   SSL-VAE checkpoint from step 1.

   ```bash
   uv run python -m sidon.train \
     model=diffusion_dialogue_sidon \
     data=dialogue_preprocessed \
     model.cfg.vae_checkpoint_path=/path/to/ssl_vae.ckpt
   ```

PBS templates for each variant are provided in
`scripts/pbs/diffusion_dialogue_sidon*.sh`.

### Inference

`infer.py` runs chunked inference with overlap, resolves speaker permutation
across chunks by cosine similarity in VAE latent space, concatenates the
per-chunk latents, and performs a single VAE decode at the end.

```bash
# Batch mode — directory of wav files
python infer.py \
  --checkpoint sidon/<run_id> \
  --input-dir ./wavs \
  --output-dir ./out \
  --device cuda:0 \
  --num-steps 30 \
  --chunk-seconds 20 \
  --overlap-seconds 5

# Single audio or video file (replaces the audio track when given a video)
python infer.py \
  --checkpoint sidon/<run_id> \
  --input-video input.mp4 \
  --output-wav separated.wav \
  --output-video output.mp4
```

Use `scripts/pbs/infer_dialogue.sh` to submit the same job on an `rt_QG` (single
GPU) PBS queue.

## DialogueSidon2 — direct-readout dialogue separation

DialogueSidon2 jointly separates and restores degraded monaural recordings of
two-speaker conversations. It keeps DialogueSidon's latent space idea, an
SSL-VAE over w2v-BERT 2.0 features, but predicts both speakers' latents
directly: there is no diffusion head and no sampling, so the same input always
gives the same output, in a single forward pass.

- Pretrained model (CC BY-NC 4.0): [sarulab-speech/DialogueSidon2](https://huggingface.co/sarulab-speech/DialogueSidon2)
- Demo: [sarulab-speech/DialogueSidon2-demo](https://huggingface.co/spaces/sarulab-speech/DialogueSidon2-demo)

### Architecture

- **Conditioner** — `facebook/w2v-bert-2.0` truncated to 13 layers and fully
  fine-tuned, reading the noisy mixture.
- **Readout head** — Linear(1024→768), then an MLP (768→768, GELU, 768→64) that
  gives both speakers' 32-dimensional latents per frame. It is trained with a
  permutation-invariant MSE against the standardized latents of the two clean
  tracks.
- **SSL-VAE** (stage 1) — DialogueSidon's architecture: a frozen 8-layer
  w2v-BERT 2.0, a KL bottleneck to 32 dimensions and a DAC decoder at 24 kHz,
  trained with mel, adversarial and feature-matching losses on Sidon's
  single-speaker training data. Its encoder provides the training targets, and
  its decoder renders the predicted latents.
- **Long inputs** — the model is trained on 30 s segments. Longer recordings
  are separated in 30 s windows with 5 s of overlap; each window's speakers are
  aligned to the previous window by the correlation of the outputs in the
  overlap, and the windows are cross-faded.

### Code layout

- `src/sidon/model/dialoguesidon2/` — the SSL-VAE (`vae.py`), the readout head
  (`readout.py`) and the Lightning modules of both stages (`lightning_module.py`).
- `src/sidon/dialoguesidon2/` — features, data modules, the Hydra training entry
  (`train.py`), latent statistics, inference (`infer.py`), release packaging
  (`release.py`) and `torch.export` (`export.py`).
- `config/dialoguesidon2.yaml` and `config/{model,data,train}/dialoguesidon2*.yaml`
  — the settings of the released model.
- `scripts/dialoguesidon2/` — launch scripts for each step.
- `scripts/dialoguesidon2/data/` — the pipelines that produced the training
  data; see [its README](scripts/dialoguesidon2/data/README.md).
- `spaces/dialoguesidon2/` — the Gradio demo and its standalone inference
  (`ds2_infer.py`, which needs only `torch` and `torchaudio`).

### Training

1. **Data** — two-speaker dialogue shards (clean per-speaker tracks and a
   degraded mixture, 120 s records at 24 kHz). The released model used four
   roots: telephone conversations (CallHome, Fisher), OpenYap-1k, Basis
   Conversations 1500 and simulated LibriTTS-R conversations. The recipes and
   the shard format are in `scripts/dialoguesidon2/data/README.md`.
2. **Stage 1: SSL-VAE** on Sidon's single-speaker training shards (written by
   `python -m sidon.preprocess`, see [Generating preprocessed shards](#generating-preprocessed-shards);
   `data=dialoguesidon2_ssl_vae` pairs two utterances into a two-track example):

   ```bash
   TRAIN_URLS=/data/sidon/train VAL_URLS=/data/sidon/valid OUTPUT_DIR=runs/ssl_vae \
     bash scripts/dialoguesidon2/train_ssl_vae.sh
   ```

3. **Latent statistics** of the stage 1 posterior on dialogue segments:

   ```bash
   VAE_CHECKPOINT=runs/ssl_vae/.../final.ckpt TRAIN_URLS=/data/dialogue/train \
   OUTPUT=runs/latent_statistics.pt bash scripts/dialoguesidon2/latent_statistics.sh
   ```

4. **Stage 2: separation model** (the released model: 2 nodes × 8 GPUs,
   16 segments of 30 s per GPU, 200k steps):

   ```bash
   VAE_CHECKPOINT=runs/ssl_vae/.../final.ckpt STATISTICS=runs/latent_statistics.pt \
   TRAIN_URLS=/data/dialogue/train VAL_URLS=/data/dialogue/valid OUTPUT_DIR=runs/separator \
     bash scripts/dialoguesidon2/train_separator.sh
   ```

5. **Release and export** — the release checkpoint (`model.safetensors` and
   `config.json`) and the `torch.export` programs the demo loads:

   ```bash
   CHECKPOINT=runs/separator/.../final.ckpt OUTPUT_DIR=release bash scripts/dialoguesidon2/export.sh
   ```

### Inference

```bash
# A release directory (e.g. a download of sarulab-speech/DialogueSidon2) or a training checkpoint
python -m sidon.dialoguesidon2.infer --model ./release --output-dir ./out dialogue1.wav dialogue2.wav
```

Each input gives `<name>_speaker1.wav` and `<name>_speaker2.wav` at 24 kHz; the
order of the two speakers is arbitrary. Without the repository, `spaces/dialoguesidon2/ds2_infer.py`
runs the exported programs with `torch` and `torchaudio` only (see the model
card).

### Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_dialoguesidon2*.py'
```

The tests run on CPU with tiny random models and synthetic data.

## Validation and troubleshooting

- Perform a quick syntax sweep with `python -m compileall src` before submitting
  jobs.
- Ensure CUDA kernels are available and match the Torch build; most sidon
  experiments assume a GPU-backed environment.
- If streaming from S3, check that the AWS CLI is installed and accessible in
  your job environment.
- The stack is ported from an internal codebase and only partially smoke-checked; if something breaks, please open an issue with details so we can follow up.
