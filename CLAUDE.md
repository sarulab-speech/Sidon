# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sidon is a speech restoration research system. It pairs a LoRA-adapted w2v-BERT 2.0 feature predictor with a DAC-based vocoder to restore degraded speech. A separate branch (`dialogue`) extends the system with **GENESES** — a flow-matching two-speaker dialogue separator built on MMDiT + DACVAE.

## Environment Setup

```bash
uv sync               # install all dependencies
python -m compileall src   # quick syntax sweep before submitting cluster jobs
```

The project uses `uv` for dependency management (Python 3.10+, PyTorch 2.6+, CUDA required). A `.venv/` is present in the repo root.

## Common Commands

### Training
```bash
# Feature predictor pretraining (stage 1)
uv run python -m sidon.train model=sidon_feature_predictor data=preprocessed

# Vocoder pretraining (stage 2)
uv run python -m sidon.train model=sidon_vocoder_pretrain data=preprocessed train=gan

# Vocoder finetuning (stage 3)
uv run python -m sidon.train model=sidon_vocoder_finetune data=preprocessed_48k train=gan \
  model.cfg.ssl_model_name=/path/to/feature_predictor.ckpt \
  model.cfg.pretrain_path=/path/to/vocoder_pretrain.ckpt

# GENESES dialogue separation
uv run python -m sidon.train model=geneses_dialogue data=dialogue_preprocessed

# Resume from checkpoint
uv run python -m sidon.train ... train.ckpt_path=/path/to/last.ckpt
```

### Preprocessing (generate WebDataset shards)
```bash
uv run python -m sidon.preprocess \
  data=webdataset_preprocess_24k \
  preprocess.writer_name=my_preprocessed_run
```

Hydra writes run artifacts under `outputs/<timestamped_run>/` and shards into `${preprocess.output_root}/{writer_name}/{split}/{job_id}/`.

### Inference (GENESES)
```bash
# Batch mode (directory of wav files)
python infer_geneses.py --input-dir ./wavs --checkpoint ./ckpt --output-dir ./out \
  --num-steps 100 --chunk-seconds 20 --overlap-seconds 1

# Single video/audio file
python infer_geneses.py --input-video input.mp4 --checkpoint ./ckpt \
  --output-wav separated.wav [--output-video output.mp4]
```

### PBS cluster submission
```bash
qsub scripts/pbs/train_geneses_dialogue.sh
qsub scripts/pbs/diffusion_dialogue_sidon_small.sh
```

## Architecture

### Training entry point
`src/sidon/train.py` — Hydra entrypoint. Resolves `cfg.data.datamodule` and `cfg.model.lightning_module` from config and calls `trainer.fit()`. The top-level config is `config/config.yaml`; override via `model=<name>` and `data=<name>`.

### Model variants and their Lightning modules

| Config key | Lightning module | Description |
|---|---|---|
| `sidon_feature_predictor` | `sidon.model.sidon.lightning_module.FeaturePredictorLightningModule` | LoRA-adapts w2v-BERT student to predict clean SSL features from noisy input |
| `sidon_vocoder_pretrain` / `sidon_vocoder_finetune` | `sidon.model.sidon.lightning_module.SidonLightningModule` | DAC decoder + GAN discriminator; pretrain on clean SSL, finetune on denoised SSL |
| `geneses_dialogue` | `sidon.model.geneses.lightning_module.GenesesLightningModule` | Flow-matching separator: MMDiT + DACVAE + w2v-BERT conditioning |
| `diffusion_dialogue_sidon*` | `sidon.model.dialogue_sidion.lightning_module.*` | Diffusion-based dialogue separation variants |

### GENESES data flow (dialogue separation)
1. Batch delivers `input_wav` [B, 2, T] (stereo/two-speaker) and `noisy_16k_mixture` [B, T]
2. Both speakers encoded via frozen `DACVAE` → VAE latents [B, C, T_latent]
3. Noisy mixture extracted via `SSLFeatureExtractor` (w2v-BERT layer features) → `ssl_merged` conditioning
4. `MMDiT` (three-modality: ssl_merged + vae_1 + vae_2) trained with affine flow-matching loss
5. Inference uses `ODESolver` from the `flow_matching` library to sample separated latents, then decoded by `DACVAE`

### Key components
- `src/sidon/model/geneses/components.py` — `MMDiT` wrapper with sinusoidal positional embeddings and `TimestepEmbedder`
- `src/sidon/model/geneses/dacvae.py` — Thin wrapper around a TorchScript-exported DAC VAE checkpoint (loaded from HF Hub: `koacai/geneses`)
- `src/sidon/model/geneses/ssl_feature_extractor.py` — Extracts intermediate layer features from w2v-BERT 2.0 (`facebook/w2v-bert-2.0`)
- `src/sidon/model/losses.py` — `DACLoss` (mel) and `GANLoss` wrappers

### Data pipeline
- `src/sidon/data/datamodule.py` — `PreprocessedDataModule`: loads pre-materialized WebDataset shards containing `input_wav.pth`, `noisy_input_wav.pth`, and optional `ssl_inputs.pickle` / `noisy_ssl_inputs.pickle`
- `src/sidon/data/preprocess/dialogue_datamodule.py` — `DialogueDatasetDataModule` (online) and `PreprocessedDialogueDataModule` (from shards): handles stereo multi-speaker audio, VAD filtering, noise augmentation, per-channel splitting and re-merging
- `src/sidon/preprocess.py` — Materializes datamodule output into WebDataset shards via parallel writer processes

### Config structure (`config/`)
```
config/
  config.yaml          # top-level defaults: data=preprocessed, model=sidon_vocoder_pretrain
  preprocess.yaml      # preprocessing entry point defaults
  model/               # per-model configs (cfg.* → Lightning module constructor)
  data/                # datamodule configs
  train/default.yaml   # trainer, logger (WandB), callbacks, scheduler
```

All Hydra overrides are passed as CLI args (e.g., `model.cfg.ssl_model_name=...`).

## Cluster (PBS) Notes

- Scripts in `scripts/pbs/` use `mpirun` for multi-node, multi-GPU training
- `WANDB_NAME` is auto-set from `$PBS_JOBID`; override with `export WANDB_NAME=...`
- `.venv/bin/python` is called directly (not `uv run`) inside PBS jobs after activating `.venv`
- Job output logs go to `<script_name>.o<jobid>` in the working directory
