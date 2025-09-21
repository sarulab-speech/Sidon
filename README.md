# Sidon
[![ArXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/abs/TODO)
[![Gradio Demo](https://img.shields.io/badge/Gradio-demo-orange.svg)](https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-demo)](https://huggingface.co/spaces/Wataru/SidonSamples)


Large-scale text-to-speech (TTS) systems are bottlenecked by the scarcity of clean, multilingual recordings. Sidon tackles this by pairing a fast, open-source speech restoration model with reproducible tooling so researchers can turn noisy in-the-wild corpora into studio-quality datasets that scale across dozens of languages.

Sidon consists of two stages: a w2v-BERT 2.0 feature predictor finetuned to cleanse representations from degraded speech, and a vocoder trained to synthesise restored waveforms from those features. The stack achieves restoration quality comparable to Miipher—Google's internal speech restoration pipeline—while running up to 3,390× faster than real time on a single GPU. We also observe that training downstream TTS models on Sidon-cleansed automatic speech recognition corpora improves zero-shot synthesis quality. This repository releases the code, configs, and models needed to reproduce Sidon's dataset cleansing workflow for the community.

## Requirements

- Python 3.10+
- Recent PyTorch / CUDA stack (tested with `torch>=2.8`, `torchaudio>=2.8`)
- `uv` for dependency management (or an equivalent toolchain you are
  comfortable with)

Install project dependencies:

```bash
uv sync
```

If you rely on a different environment manager, replicate the dependencies
listed in `pyproject.toml`.

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
     data=preprocessed
   ```

   Capture the checkpoint path; it will be referenced as `model.cfg.pretrain_path`
   during finetuning.

3. **Vocoder finetuning** — Warm-starts from the pretraining weights and swaps
   in the denoised SSL features predicted by the feature predictor.

   ```bash
   uv run python -m sidon.train \
     model=sidon_vocoder_finetune \
     data=preprocessed_48k \
     model.cfg.ssl_model_name=/path/to/feature_predictor.ckpt \
     model.cfg.pretrain_path=/path/to/vocoder_pretrain.ckpt
   ```

Adjust optimiser, scheduler, or trainer parameters via the files in
`config/model/` and `config/train/`, and use `train.ckpt_path` to resume a run.

## Validation and troubleshooting

- Perform a quick syntax sweep with `python -m compileall src` before submitting
  jobs.
- Ensure CUDA kernels are available and match the Torch build; most sidon
  experiments assume a GPU-backed environment.
- If streaming from S3, check that the AWS CLI is installed and accessible in
  your job environment.
- The stack is ported from an internal codebase and only partially smoke-checked; if something breaks, please open an issue with details so we can follow up.
