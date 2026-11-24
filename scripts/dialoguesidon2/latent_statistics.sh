#!/usr/bin/env bash
# Between the stages: per-dimension mean and variance of the SSL-VAE posterior draws on dialogue segments, which
# standardize the separation targets. As for the released model: both clean speakers of 10 s segments of the
# dialogue training shards, 1000 batches of 12, bf16 autocast, one GPU, seed 42.
#
#   VAE_CHECKPOINT=runs/ssl_vae/.../final.ckpt TRAIN_URLS=/data/dialogue/train OUTPUT=runs/latent_statistics.pt \
#     bash scripts/dialoguesidon2/latent_statistics.sh [extra Hydra overrides]
set -euo pipefail
: "${VAE_CHECKPOINT:?the stage 1 checkpoint}"
: "${TRAIN_URLS:?comma-separated dialogue training shard directories}"
: "${OUTPUT:?the statistics file to write}"
export HYDRA_FULL_ERROR=1

python -m sidon.dialoguesidon2.statistics \
  "statistics.vae_checkpoint=$VAE_CHECKPOINT" "statistics.train_urls=[$TRAIN_URLS]" "statistics.output=$OUTPUT" \
  "hydra.run.dir=$(dirname "$OUTPUT")/statistics_hydra" \
  "$@"
