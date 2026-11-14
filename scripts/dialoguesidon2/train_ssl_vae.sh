#!/usr/bin/env bash
# Stage 1: train the SSL-VAE (frozen w2v-BERT 2.0 layer 8, VAE bottleneck, DAC decoder) on Sidon's single-speaker
# preprocessed shards. The released model: 8 GPUs, 4 two-track 10 s examples per GPU, 510k optimizer steps (two per
# batch: discriminator and generator).
#
#   TRAIN_URLS=/data/sidon/train VAL_URLS=/data/sidon/valid OUTPUT_DIR=runs/ssl_vae \
#     bash scripts/dialoguesidon2/train_ssl_vae.sh [extra Hydra overrides]
#
# Multi-node (run on every node; the trainer's num_nodes must match NNODES):
#   NNODES=2 NODE_RANK=<0|1> MASTER_ADDR=<host of rank 0> MASTER_PORT=29500 bash scripts/dialoguesidon2/train_ssl_vae.sh
# The result is $OUTPUT_DIR/<run>/version_*/checkpoints/final.ckpt (model.cfg.vae_checkpoint of stage 2).
set -euo pipefail
: "${TRAIN_URLS:?comma-separated training shard directories}"
: "${VAL_URLS:?comma-separated validation shard directories}"
OUTPUT_DIR=${OUTPUT_DIR:-runs/dialoguesidon2_ssl_vae}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1

torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node="$GPUS_PER_NODE" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m sidon.dialoguesidon2.train \
  model=dialoguesidon2_ssl_vae data=dialoguesidon2_ssl_vae train=dialoguesidon2_ssl_vae \
  "data.datamodule.train_urls=[$TRAIN_URLS]" "data.datamodule.val_urls=[$VAL_URLS]" \
  "train.trainer.devices=$GPUS_PER_NODE" "train.trainer.num_nodes=$NNODES" \
  "train.logger.save_dir=$OUTPUT_DIR" "hydra.run.dir=$OUTPUT_DIR/hydra" \
  "$@"
