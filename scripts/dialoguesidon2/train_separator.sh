#!/usr/bin/env bash
# Stage 2: train the separation model (fully fine-tuned w2v-BERT 2.0, 13 layers, plus the readout head) on
# two-speaker dialogue shards, with the stage 1 checkpoint and its latent statistics. The released model: 2 nodes x 8
# GPUs, 16 segments of 30 s per GPU, 200k steps.
#
#   VAE_CHECKPOINT=runs/ssl_vae/.../final.ckpt STATISTICS=runs/latent_statistics.pt \
#   TRAIN_URLS=/data/dialogue/train VAL_URLS=/data/dialogue/valid OUTPUT_DIR=runs/separator \
#     bash scripts/dialoguesidon2/train_separator.sh [extra Hydra overrides]
#
# Multi-node (run on every node; the defaults expect NNODES=2):
#   NNODES=2 NODE_RANK=<0|1> MASTER_ADDR=<host of rank 0> MASTER_PORT=29500 bash scripts/dialoguesidon2/train_separator.sh
# Resume with the extra override train.ckpt_path=<checkpoint>. The result is
# $OUTPUT_DIR/<run>/version_*/checkpoints/final.ckpt; package it with python -m sidon.dialoguesidon2.release.
set -euo pipefail
: "${VAE_CHECKPOINT:?the stage 1 checkpoint}"
: "${STATISTICS:?the latent statistics file}"
: "${TRAIN_URLS:?comma-separated training shard directories}"
: "${VAL_URLS:?comma-separated validation shard directories}"
OUTPUT_DIR=${OUTPUT_DIR:-runs/dialoguesidon2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1

torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node="$GPUS_PER_NODE" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m sidon.dialoguesidon2.train \
  "model.cfg.vae_checkpoint=$VAE_CHECKPOINT" "model.cfg.statistics_path=$STATISTICS" \
  "data.datamodule.train_urls=[$TRAIN_URLS]" "data.datamodule.val_urls=[$VAL_URLS]" \
  "train.trainer.devices=$GPUS_PER_NODE" "train.trainer.num_nodes=$NNODES" \
  "train.logger.save_dir=$OUTPUT_DIR" "hydra.run.dir=$OUTPUT_DIR/hydra" \
  "$@"
