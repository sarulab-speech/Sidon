#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=32:ngpus=4:mem=256gb
#PBS -l walltime=72:00:00
#PBS -P gag51394
#PBS -j oe
#PBS -k oed

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi
module load hpcx-mt/2.20
export PATH="/home/acc12576tt/miniconda3/bin/:$PATH"
export LD_LIBRARY_PATH="/home/acc12576tt/miniconda3/lib/:$LD_LIBRARY_PATH"
export MAIN_ADDR="$(hostname)"
export MAIN_PORT=$((10000 + RANDOM % 20000))
export HYDRA_FULL_ERROR=1

uv run python src/sidon/train.py \
  data=preprocessed \
  model=sidon_vocoder_pretrain \
  train=default \
  train.trainer.accelerator=gpu \
  train.trainer.devices=4 \
  train.trainer.strategy=ddp \
  hydra.run.dir=/groups/gag51394/users/nakata/sidon_runs/${env:PBS_JOBID,local_run}
