#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1
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
  data=preprocessed_48k \
  data.datamodule.batch_size=4 \
  model=sidon_vocoder_finetune \
  "model.cfg.ssl_model_name='/home/acc12576tt/github.com/Sidon/sidon/t2h90k5j/checkpoints/epoch=13-step=388305.ckpt'" \
  "model.cfg.pretrain_path='/home/acc12576tt/github.com/Sidon/sidon/60n5ebis/checkpoints/epoch=2-step=366468.ckpt'" \
  train=gan \
  train.trainer.gradient_clip_val=null \
  hydra.run.dir=/groups/gag51394/users/nakata/sidon_runs/${PBS_JOBID}