#!/bin/bash
#PBS -l rt_QF=2
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -W group_list=qgah50068
set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"
source /etc/profile.d/modules.sh

module load hpc_sdk/24.9 nvhpc-hpcx-cuda12/24.9
export PATH="/work/gj18/e43001/miniconda3/bin/:$PATH"
export LD_LIBRARY_PATH="/work/gj18/e43001/miniconda3/lib/"
export MAIN_ADDR="$(hostname)"
export MAIN_PORT=$((10000 + RANDOM % 20000))
export HYDRA_FULL_ERROR=1
# limit per-process thread pools so dataloader workers don't oversubscribe the node
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi
num_gpus=$(nvidia-smi -L | wc -l)
num_nodes=$(sort -u $PBS_NODEFILE | wc -l)
#multiply num_nodes by num_procs to get total number of processes
num_procs=$((num_nodes*num_gpus))
export WANDB_NAME="ssl_vae_8_${PBS_JOBID}"

unset OMPI_MCA_mca_base_env_list
mpirun  \
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x MAIN_ADDR \
    -x MAIN_PORT \
    -x HYDRA_FULL_ERROR \
    -bind-to none \
     -np $num_procs -map-by ppr:$num_gpus:node -hostfile $PBS_NODEFILE \
    .venv/bin/python src/sidon/train.py \
  data=dialogue_preprocessed \
  data.datamodule.batch_size=4 \
  model=ssl_vae \
  train=gan \
  train.trainer.gradient_clip_val=null \
  train.trainer.precision=bf16-mixed \
  hydra.run.dir=./sidon_runs/${PBS_JOBID} \
  +train.trainer.num_nodes=$num_nodes +train.trainer.devices=$num_gpus \
  model.cfg.vae.latent_dim=8
