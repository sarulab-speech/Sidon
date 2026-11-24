#!/bin/bash
# Restore every FLAC below one or more input trees with Sidon v0.1 (common/sidon_restore.py), one process per GPU.
# GPU stage.
#
#   bash run/restore.sh OUTPUT_DIR TREE [TREE ...]
#
# Each TREE is a directory named after the corpus (for example .../pending/openyap-1k); its files are written to
# OUTPUT_DIR/<tree name>/<relative path>. Existing outputs are overwritten, so pass trees that hold only the tracks
# still to be restored (the corpus scripts' `pending` commands build them). GPUS (default: all visible GPUs) and
# WORKERS (data loader workers per GPU, default 4) tune the run; HF_HOME selects the model cache.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[[ $# -ge 2 ]] || { echo "usage: $0 OUTPUT_DIR TREE [TREE ...]" >&2; exit 2; }
output=$1
shift
gpus=${GPUS:-$(nvidia-smi -L | wc -l)}
(( gpus > 0 )) || { echo "No GPU found" >&2; exit 1; }
data_log restore
status=0
for tree in "$@"; do
  [[ -d "$tree" ]] || { echo "Skipping $tree (no such directory)"; continue; }
  echo "Restoring $tree on $gpus GPUs"
  pids=()
  for ((i = 0; i < gpus; i++)); do
    CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=16 "$PYTHON" "$DATA_DIR/common/sidon_restore.py" --input_dir "$tree" \
      --output_dir "$output" --num_shards "$gpus" --shard_idx "$i" --num_workers "${WORKERS:-4}" &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "Restoration of $tree failed on GPU $i" >&2; status=1; }
  done
done
echo "Restoration finished at $(date -Is) with status $status"
exit "$status"
