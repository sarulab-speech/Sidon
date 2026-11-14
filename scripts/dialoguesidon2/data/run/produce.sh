#!/bin/bash
# Run the producer (scripts/dialoguesidon2/data/producer) on one corpus's source tars. CPU stage (no GPU).
#
#   NOISE_DIR=... bash run/produce.sh SOURCE_DIR OUTPUT_ROOT WRITER_NAME
#
# SOURCE_DIR holds {train,dev}/**/*.tar, and for the pre-cut corpora manifest.tsv and summary.json with
# "status": "complete" (written by the corpus's cut stage); then each task's record count is checked against the
# manifest. Without manifest.tsv (the base root, packed by common/pack_sources.py) the counts are not checked.
# The shards go to OUTPUT_ROOT/WRITER_NAME/{train,valid}/t<NN>/. Each task needs at least 8 training source tars
# (one per data loader worker). Variables:
#   NOISE_DIR      directory of WebDataset noise shards (required)
#   TASKS          producer tasks to run here (default 0-15: all, concurrently; e.g. TASKS=3 on one machine of
#                  a job array). The final check runs when all 16 tasks have a done marker.
#   MAX_SHORTFALL  windows a task may lack, dropped by the producer itself (default 0, see the README)
#   KEY_PATTERN    regex of the corpus's source keys, checked on the pre-flight record (optional)
# A task with a done marker (_done/t<NN>.json) is skipped; a task without one has its partial output moved to
# OUTPUT_ROOT/_partial/ before it runs again. A task is done when its record count matches the windows of its
# source tars (common/shards_qc.py) and its log holds no Traceback or ValueError.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[[ $# -eq 3 ]] || { echo "usage: NOISE_DIR=... $0 SOURCE_DIR OUTPUT_ROOT WRITER_NAME" >&2; exit 2; }
source_dir=$(cd "$1" && pwd)
output_root=$2
writer=$3
require NOISE_DIR
mkdir -p "$output_root"
output_root=$(cd "$output_root" && pwd)
out="$output_root/$writer"
data_log "produce_$writer"
data_scratch "produce-$writer"
scratch=$DATA_SCRATCH
logs=${LOG_DIR:-$out/_logs}
mkdir -p "$out/_done" "$logs"
check_args=()
finalize_args=()
if [[ -f "$source_dir/manifest.tsv" ]]; then
  if ! grep -q '"status": "complete"' "$source_dir/summary.json" 2>/dev/null; then
    echo "Source shards are not complete ($source_dir/summary.json)" >&2
    exit 1
  fi
  check_args=(--expected "$logs/expected.json")
  finalize_args=(--source "$source_dir")
else
  echo "No $source_dir/manifest.tsv: record counts are not checked"
fi
export PYTHONPATH="$REPO_DIR/src:$DATA_DIR${PYTHONPATH:+:$PYTHONPATH}" HYDRA_FULL_ERROR=1
pre=(--out "$logs/preflight.json" --source "$source_dir" --noise "$NOISE_DIR")
[[ -n "${KEY_PATTERN:-}" ]] && pre+=(--key-pattern "$KEY_PATTERN")
if ! (cd "$scratch" && "$PYTHON" "$DATA_DIR/producer/preflight.py" "${pre[@]}"); then
  echo "Pre-flight failed; see $logs/preflight.json" >&2
  exit 1
fi
if [[ -f "$source_dir/manifest.tsv" ]]; then
  "$PYTHON" "$DATA_DIR/producer/tasks.py" expected --source "$source_dir" > "$logs/expected.json"
fi
tasks=$("$PYTHON" -c "import sys; sys.path.insert(0, sys.argv[1]); from common import core
print(' '.join(map(str, core.parse_parts(sys.argv[2], 16))))" "$DATA_DIR" "${TASKS:-0-15}")

run_task() {
  local index=$1 name status=0
  name=$(printf 't%02d' "$index")
  if [[ -f "$out/_done/$name.json" ]]; then
    echo "$name already done"
    return 0
  fi
  for split in train valid; do
    if [[ -e "$out/$split/$name" ]]; then
      local aside
      aside="$output_root/_partial/$(date +%Y%m%d_%H%M%S)_$name/$split"
      mkdir -p "$aside"
      mv "$out/$split/$name" "$aside/"
      echo "Moved the partial $split/$name to $aside"
    fi
  done
  mkdir -p "$scratch/$name"
  (cd "$scratch/$name" && PRODUCER_TASK_NAME="$name" "$PYTHON" "$DATA_DIR/producer/produce.py" \
    data=dialoguesidon2_producer "data.datamodule.train_wds_patterns=[$source_dir/train]" \
    "data.datamodule.val_wds_patterns=[$source_dir/dev]" "data.datamodule.noise_path=[$NOISE_DIR]" \
    "data.datamodule.shard_idx=$index" "preprocess.output_root=$output_root" "preprocess.writer_name=$writer" \
    "hydra.run.dir=$logs/hydra/$name") > "$logs/$name.log" 2>&1 || status=$?
  if (( status != 0 )); then
    echo "$name failed with status $status; see $logs/$name.log" >&2
    return 1
  fi
  "$PYTHON" "$DATA_DIR/producer/tasks.py" check --shards "$out" --task "$index" "${check_args[@]}" \
    --log "$logs/$name.log" --max-shortfall "${MAX_SHORTFALL:-0}"
}

pids=()
for index in $tasks; do
  run_task "$index" &
  pids+=($!)
done
failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "A producer task failed" >&2; failed=1; }
done
(( failed == 0 )) || { echo 'Run again to redo the failed tasks (finished ones are skipped)' >&2; exit 1; }
done_count=$(find "$out/_done" -name 't*.json' | wc -l)
if (( done_count == 16 )); then
  "$PYTHON" "$DATA_DIR/producer/tasks.py" finalize --shards "$out" "${finalize_args[@]}" \
    --max-shortfall "${MAX_SHORTFALL:-0}"
else
  echo "$done_count of 16 tasks done; the final check runs when all are done"
fi
echo "Producer finished at $(date -Is)"
