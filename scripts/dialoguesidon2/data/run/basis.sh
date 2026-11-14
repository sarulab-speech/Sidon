#!/bin/bash
# Basis Conversations 1500: from the downloaded release to the producer's shards.
#
#   bash run/basis.sh STAGE
#
# Stages, in order (each is resumable: finished units are skipped on a rerun):
#   s0       CPU, metadata only: windows, pairs and activity rules (basis/plan_pairs.py)
#   verify   CPU: FLAC headers and sha256 of the downloaded tracks; which segments are ready (basis/verify_audio.py)
#   screen   CPU: raw-audio bleed screen, clip table and the decisions report (basis/screen.py)
#   dnsmos   CPU, optional: DNSMOS scores (basis/dnsmos_windows.py); the shipped thresholds do not use them
#   restore  GPU: Sidon v0.1 restoration of the pair tracks and controls (basis/restore_inputs.py, run/restore.sh)
#   cut      CPU: restored-audio check, dev split and the producer's source tars (basis/cut.py)
#   produce  the producer (run/produce.sh), then the final shard check (basis/qc.py finalize)
#   qc       CPU: shard header check and the audio report (basis/qc.py)
#
# Variables:
#   BASIS_ROOT      the downloaded release (huggingface.co/datasets/basis-ai/basis-conversations-1500)
#   BASIS_WORK      the work directory: screen/, restore_input/, restored/, source/, shards/, qc/
#   DECISIONS       thresholds file (default: basis/thresholds.json, the values of the released data)
#   DNSMOS_MODEL    DNSMOS P.835 sig_bak_ovr.onnx (dnsmos stage only)
#   ALLOW_MISSING   TSV with a segment_id column: segments the cut may leave out (for example absent audio)
#   ACCEPT_S2_STOP  1 to write the source tars although an S2 stop condition holds
#   NOISE_DIR       noise shards of the producer (produce stage, see run/produce.sh)
#   MAX_SHORTFALL   windows a producer task may miss in the final check (default 0)
# and those of run/env.sh (PYTHON, PROCESSES, SCRATCH, LOG_DIR).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require BASIS_ROOT BASIS_WORK
stage=${1:?usage: $0 s0|verify|screen|dnsmos|restore|cut|produce|qc}
BASIS="$DATA_DIR/basis"
SCREEN="$BASIS_WORK/screen"
DECISIONS=${DECISIONS:-$BASIS/thresholds.json}
SHARDS="$BASIS_WORK/shards/basis_24k_120s"
common=(--root "$BASIS_ROOT" --screen "$SCREEN")
processes=$(data_processes)
data_log "basis_$stage"

case "$stage" in
  s0)
    "$PYTHON" "$BASIS/plan_pairs.py" --root "$BASIS_ROOT" --out "$SCREEN" --processes "$processes"
    ;;
  verify)
    "$PYTHON" "$BASIS/verify_audio.py" "${common[@]}" headers
    "$PYTHON" "$BASIS/verify_audio.py" "${common[@]}" checksums --processes "$processes"
    ;;
  screen)
    "$PYTHON" "$BASIS/screen.py" "${common[@]}" features --processes "$processes"
    "$PYTHON" "$BASIS/screen.py" "${common[@]}" run --processes "$processes"
    "$PYTHON" "$BASIS/screen.py" "${common[@]}" clip-table --processes 2
    "$PYTHON" "$BASIS/screen.py" "${common[@]}" rescore --decisions "$DECISIONS" --tag released
    ;;
  dnsmos)
    require DNSMOS_MODEL
    "$PYTHON" "$BASIS/dnsmos_windows.py" "${common[@]}" run --model "$DNSMOS_MODEL" --processes "$processes"
    coverage=$("$PYTHON" -c "import json, sys; print(json.load(open(sys.argv[1]))['thresholds'].get('dnsmos_clip_coverage', 0.25))" "$DECISIONS")
    "$PYTHON" "$BASIS/dnsmos_windows.py" "${common[@]}" summarize --coverage "$coverage"
    ;;
  restore)
    inputs=("${common[@]}" --data-root "$BASIS_WORK" --processes "$processes")
    "$PYTHON" "$BASIS/restore_inputs.py" "${inputs[@]}" build
    data_scratch basis_restore
    "$PYTHON" "$BASIS/restore_inputs.py" "${inputs[@]}" pending --parts 0-15 --into "$DATA_SCRATCH/pending"
    shopt -s nullglob
    trees=("$DATA_SCRATCH"/pending/p*/basis-1500)
    shopt -u nullglob
    if (( ${#trees[@]} )); then
      bash "$DATA_DIR/run/restore.sh" "$BASIS_WORK/restored" "${trees[@]}" || echo "Some restorations failed" >&2
    else
      echo "Every verified track already has a valid restored output"
    fi
    "$PYTHON" "$BASIS/restore_inputs.py" "${inputs[@]}" verify --parts 0-15
    ;;
  cut)
    # Recheck the restored outputs; cut.py refuses segments whose outputs failed unless ALLOW_MISSING lists them.
    "$PYTHON" "$BASIS/restore_inputs.py" "${common[@]}" --data-root "$BASIS_WORK" --processes "$processes" \
      verify --parts 0-15 || echo "The output check lists failures (restored/basis-1500.check.p*.json)" >&2
    if [[ -d "$SCREEN/dnsmos" ]]; then
      coverage=$("$PYTHON" -c "import json, sys; print(json.load(open(sys.argv[1]))['thresholds'].get('dnsmos_clip_coverage', 0.25))" "$DECISIONS")
      "$PYTHON" "$BASIS/dnsmos_windows.py" "${common[@]}" summarize --coverage "$coverage"
    fi
    extra=()
    [[ -n "${ALLOW_MISSING:-}" ]] && extra+=(--allow-missing "$ALLOW_MISSING")
    [[ "${ACCEPT_S2_STOP:-0}" == 1 ]] && extra+=(--accept-s2-stop)
    "$PYTHON" "$BASIS/cut.py" "${common[@]}" --decisions "$DECISIONS" --data-root "$BASIS_WORK" \
      --out "$BASIS_WORK/source" --processes "$processes" "${extra[@]}"
    ;;
  produce)
    bash "$DATA_DIR/run/produce.sh" "$BASIS_WORK/source" "$BASIS_WORK/shards" basis_24k_120s
    "$PYTHON" "$BASIS/qc.py" --shards "$SHARDS" --source "$BASIS_WORK/source" finalize \
      --max-shortfall "${MAX_SHORTFALL:-0}"
    ;;
  qc)
    "$PYTHON" "$BASIS/qc.py" --shards "$SHARDS" --source "$BASIS_WORK/source" headers \
      --max-shortfall "${MAX_SHORTFALL:-0}"
    "$PYTHON" "$BASIS/qc.py" --shards "$SHARDS" --source "$BASIS_WORK/source" audio --out "$BASIS_WORK/qc" \
      --screen "$SCREEN" --root "$BASIS_ROOT"
    ;;
  *)
    echo "Unknown stage $stage" >&2
    exit 2
    ;;
esac
echo "basis $stage finished at $(date -Is)"
