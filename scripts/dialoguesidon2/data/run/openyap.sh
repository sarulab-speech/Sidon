#!/bin/bash
# OpenYap-1k pipeline, one stage per call (see the README for what each stage reads and writes):
#
#   bash run/openyap.sh prescreen   # CPU, metadata only (minutes)
#   bash run/openyap.sh scan        # CPU, transcripts only (minutes)
#   bash run/openyap.sh screen      # CPU, decodes every raw track (hours on 64 cores)
#   bash run/openyap.sh restore     # GPU: Sidon v0.1 on the tracks of kept conversations, then the output check
#   bash run/openyap.sh cut         # CPU: post-restoration check, dev split, source tars
#   bash run/openyap.sh produce     # CPU: the producer (16 tasks), then the final check of its shards
#   bash run/openyap.sh qc          # CPU: header and audio checks of the shards
#
# Variables:
#   OY_ROOT        the release's conversations/ directory (conv_<12 hex>/speaker_{a,b}.flac, transcripts, metadata)
#   OY_WORK        output directory (screen/, restore_input/, restored/, source/, shards/, qc/)
#   NOISE_DIR      noise shards for the producer (produce stage)
#   OY_DECISIONS   thresholds JSON (default: openyap/thresholds.json)
#   OY_ACCEPT_S2_STOP=1   write the tars although a post-restoration stop condition holds (cut stage)
#   MAX_SHORTFALL  windows a producer task may drop by itself (default 0)
# and PYTHON, PROCESSES, SCRATCH, LOG_DIR (run/env.sh).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require OY_ROOT OY_WORK
stage=${1:?usage: $0 prescreen|scan|screen|restore|cut|produce|qc}
OY="$DATA_DIR/openyap"
SCREEN="$OY_WORK/screen"
EXTRA="$SCREEN/transcript_lags_wide.json"
WRITER=openyap_24k_120s
decisions=()
[[ -n "${OY_DECISIONS:-}" ]] && decisions=(--decisions "$OY_DECISIONS")
processes=$(data_processes)
data_log "openyap_$stage"
mkdir -p "$SCREEN"

case "$stage" in
  prescreen)
    "$PYTHON" "$OY/prescreen.py" --root "$OY_ROOT" --out "$SCREEN" --processes "$processes"
    ;;
  scan)
    "$PYTHON" "$OY/scan_transcript_lags.py" --root "$OY_ROOT" --s0 "$SCREEN" --out "$EXTRA" --processes "$processes"
    ;;
  screen)
    [[ -f "$SCREEN/s0_windows.tsv" && -f "$EXTRA" ]] || { echo "Run the prescreen and scan stages first" >&2; exit 1; }
    data_scratch openyap_screen
    "$PYTHON" "$OY/screen_crosstalk.py" --root "$OY_ROOT" --out "$SCREEN" run --processes "$processes" \
      --scratch "$DATA_SCRATCH"
    "$PYTHON" "$OY/screen_crosstalk.py" --root "$OY_ROOT" --out "$SCREEN" rescore "${decisions[@]}" \
      --extra-reject "$EXTRA"
    ;;
  restore)
    common=(--root "$OY_ROOT" --s0 "$SCREEN" --work "$OY_WORK" --processes 32)
    "$PYTHON" "$OY/restore_inputs.py" "${common[@]}" build
    data_scratch openyap_restore
    "$PYTHON" "$OY/restore_inputs.py" "${common[@]}" pending --into "$DATA_SCRATCH/pending"
    if [[ -d "$DATA_SCRATCH/pending/openyap-1k" ]]; then
      bash "$DATA_DIR/run/restore.sh" "$OY_WORK/restored" "$DATA_SCRATCH/pending/openyap-1k" \
        || echo 'Some restorations failed; the check below lists them, and rerunning this stage redoes only those' >&2
    else
      echo 'Every track already has a valid output'
    fi
    "$PYTHON" "$OY/restore_inputs.py" "${common[@]}" verify
    ;;
  cut)
    "$PYTHON" "$OY/restore_inputs.py" --root "$OY_ROOT" --s0 "$SCREEN" --work "$OY_WORK" --processes 32 verify
    accept=()
    [[ "${OY_ACCEPT_S2_STOP:-0}" == 1 ]] && accept=(--accept-s2-stop)
    "$PYTHON" "$OY/cut_windows.py" --root "$OY_ROOT" --screen "$SCREEN" "${decisions[@]}" --extra-reject "$EXTRA" \
      --restored "$OY_WORK/restored/openyap-1k" --out "$OY_WORK/source" --processes "$processes" "${accept[@]}"
    ;;
  produce)
    require NOISE_DIR
    bash "$DATA_DIR/run/produce.sh" "$OY_WORK/source" "$OY_WORK/shards" "$WRITER"
    "$PYTHON" "$OY/qc_shards.py" --shards "$OY_WORK/shards/$WRITER" --source "$OY_WORK/source" \
      --max-shortfall "${MAX_SHORTFALL:-0}" finalize
    ;;
  qc)
    "$PYTHON" "$OY/qc_shards.py" --shards "$OY_WORK/shards/$WRITER" --source "$OY_WORK/source" \
      --max-shortfall "${MAX_SHORTFALL:-0}" headers
    "$PYTHON" "$OY/qc_shards.py" --shards "$OY_WORK/shards/$WRITER" --source "$OY_WORK/source" audio \
      --out "$OY_WORK/qc" --screen "$SCREEN" --root "$OY_ROOT"
    ;;
  *)
    echo "Unknown stage $stage" >&2
    exit 2
    ;;
esac
echo "openyap $stage finished at $(date -Is)"
