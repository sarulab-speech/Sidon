#!/usr/bin/env bash
# Package a trained separation model: the release checkpoint (model.safetensors + config.json) and the torch.export
# programs read by spaces/dialoguesidon2/ds2_infer.py (separator.pt2, vae_decoder.pt2, metadata.json). CPU is enough.
#
#   CHECKPOINT=runs/separator/.../final.ckpt OUTPUT_DIR=release bash scripts/dialoguesidon2/export.sh
set -euo pipefail
: "${CHECKPOINT:?the stage 2 training checkpoint}"
: "${OUTPUT_DIR:?the directory to write}"
WINDOW_SECONDS=${WINDOW_SECONDS:-30}
OVERLAP_SECONDS=${OVERLAP_SECONDS:-5}

python -m sidon.dialoguesidon2.release --checkpoint "$CHECKPOINT" --out "$OUTPUT_DIR/checkpoint"
python -m sidon.dialoguesidon2.export --model "$OUTPUT_DIR/checkpoint" --out "$OUTPUT_DIR/export" \
  --window-seconds "$WINDOW_SECONDS" --overlap-seconds "$OVERLAP_SECONDS"
