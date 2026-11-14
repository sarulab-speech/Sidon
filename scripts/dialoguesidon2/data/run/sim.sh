#!/bin/bash
# Simulated two-speaker conversations from LibriTTS-R, stage by stage. Every stage is CPU only.
#
#   bash run/sim.sh STAGE
#
# Stages, in order:
#   pack      (optional) pack an extracted LibriTTS-R release (LIBRITTS_RELEASE) into $SIM_WORK/libritts_tars,
#             the tar.gz layout the index reads; then set LIBRITTS_DIR=$SIM_WORK/libritts_tars
#   index     index the three train subsets: trim, levels, VAD runs and a PCM cache ($SIM_WORK/index)
#   fit       (optional) refit the turn-taking table from the OpenYap-1k outputs (OY_MANIFEST: the OpenYap source
#             manifest.tsv; OY_FRAMES: the frames/ directory of the OpenYap crosstalk screen) into
#             $SIM_WORK/fit/fit.json; then set SIM_FIT to it. Without it the shipped sim/turn_fit.json is used.
#   dry-run   plan without rendering ($SIM_WORK/dry_run/plan_dry.{tsv,json}): record counts and speech budget
#   simulate  plan ($SIM_WORK/clean/plan.{tsv,json}; must equal the dry run when there is one), render the clean
#             24 kHz tracks, then the S0 and S1 checks ($SIM_WORK/qc/S0.json, S1.json); exits 1 if a check fails
#   cut       stereo 48 kHz source tars for the producer ($SIM_WORK/source); refuses unless S0 and S1 passed
#             (ACCEPT_S1_STOP=1 proceeds after a failed S1)
#   produce   the producer (run/produce.sh; NOISE_DIR required) into $SIM_WORK/shards/sim_24k_120s, then the final
#             check (MAX_SHORTFALL windows per task may be missing, default 0) and N_SHARDS.txt
#   qc        audio statistics of the produced shards and 20 WAV exports ($SIM_WORK/qc/audio)
#
# Variables: SIM_WORK (required), LIBRITTS_DIR (index), LIBRITTS_RELEASE (pack), OY_MANIFEST and OY_FRAMES (fit),
# SIM_FIT (default: sim/turn_fit.json), MAX_USES (default 4), plus those of run/env.sh.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
stage=${1:?usage: $0 pack|index|fit|dry-run|simulate|cut|produce|qc}
require SIM_WORK
SIM="$DATA_DIR/sim"
fit=${SIM_FIT:-$SIM/turn_fit.json}
max_uses=${MAX_USES:-4}
processes=$(data_processes)
data_log "sim_$stage"
case "$stage" in
  pack)
    require LIBRITTS_RELEASE
    "$PYTHON" "$SIM/pack_libritts.py" --release "$LIBRITTS_RELEASE" --out "$SIM_WORK/libritts_tars" \
      --processes "$processes"
    ;;
  index)
    require LIBRITTS_DIR
    "$PYTHON" "$SIM/index_libritts.py" --libritts "$LIBRITTS_DIR" --out "$SIM_WORK/index" run --processes "$processes"
    "$PYTHON" "$SIM/index_libritts.py" --libritts "$LIBRITTS_DIR" --out "$SIM_WORK/index" merge
    ;;
  fit)
    require OY_MANIFEST OY_FRAMES
    "$PYTHON" "$SIM/fit_turns.py" --manifest "$OY_MANIFEST" --frames "$OY_FRAMES" --out "$SIM_WORK/fit" \
      --processes "$processes"
    ;;
  dry-run)
    "$PYTHON" "$SIM/simulate.py" --work "$SIM_WORK" plan --dry-run --fit "$fit" --max-uses "$max_uses"
    ;;
  simulate)
    [[ -f "$SIM_WORK/index/index.tsv" ]] || { echo "No $SIM_WORK/index/index.tsv; run the index stage" >&2; exit 1; }
    "$PYTHON" "$SIM/simulate.py" --work "$SIM_WORK" plan --fit "$fit" --max-uses "$max_uses"
    if [[ -f "$SIM_WORK/dry_run/plan_dry.json" ]]; then
      "$PYTHON" - "$SIM_WORK" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
seen = json.load(open(root / "dry_run" / "plan_dry.json"))["plan_hash"]
now = json.load(open(root / "clean" / "plan.json"))["plan_hash"]
if seen != now:
    sys.exit(f"The plan {now[:12]} differs from the dry run {seen[:12]} (the index, fit or settings changed)")
print(f"Plan {now[:12]} equals the dry run")
PY
    fi
    "$PYTHON" "$SIM/simulate.py" --work "$SIM_WORK" render --processes "$processes"
    "$PYTHON" "$SIM/qc.py" --work "$SIM_WORK" s0 --processes "$processes"
    "$PYTHON" "$SIM/qc.py" --work "$SIM_WORK" s1 --fit "$fit"
    ;;
  cut)
    accept=()
    [[ "${ACCEPT_S1_STOP:-0}" == 1 ]] && accept=(--accept-s1-stop)
    "$PYTHON" "$SIM/cut.py" --work "$SIM_WORK" --out "$SIM_WORK/source" --processes "$processes" "${accept[@]}"
    ;;
  produce)
    require NOISE_DIR
    bash "$DATA_DIR/run/produce.sh" "$SIM_WORK/source" "$SIM_WORK/shards" sim_24k_120s
    "$PYTHON" "$SIM/qc.py" --work "$SIM_WORK" --shards "$SIM_WORK/shards/sim_24k_120s" --source "$SIM_WORK/source" \
      finalize --max-shortfall "${MAX_SHORTFALL:-0}"
    ;;
  qc)
    "$PYTHON" "$SIM/qc.py" --work "$SIM_WORK" --shards "$SIM_WORK/shards/sim_24k_120s" --source "$SIM_WORK/source" \
      audio --out "$SIM_WORK/qc/audio"
    ;;
  *)
    echo "Unknown stage $stage" >&2
    exit 2
    ;;
esac
echo "sim $stage finished at $(date -Is)"
