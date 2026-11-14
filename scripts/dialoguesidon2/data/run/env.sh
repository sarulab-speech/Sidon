# Sourced by the stage scripts in this directory. Nothing here submits jobs: run the stage scripts directly, or wrap
# them in your scheduler's job script (each stage says whether it needs GPUs or only CPUs).
#
# Variables (set them in the environment before calling a stage script):
#   PYTHON        python with the Sidon environment (default: python)
#   PROCESSES     worker processes of CPU stages (default: min(64, available CPUs))
#   SCRATCH       fast local scratch for temporary files (default: a new directory under ${TMPDIR:-/tmp})
#   LOG_DIR       where stage logs are written (default: none; output goes to the terminal)
set -euo pipefail

DATA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_DIR=$(cd "$DATA_DIR/../../.." && pwd)
PYTHON=${PYTHON:-python}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} \
  MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# CPUs this process may use (GNU nproc honours OMP_NUM_THREADS, so it is unset for the count).
data_cpus() {
  env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc
}

data_processes() {
  local cpus
  cpus=$(data_cpus)
  echo "${PROCESSES:-$(( cpus > 64 ? 64 : cpus ))}"
}

# data_scratch NAME: sets DATA_SCRATCH to a fresh scratch directory, removed when the calling script exits. Call it
# directly (not inside $(...)), so the exit trap belongs to the calling script.
data_scratch() {
  if [[ -n "${SCRATCH:-}" ]]; then
    mkdir -p "$SCRATCH"
    DATA_SCRATCH=$(mktemp -d "$SCRATCH/$1.XXXXXX")
  else
    DATA_SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/$1.XXXXXX")
  fi
  # shellcheck disable=SC2064
  trap "rm -rf -- '$DATA_SCRATCH'" EXIT
}

# data_log STAGE: send the rest of the script's output to $LOG_DIR/<stage>_<time>.log as well (when LOG_DIR is set),
# and record the code commit next to it.
data_log() {
  if [[ -n "${LOG_DIR:-}" ]]; then
    mkdir -p "$LOG_DIR"
    local stem
    stem="$LOG_DIR/$1_$(date +%Y%m%d_%H%M%S)"
    git -C "$REPO_DIR" rev-parse HEAD > "$stem.commit" 2>/dev/null || true
    exec > >(tee -a "$stem.log") 2>&1
  fi
  echo "$1 started at $(date -Is) on $(hostname)"
}

# require VAR...: fail unless every named variable is set.
require() {
  local name
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || { echo "Set $name (see the README)" >&2; exit 1; }
  done
}
