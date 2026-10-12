#!/usr/bin/env bash
# Build the 6-bin signed voxel npz files used by RTEIR training/testing from the
# raw GoPro event npz files.
#
# Expected layout:
#   <DATA_ROOT>/{train,test}/{blur,gt,event}
# and it produces:
#   <DATA_ROOT>/{train,test}/voxel/<sequence>/<frame>.npz   (HWC, 6 channels)
#
# Usage:
#   bash scripts/build_gopro_voxel.sh /path/to/GoPro
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${1:-${DATA_ROOT:-./datasets/GoPro}}"
NUM_BINS="${NUM_BINS:-6}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_DTYPE="${SAVE_DTYPE:-float32}"     # float32 or float16
COMPRESSED="${COMPRESSED:-false}"       # true or false
OVERWRITE="${OVERWRITE:-false}"         # true or false

build_one_split() {
  local split="$1"
  local split_root="${DATA_ROOT}/${split}"

  if [[ ! -d "${split_root}" ]]; then
    echo "[WARN] Split directory not found, skip: ${split_root}"
    return
  fi

  echo "[INFO] Building voxel for split=${split}, root=${split_root}"

  local cmd=(
    python scripts/gopro_event_to_voxel.py
    --dataroot "${split_root}"
    --event-dir-name event
    --output-root "${split_root}"
    --output-dir-name voxel
    --ref-dir-name blur
    --num-bins "${NUM_BINS}"
    --num-workers "${NUM_WORKERS}"
    --save-dtype "${SAVE_DTYPE}"
  )

  if [[ "${COMPRESSED}" == "true" ]]; then
    cmd+=(--compressed)
  fi
  if [[ "${OVERWRITE}" == "true" ]]; then
    cmd+=(--overwrite)
  fi

  "${cmd[@]}"
}

build_one_split "train"
build_one_split "test"

echo "[DONE] GoPro train/test voxel generation finished."
