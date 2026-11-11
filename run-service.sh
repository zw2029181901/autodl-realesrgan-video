#!/usr/bin/env bash
set -Eeuo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$app_dir"
export MODEL_ROOT="$app_dir/models" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec /bin/bash "$app_dir/start.sh"
