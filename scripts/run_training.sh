#!/usr/bin/env bash
# Run the pipeline on a GPU box with no scheduler (the USFQ DGX has no SLURM).
#
#   GPU=1 bash scripts/run_training.sh
#
# The machine is shared. Pin a single GPU rather than letting torch grab the
# first visible one, and pick it by FREE MEMORY, not by index: several GPUs here
# sit at 0% utilisation with their memory already taken by idle processes.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONFIG="${CONFIG:-configs/server.yaml}"
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"

if [ -z "${GPU:-}" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  echo "==> auto-selected GPU $GPU (most free memory)"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

nvidia-smi --id="$GPU" --query-gpu=index,name,memory.free,utilization.gpu \
  --format=csv,noheader

echo "==> preflight"
.venv/bin/python scripts/preflight.py --config "$CONFIG"

echo "==> smoke run (whisper-tiny, minutes)"
.venv/bin/python -m simulst.train --config configs/smoke.yaml

echo "==> zero-shot baseline, recorded before any weights move"
.venv/bin/python -m simulst.evaluate --config "$CONFIG" --out-dir runs/eval

echo "==> training"
.venv/bin/python -m simulst.train --config "$CONFIG"

echo "==> done. results in runs/"
