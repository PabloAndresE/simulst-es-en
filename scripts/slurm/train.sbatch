#!/bin/bash
#SBATCH --job-name=simulst
#SBATCH --output=logs/simulst_%j.out
#SBATCH --error=logs/simulst_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# NOTE: only for clusters that actually run SLURM. The USFQ DGX does not have
# sbatch; use scripts/run_training.sh there.
#
# Partition and account are cluster-specific; add them here or on the command
# line: sbatch -p gpu -A myaccount scripts/slurm/train.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
CONFIG="${CONFIG:-configs/server.yaml}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Gate, then the zero-shot baseline, then training. The baseline is recorded
# before any weights move so the comparison cannot be reconstructed after the
# fact.
.venv/bin/python scripts/preflight.py --config "$CONFIG"
.venv/bin/python -m simulst.evaluate --config "$CONFIG" --out-dir runs/eval
.venv/bin/python -m simulst.train --config "$CONFIG"
