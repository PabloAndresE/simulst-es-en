#!/usr/bin/env bash
# Environment setup for a CUDA node. Run on a LOGIN node: it needs internet for
# the wheels, the Whisper weights and the dataset, and compute nodes are often
# firewalled.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CUDA_TAG="${CUDA_TAG:-cu124}"
PY="${PYTHON:-python3}"

echo "==> venv with $($PY --version)"
"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip wheel

echo "==> torch for $CUDA_TAG"
.venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> project"
.venv/bin/pip install -e ".[dev]"

export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"
echo "==> caching Whisper into $HF_HOME"
.venv/bin/python - <<'PY'
from transformers import WhisperForConditionalGeneration, WhisperProcessor
for name in ("openai/whisper-tiny", "openai/whisper-small"):
    WhisperProcessor.from_pretrained(name, language="spanish", task="translate")
    WhisperForConditionalGeneration.from_pretrained(name)
    print("cached", name)
PY

echo "==> dataset (4.53 GB)"
if [ ! -d data/es_en ]; then
  .venv/bin/python scripts/download_data.py --out data
else
  echo "    present: $(find data -name '*.parquet' | wc -l | tr -d ' ') shards"
fi

echo
echo "==> verifying"
.venv/bin/ruff check src tests scripts
.venv/bin/pytest -q

cat <<EOF

Next:
  export HF_HOME=$HF_HOME
  .venv/bin/python scripts/preflight.py --config configs/server.yaml
  make smoke        # minutes, proves the pipeline
  make eval-zero    # the zero-shot baseline to beat
  make train
EOF
