# Agent brief: phase 1 on the GPU server

Self-contained. Do not assume knowledge of any prior conversation.

## Goal

Fine-tune `openai/whisper-small` for Spanish→English **speech** translation on CoVoST 2,
and report it against the zero-shot baseline. This is phase 1 of a simultaneous speech
translation project; the streaming policy comes later and is not your job.

## What is already known and measured

- The dataset is `fixie-ai/covost2`, config `es_en`: CoVoST 2 with audio bundled.
  79,015 train / 13,221 validation / 13,221 test. 4.53 GB.
- **The audio is 48 kHz**, not the 16 kHz the dataset card claims. Whisper needs 16 kHz.
  `simulst.audio.resample` does it; `preflight.py` prints the native rate. If you see
  16000 reported as native, something changed upstream — say so, do not ignore it.
- **HTTP streaming of this dataset stalls** on a slow link. Use
  `scripts/download_data.py`, which snapshots the parquet shards.
- Whisper already does X→English translation with `task="translate"`, so the zero-shot
  model is a real baseline. Measured on a laptop over 32 validation clips:
  `whisper-small` **BLEU 23.99**, `whisper-tiny` **10.69**. Those are sanity checks on a
  tiny sample, not results — re-measure properly.
- Clips average 6.26 s (p90 8.74 s). This matters for phase 2, not phase 1.

## Hard rules

- **Record the zero-shot baseline before training.** `scripts/slurm/train.sbatch` already
  does this. A fine-tune that fails to beat zero-shot is a result to report, not a bug
  to bury.
- **Never read the test split.** `evaluate.py --split test` refuses without
  `--confirm-final`. Phase 1 selects on validation. There is no reason to touch test yet.
- **Report only what you measured.** If a run diverged or OOM'd, say so.
- **Do not change the protocol between runs you intend to compare.** Same epochs, same
  batch size, same data cap, or the comparison is void.
- Whisper must have the translate task forced. If `forced_decoder_ids` is empty the model
  transcribes Spanish into Spanish and scores a plausible-looking but meaningless BLEU
  against English references. `preflight.py` checks this.

## Procedure

```bash
# login node: needs internet for wheels, weights and data
CUDA_TAG=cu124 bash scripts/setup_server.sh
export HF_HOME=$PWD/.hf_cache

# gate: verifies data, sample rate, offline weight loading, and times real steps
.venv/bin/python scripts/preflight.py --config configs/server.yaml

# minutes: proves the whole pipeline on whisper-tiny before spending GPU hours
make smoke

# the number to beat, written to runs/eval/
make eval-zero

# the real run
sbatch scripts/slurm/train.sbatch
```

If `preflight.py` reports OOM, lower `data.batch_size` in `configs/server.yaml`
(16 → 8 → 4) and re-run until it passes. The epoch estimate it prints excludes evaluation
and assumes data is already decoded in memory, so budget `--time` with headroom.

## What to report back

1. Preflight output: ms/step, projected epoch time, peak memory, batch size used.
2. Zero-shot BLEU on the full validation set (not 32 clips).
3. Fine-tuned BLEU per epoch, and whether it beat zero-shot.
4. Wall-clock per epoch and total.
5. Any `#SBATCH` changes the cluster required, so the scripts can be fixed in the repo.
6. Anything in this brief that turned out to be wrong.

## Optional, if time allows

`whisper-small` with `freeze_encoder: true` trains far fewer parameters. Whether that
costs BLEU is a cheap, useful ablation — and it is the kind of comparison that makes the
difference between a training log and a result.
