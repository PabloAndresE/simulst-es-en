# Simultaneous Speech Translation, Spanish → English

Building a live speech translator from a base model: fine-tune an offline speech
translation model, then **implement the simultaneous policy from the paper** so it emits
English while the Spanish is still being spoken.

**Status: phase 1 (offline baseline). Nothing below is a final result yet.**

## The question

A simultaneous system is not a different model, it is a different *decision*: given half
an utterance, do you emit now and risk being wrong, or wait and accumulate lag? That
decision — the emission policy — is where the research sits, and it is inference-time
logic you write yourself.

This repository builds that in three phases:

| Phase | What it produces | What it demonstrates |
|---|---|---|
| **1. Offline baseline** | A fine-tuned ES→EN speech translation model | Training speech models, not just calling them |
| **2. Simultaneous policy** | AlignAtt-style decoding implemented from the paper | Reading research and implementing it correctly |
| **3. Monotonic attention** | A model trained to be natively simultaneous | Understanding the mechanism, not the recipe |

Phases 1 and 2 are the target; phase 3 is a stretch.

## Why Whisper is the starting point, and the baseline

Whisper ships an X→English translation task, so `openai/whisper-small` with
`task="translate"` **already performs this job**. That makes the zero-shot model a real
baseline to beat rather than an arbitrary initialisation.

Measured locally on 32 validation clips, greedy decoding:

| Model | BLEU |
|---|---|
| `whisper-small`, zero-shot | **23.99** |
| `whisper-tiny`, zero-shot | 10.69 |

Small sample, so treat these as sanity checks rather than results. Any fine-tune is
reported against the zero-shot number on the same split, and `scripts/slurm/train.sbatch`
records the baseline *before* training starts so the comparison cannot be reconstructed
after the fact.

Sample output (zero-shot `whisper-small`):

```
hyp: They inhabit shallow waters and rocky.
ref: It lives in shallow, rocky water.

hyp: It mainly operates cargo and cargo regional flights.
ref: It mainly operates regional cabotage and cargo flights.
```

Reasonable, imperfect, clear headroom. That is the right place to start.

## Data

[`fixie-ai/covost2`](https://huggingface.co/datasets/fixie-ai/covost2), config `es_en` —
CoVoST 2 with the audio bundled. The canonical `facebook/covost2` ships only a loading
script and expects a manual Common Voice download; this one does not.

| Split | Examples |
|---|---|
| train | 79,015 (~138 h) |
| validation | 13,221 |
| test | 13,221 |

**Two measured facts that the dataset card gets wrong or omits:**

- **The audio is 48 kHz, not the 16 kHz the card advertises.** Whisper assumes 16 kHz;
  feeding it 48 kHz produces mel features for audio three times too fast, trains without
  error and never converges. `simulst.audio.resample` handles it and `preflight.py`
  reports the native rate so the mismatch is visible.
- **HTTP streaming stalls.** Shards are 82 MB; on a 2.1 MB/s link `load_dataset(streaming=True)`
  hangs for over ten minutes without yielding a row. `scripts/download_data.py`
  materialises the 4.53 GB once (~36 minutes at that speed).

### A constraint that shapes phase 2

Measured over 300 clips: **mean 6.26 s, p90 8.74 s, max 12.4 s.**

Simultaneous translation only has room to differ from offline translation on *long*
utterances — in six seconds there is little space between "wait for the end" and "emit as
you go". CoVoST 2 is therefore the right corpus to **train** on and the wrong one to
**evaluate latency** on. Phase 2 will use longer-form audio (Europarl-ST, or concatenated
CoVoST utterances) for the quality-latency curves.

## Metrics

- **BLEU** via sacreBLEU, so numbers are comparable with the speech translation literature.
- **Average Lagging** (Ma et al., 2019): how far behind a constant-rate oracle the policy
  emits, in seconds of source audio. Returns `None` rather than `0.0` for an empty
  hypothesis, where lagging is undefined.
- **Revision rate**: the share of partial outputs that rewrote earlier text instead of
  extending it. This is what a viewer perceives as flicker, and it is the thing a
  re-translation baseline does badly.

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/download_data.py --out data     # 4.53 GB
python scripts/preflight.py --config configs/server.yaml
make smoke        # whisper-tiny, 64 examples, minutes
make eval-zero    # the zero-shot baseline
make train
```

On a GPU server:

```bash
CUDA_TAG=cu124 bash scripts/setup_server.sh
GPU=1 bash scripts/run_training.sh        # no scheduler
sbatch scripts/slurm/train.sbatch         # only where SLURM exists
```

On a shared box, pick the GPU by **free memory, not index**: a card can sit at 0%
utilisation with its memory already held by someone else's idle process.
`run_training.sh` does that automatically when `GPU` is unset.

`scripts/preflight.py` is the gate: it verifies the cached shards, reports the native
sample rate, confirms the weights load **offline** (compute nodes are often firewalled),
checks that the translate task is actually forced, and times real training steps to
project epoch time and peak memory.

## The test split

`simulst.evaluate --split test` refuses without `--confirm-final`. Tune on validation;
read test once, when the model is frozen.

## Layout

```
src/simulst/
  config.py     dataclass config loaded from YAML
  audio.py      decoding and 48 kHz -> 16 kHz resampling
  data.py       parquet shards, dataset, Whisper collator
  model.py      Whisper configured for es->en translation
  metrics.py    BLEU, Average Lagging, revision rate
  train.py      fine-tuning CLI
  evaluate.py   scoring CLI, gated test split
scripts/        data download, preflight, server setup, SLURM
configs/        server and smoke configurations
tests/          20 tests: resampling, data, metrics
```

## Limitations

- Phase 1 only. No simultaneous policy is implemented yet; the latency metrics exist and
  are tested, but nothing produces emissions to feed them.
- The BLEU numbers above are from 24-32 clips on a laptop, not an evaluation.
- Resampling is linear interpolation, not a windowed-sinc filter. Adequate for 48→16 kHz
  of speech; a polyphase resampler would be more faithful.
- CoVoST 2 is read speech from Common Voice. Spontaneous speech and Ecuadorian Spanish
  are both out of distribution and untested.

## Reference

- Ma et al., *STACL: Simultaneous Translation with Implicit Anticipation and Controllable
  Latency* (2019) — Average Lagging
- Papi et al., *AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide
  for Simultaneous Speech Translation* (2023)
- Wang et al., *CoVoST 2* (2020)
- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (2022)

MIT licensed.
