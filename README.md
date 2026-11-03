# Simultaneous Speech Translation, Spanish → English

English appears while the Spanish is still being spoken, and nothing on screen is ever
rewritten.

![The policy translating a 10.4 s utterance](docs/assets/demo.gif)

*The GIF is a frame-accurate rendering of the terminal panel the CLI draws, from the same
policy run — not a mock-up of an interface the repository does not ship.*

```bash
pip install -e .
python -m simulst.live --mic --checkpoint PabloAndresE/simulst-es-en
python -m simulst.live --file clip.wav --compare --checkpoint PabloAndresE/simulst-es-en
```

Weights: [`PabloAndresE/simulst-es-en`](https://huggingface.co/PabloAndresE/simulst-es-en).
Container: `docker build -f deploy/Dockerfile -t simulst .`

## What the hard part actually is

A simultaneous system is not a different model, it is a different *decision*: given half
an utterance, emit now and risk being wrong, or wait and accumulate lag. That decision is
inference-time logic, and this repository implements it rather than calling a library that
hides it.

The policy is **AlignAtt** (Papi et al., 2023). Generate greedily on the audio received so
far; for each candidate token, read which encoder frame its cross-attention points at. A
token leaning on the last `f` frames depends on sound that is still arriving, so it is
withheld. Everything else is committed permanently — which is why the measured revision
rate is **0.000 at every operating point**, and why the cost shows up as latency instead
of as text flickering in front of the reader.

## Does the policy earn its complexity?

A curve against the offline ceiling says what a policy costs. It does not say whether
writing it was worth the trouble. Two baselines answer that: **wait-k**, which emits on a
fixed schedule and reads nothing from the model, and **re-translation**, which redecodes
the whole prefix every chunk and is free to rewrite the screen.

133 synthetic long-form utterances (mean 23.7 s), three disjoint slices of the split,
44 utterances each, one-second chunks, fine-tuned `whisper-small`:

| Policy | BLEU | Average Lagging | Revisions |
|---|---|---|---|
| Offline (wait for everything) | 35.44 ±1.80 | 23.68 s | 0.000 |
| wait-k `k=4` | 35.07 ±1.32 | 4.60 s ±0.32 | 0.009 |
| wait-k `k=8` | 35.04 ±1.24 | 8.10 s ±0.19 | 0.006 |
| Re-translation | 35.44 ±1.80 | **0.45 s** ±0.06 | **0.502** |
| AlignAtt `f=32` | 32.77 ±2.07 | 1.63 s ±0.43 | 0.014 |
| **AlignAtt `f=64`** | **34.30 ±1.75** | **2.12 s** ±0.32 | 0.006 |

**In BLEU, AlignAtt and wait-k cannot be told apart** — 34.30 ±1.75 against 35.07 ±1.32,
intervals overlapping. **In lag they can**: 2.12 ±0.32 against 4.60 ±0.32, intervals
disjoint. The defensible claim is therefore narrow and worth stating exactly: *at a
quality difference this sample cannot resolve, AlignAtt runs at under half the delay.*

Re-translation is the useful foil. It matches offline quality at 0.45 s, and pays for it
by rewriting the screen in **half** of all updates. No policy dominates: they occupy
different corners of quality, latency and stability, and which corner matters is a
property of the application, not of the method.

### The comparison is meaningless on short clips, and that is a finding

Run on raw CoVoST (mean 5.15 s) the same table said the opposite — wait-k `k=2` at 16.30
BLEU and 1.96 s beating AlignAtt `f=64` at 15.55 BLEU and 2.24 s, better on both axes.

That reading was an artefact. At a one-second chunk a five-second clip allows five
decisions, and most of the output comes from the final flush, where every policy alike
holds the complete utterance. The benchmark was measuring the flush, not the policy.

Long-form here is **synthetic**: consecutive clips concatenated with a short gap and
references joined in the same order. It buys length, not naturalness — real speech has
discourse flow and prosodic continuity across sentence boundaries that concatenated
Common Voice readings do not.

## The quality-latency curve

200 held-out utterances, 0.5 s chunks, one H200, base against fine-tuned:

| Policy | Base BLEU | Fine-tuned BLEU | Lag | Token revisions |
|---|---|---|---|---|
| Offline | 37.61 | 38.31 | 6.26 s | 0.000 |
| AlignAtt `f=8` | 13.21 | 22.68 | **0.61 s** | 0.000 |
| AlignAtt `f=16` | 19.04 | 27.96 | 0.83 s | 0.000 |
| AlignAtt `f=32` | 24.12 | 33.49 | 1.15 s | 0.000 |
| AlignAtt `f=64` | 32.90 | 36.36 | 1.76 s | 0.000 |
| **AlignAtt `f=100`** | 35.00 | **38.01** | **2.55 s** | 0.000 |

### Fine-tuning is worth thirteen times more under streaming than offline

The two columns are the same two checkpoints, and the gap between them is not constant:

| | Offline | `f=8` | `f=16` | `f=32` |
|---|---|---|---|---|
| Gain from fine-tuning | **+0.70** | **+9.47** | +8.92 | +9.37 |

An offline benchmark values this fine-tune at +0.70 BLEU, which reads as noise. The same
weights are worth +9.47 BLEU when the model must translate an audio prefix. Training on
complete utterances made the model robust to *incomplete* ones, and the standard
evaluation cannot see that at all.

**The offline number was measuring the wrong thing.** Selecting a checkpoint by offline
BLEU, for a system that will stream, discards most of the quality already paid for.

### Revisions: measure the tokens, not the string

The policy commits permanently, so no emitted token can ever be retracted and the
**token-level revision rate is 0.000 at every operating point** — verified, not assumed.

The text-level rate is not zero, around 0.011, and the cause is worth knowing. Whisper's
byte-level BPE can emit a token carrying part of a multi-byte character; decoded alone it
is U+FFFD, and it rewrites itself once the remaining bytes arrive:

```
before : '�'
after  : '“Suo'
token ids before are a prefix of after: True
```

Every instance was a curly quote or an accented character. The guarantee held; the metric
was measuring the display. Emissions are now the delta of the decoded prefix, so no
replacement glyph reaches the screen, and both rates are reported.

## Silence and background noise become words

Whisper invents text from nothing. Six seconds of digital silence produced
*"I will be there."*; a 60 Hz hum produced *"The End"*. Under a commit-forever policy
those are permanent, and in a live session they arrive between every sentence.

The model already scores this: it has a no-speech token, and one small decoder pass reads
it. Measured on the fine-tuned checkpoint:

| Input | p(no speech) |
|---|---|
| Digital silence | 0.89 |
| White noise | 0.57 |
| 60 Hz hum | 0.82 |
| **Speech** | **0.0000** |

A threshold of 0.5 separates them cleanly and silence, hum and noise all go to empty
output while speech — including speech preceded by three seconds of noise — is untouched.
Fine-tuning *sharpened* this calibration: the base checkpoint scores real speech at 0.09
to 0.45, the fine-tuned one at 0.0000.

Two traps on the way there, both of which fail quietly:

- The token is named `<|nocaptions|>` in the released checkpoints, not `<|nospeech|>`.
  Asking a tokenizer for a token it does not have returns the *unknown* id rather than
  raising, so the probability came back as a constant 0.0000 for every input, silence
  included. `no_speech_token_id` raises instead.
- The gate first ran only while audio was arriving, exempting the final flush. Every
  hallucination escaped through it unchanged. Whether more audio is coming has no bearing
  on whether what already arrived was speech.

## Two defects that produce plausible, wrong output

Both were mine, both were found only by running the thing on real audio.

**PyTorch's fused attention kernel silently returns no attention.** A model loaded with the
default `sdpa` implementation and asked for `output_attentions=True` yields an *empty*
`cross_attentions` tuple, and the warning goes to stderr where a training log buries it.
An attention-based policy gets nothing to read. `build()` takes `attn_implementation` and
`attended_frame` raises a named error instead of failing twenty lines later.

**Clearing `suppress_tokens` lets Whisper hallucinate `[Music]` into the first half
second.** Fine-tuning recipes clear that list routinely and it is harmless offline. Under a
commit-forever policy it is fatal: the marker is emitted at 0.5 s and can never be taken
back, poisoning the rest of the utterance. The pretrained suppression list is kept, and
since decoding by hand bypasses `generate()`, the policy applies the list itself — a
hand-written decode loop runs none of the logits processors you may believe are protecting
you.

## The model underneath

### Held-out test, read once

The test split is gated behind `--confirm-final` and was read a single time, after the
model was frozen. All 13,221 utterances:

| Model | Test BLEU |
|---|---|
| `whisper-small` zero-shot | 31.23 |
| **Fine-tuned, one epoch** | **35.45** |

**+4.22 BLEU.** Validation had reported +2.21 on the same comparison, so tuning decisions
made on validation understated what the fine-tune was worth rather than overstating it —
the less common direction, and the reason the split stays sealed until the end.

For comparison, published CoVoST 2 es→en systems of the era this dataset was released
report in the low-to-mid thirties. This number is **not** a benchmark claim: Whisper is
pretrained on vastly more audio than systems trained from scratch, which makes the
comparison a different race.

### Fine-tuning `whisper-small`

Validation BLEU on 1,000 utterances, one H200:

| Configuration | Trainable params | BLEU | GPU time |
|---|---|---|---|
| Zero-shot | — | 33.61 | — |
| Full fine-tune, 3 epochs | 241.7M (100%) | 35.82 | 33 min |
| **Full fine-tune, 1 epoch** | 241.7M (100%) | **36.13** | 12.8 min |
| Frozen encoder, 1 epoch | 153.6M (63.5%) | 35.30 | 7.9 min |

**One epoch beats three, at a third of the cost.** Not because later epochs are harmful in
themselves: `OneCycleLR` spreads its decay over the configured total, so the three-epoch
run was still at a high learning rate when it passed the one-epoch mark. The honest
statement is *one epoch with a completed schedule beats three*, and the longer run then
overfits — train loss 1.01 → 0.37 while BLEU falls 35.82 → 34.76.

**Fine-tuning dips below the model it started from.** Evaluating four times inside the
first epoch shows what a per-epoch schedule hides:

| Progress | Full fine-tune | Frozen encoder |
|---|---|---|
| 25% | **32.59** (below zero-shot) | 34.15 |
| 50% | 33.63 | 34.17 |
| 75% | 35.85 | 35.28 |
| 100% | **36.13** | 35.30 |

A quarter of the way through, full fine-tuning is *worse than not training at all*. With
the encoder frozen it never happens: that arm is above zero-shot at every measured point,
for 0.83 BLEU and 40% less wall clock. **The dip is the encoder being pulled off its
pretrained solution**, not fine-tuning in general — a trade-off invisible from a final
number.

A caveat on these tables: zero-shot validation *loss* is not comparable with the
fine-tuned figures, because the base model writes different punctuation and casing and
teacher forcing penalises that regardless of translation quality. BLEU is the only column
that compares across the two.

## Running it on CPU

`deploy/Dockerfile` bakes the weights in and sets `HF_HUB_OFFLINE=1`, so a started
container does no network I/O. Measured with `scripts/bench_cpu.py` on an M1 Pro CPU,
`whisper-small`, per chunk against the chunk budget:

| Chunk | Median | Max | Budget | Verdict |
|---|---|---|---|---|
| 0.5 s | 1100 ms | 4560 ms | 500 ms | falls behind |
| 1.0 s | 1083 ms | 1254 ms | 1000 ms | falls behind |
| **2.0 s** | 1108 ms | 1252 ms | 2000 ms | **keeps up** |

Compute per chunk is roughly constant at ~1.1 s because the encoder always processes a
padded 30 s window regardless of how much real audio arrived. **On CPU the chunk must be
at least 2 s**, which puts a floor under latency: the sub-2-second figures elsewhere in
this README require a GPU. A deployment that quotes them on CPU hardware is quoting
numbers it cannot reach.

Inside the container it is worse again — 2043 ms per chunk against 1083 ms for the same
CPU natively, because Docker on macOS virtualises the CPU and loses Apple's accelerated
paths. There the chunk must be **at least 3 s**. Both measurements are in `results/`, and
`deploy/README.md` has the breakdown.

## Data

[`fixie-ai/covost2`](https://huggingface.co/datasets/fixie-ai/covost2), config `es_en` —
CoVoST 2 with the audio bundled. The canonical `facebook/covost2` ships only a loading
script and expects a manual Common Voice download; this one does not.

| Split | Examples |
|---|---|
| train | 79,015 (113.1 h of audio) |
| validation | 13,221 |
| test | 13,221 |

**Two measured facts that the dataset card gets wrong or omits:**

- **The audio is 48 kHz, not the 16 kHz the card advertises.** Whisper assumes 16 kHz;
  feeding it 48 kHz produces mel features for audio three times too fast, trains without
  error and never converges. `simulst.audio.resample` handles it and `preflight.py`
  reports the native rate so the mismatch is visible.
- **HTTP streaming stalls.** Shards are 82 MB; on a 2.1 MB/s link `load_dataset(streaming=True)`
  hangs for over ten minutes without yielding a row. `scripts/download_data.py`
  materialises the 4.53 GB once (~36 minutes at that speed, 133 s on a datacentre link).
- **Two of the 79,015 training rows carry zero audio bytes**, both in
  `train-00005-of-00016`. The loader skips undecodable rows and prints the count rather
  than killing a multi-hour job; `on_corrupt="raise"` fails loudly instead. All 79,013
  remaining rows are valid MP3.

### A constraint that shapes phase 2

Measured over the full training split: **113.1 h across 79,013 usable clips, mean 5.15 s**
(a 300-clip probe had suggested 6.26 s — a reminder to measure the corpus, not a sample).

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

# the demo, no training required
python -m simulst.live --file docs/assets/sample_es.wav --compare

# reproduce the curve
python -m simulst.stream_eval --thresholds 8 16 32 64 100 --limit 200

# every policy over the same long-form audio, three slices
python -m simulst.compare --concat 4 --seeds 0 1 2 --limit 200

# what it costs on CPU
python scripts/bench_cpu.py --chunk-s 0.5 1.0 2.0

# the model underneath
python scripts/preflight.py --config configs/server.yaml
make smoke && make eval-zero && make train
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
  simultaneous.py  the AlignAtt policy and the streaming translator
  policies.py      wait-k and re-translation, the baselines it is measured against
  compare.py       all policies over identical audio, with a spread per policy
  display.py       the in-place terminal panel the demo and the GIF share
  stream_eval.py   the quality-latency sweep
  live.py          microphone and file demos
  metrics.py       BLEU, Average Lagging, revision rate
  model.py         Whisper configured for es->en translation
  data.py          parquet shards, dataset, Whisper collator
  audio.py         decoding and 48 kHz -> 16 kHz resampling
  train.py         fine-tuning CLI
  evaluate.py      scoring CLI, gated test split
  config.py        dataclass config loaded from YAML
scripts/           download, preflight, server setup, demo GIF, CPU bench
deploy/            CPU container with the weights baked in
results/           raw metrics from every run in this README
tests/             95 tests, no weights required
```

## Limitations

- **The policy comparison is 44 utterances per slice, three slices.** Enough to separate
  the lag figures, not enough to separate the BLEU figures — which is why the text says
  only that.
- **Long-form audio is synthetic**, built by concatenating consecutive clips. Real
  long-form speech has discourse structure that this does not reproduce.
- **The streaming curve is 200 utterances on one slice.**
- **CoVoST 2 clips average 5.15 s.** Simultaneous translation has the least to prove on
  short utterances; the curve would look different on long-form speech, which is the
  honest next measurement (Europarl-ST, MuST-C).
- **`f` was swept at five points.** The optimum may lie past `f=100`, where the policy
  approaches offline behaviour and the latency advantage disappears.
- **One language pair, read speech.** CoVoST 2 is Common Voice: prepared sentences read
  aloud. Spontaneous speech, disfluencies and non-peninsular accents are untested, and a
  24-second synthetic utterance already produced one domain error — *ensayos* in a
  clinical context rendered as "rehearsals" rather than "trials". That is an anecdote;
  `docs/RECORDING_PROTOCOL.md` turns it into a measurement, and
  `simulst.evaluate --manifest` scores it per recording set.
- **Average Lagging assumes a constant-rate oracle**, which is a crude model of how a
  human interpreter actually paces output.

- No simultaneous policy is implemented yet; the latency metrics exist and are tested,
  but nothing produces emissions to feed them. That is phase 2.
- **Single seed.** The +2.52 BLEU gain has no variance estimate; nothing here is a
  significance claim.
- **The one-epoch and three-epoch runs do not share a learning-rate schedule**, since
  `OneCycleLR` is defined over total steps. They are not a clean epoch-count ablation.
- **The peak may lie past one epoch** with a schedule built for it; only 1 and 3 epochs
  were measured.
- Validation is 1,000 of 13,221 utterances, to keep per-epoch evaluation affordable.
- The test split has not been read.
- Resampling is linear interpolation, not a windowed-sinc filter. Adequate for 48→16 kHz
  of speech; a polyphase resampler would be more faithful.
- CoVoST 2 is read speech from Common Voice. Spontaneous speech and Ecuadorian Spanish
  are both out of distribution and untested.

## Reference

- Papi et al., *AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide
  for Simultaneous Speech Translation* (2023) — the policy implemented here
- Ma et al., *STACL: Simultaneous Translation with Implicit Anticipation and Controllable
  Latency* (2019) — Average Lagging, and the wait-k baseline
- Ma et al., *STACL: Simultaneous Translation with Implicit Anticipation and Controllable
  Latency* (2019) — Average Lagging
- Wang et al., *CoVoST 2* (2020)
- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (2022)

## Artefacts

| | |
|---|---|
| Fine-tuned weights | [`PabloAndresE/simulst-es-en`](https://huggingface.co/PabloAndresE/simulst-es-en) |
| Raw metrics for every table above | `results/` |
| CPU container | `deploy/Dockerfile`, builds from the published weights |
| Recording protocol for own audio | `docs/RECORDING_PROTOCOL.md` |

MIT licensed. The weights are Apache 2.0, inherited from Whisper.
