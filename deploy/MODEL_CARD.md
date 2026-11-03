---
license: apache-2.0
language:
  - es
  - en
base_model: openai/whisper-small
datasets:
  - fixie-ai/covost2
pipeline_tag: automatic-speech-recognition
tags:
  - speech-translation
  - simultaneous-translation
  - whisper
metrics:
  - bleu
library_name: transformers
---

# whisper-small, fine-tuned for Spanish to English speech translation

`openai/whisper-small` fine-tuned for **one epoch** on CoVoST 2 es→en (113 h), for use
with the simultaneous translation policy in
[PabloAndresE/simulst-es-en](https://github.com/PabloAndresE/simulst-es-en).

## Results

Held-out CoVoST 2 **test** split, all 13,221 utterances, read once after the model was
frozen:

| Model | Test BLEU |
|---|---|
| `openai/whisper-small` zero-shot | 31.23 |
| **This checkpoint** | **35.45** |

Whisper already performs X→English translation, so the zero-shot model is a real baseline
rather than an arbitrary initialisation. This is **not** a benchmark claim against the
literature: Whisper is pretrained on far more audio than systems trained from scratch,
which makes that a different comparison.

### The gain is much larger under streaming

Evaluated on audio **prefixes** rather than complete utterances — which is what a
simultaneous system sees — the same weights are worth far more:

| Setting | Base | This checkpoint | Gain |
|---|---|---|---|
| Offline | 37.61 | 38.31 | **+0.70** |
| Prefix decoding, `f=8` | 13.21 | 22.68 | **+9.47** |
| Prefix decoding, `f=16` | 19.04 | 27.96 | +8.92 |
| Prefix decoding, `f=32` | 24.12 | 33.49 | +9.37 |

Fine-tuning on complete utterances made the model robust to *incomplete* ones. An offline
benchmark values this checkpoint at +0.70 BLEU, which reads as noise; selecting a
checkpoint that way, for a system that will stream, discards most of the available
quality.

It also sharpened the no-speech calibration: this checkpoint scores real speech at 0.0000
on Whisper's `<|nocaptions|>` token where the base model scores 0.09 to 0.45, which makes
silence and background noise much easier to gate out.

## Usage

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "PabloAndresE/simulst-es-en", language="spanish", task="translate"
)
model = WhisperForConditionalGeneration.from_pretrained("PabloAndresE/simulst-es-en")
```

For the streaming policy, the live demo and the full evaluation code, see the
[repository](https://github.com/PabloAndresE/simulst-es-en).

**Load with `attn_implementation="eager"` if you intend to read attention weights.** The
default SDPA kernel returns an *empty* `cross_attentions` tuple under
`output_attentions=True`, with a warning rather than an error.

**Do not clear `suppress_tokens`.** Many fine-tuning recipes do, and it is harmless when
translating complete utterances. Under a policy that commits output permanently it is not:
the model emits `[Music]` on near-silence and it can never be retracted.

## Training

| | |
|---|---|
| Data | CoVoST 2 es→en, 79,013 utterances, 113.1 h |
| Epochs | 1 |
| Learning rate | 1e-5, OneCycle, 500 warmup steps |
| Batch size | 16 |
| Precision | AMP |
| Hardware | one H200, 12.8 minutes |

One epoch was chosen because three scored *lower* — 35.82 then 35.10 then 34.76 on
validation, while train loss fell from 1.01 to 0.37. That comparison is not clean, and
saying so matters: `OneCycleLR` spreads its decay over the configured total, so the
three-epoch run was still at a high learning rate at the one-epoch mark. The honest
statement is that one epoch *with a completed schedule* beat three.

## Limitations

- **Read speech only.** CoVoST 2 is Common Voice: prepared sentences read aloud, largely
  by European speakers. Spontaneous speech, disfluencies and Latin American accents are
  untested.
- **Domain vocabulary fails in the usual way.** In a clinical context *ensayos* was
  rendered "rehearsals" rather than "trials".
- **Single seed.** No variance estimate; nothing here is a significance claim.
- **30-second window.** Inherited from Whisper. Longer audio must be segmented.
- Two of the 79,015 training rows carry zero audio bytes and were skipped.

## Citation

The policy implemented alongside this checkpoint is AlignAtt:

> Papi, Turchi, Negri. *AlignAtt: Using Attention-based Audio-Translation Alignments as a
> Guide for Simultaneous Speech Translation.* INTERSPEECH 2023.
