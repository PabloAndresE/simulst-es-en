# Recording protocol: spontaneous Ecuadorian Spanish

Everything measured in this repository uses CoVoST 2, which is Common Voice:
prepared sentences, read aloud, mostly by European speakers, in clean
conditions. None of those properties hold for the thing a live translator is
pointed at.

One 24-second synthetic utterance already produced a domain error — *ensayos*
in a clinical context rendered as "rehearsals" rather than "trials". That is a
single anecdote. This protocol turns it into a measurement.

## What is being tested

Three variables the training distribution does not contain, separated so a drop
can be attributed rather than guessed at:

| Set | Holds constant | Varies |
|---|---|---|
| **A. Read, Ecuadorian** | prepared text | the accent |
| **B. Spontaneous, Ecuadorian** | the accent | preparation, disfluencies, pacing |
| **C. Domain vocabulary** | the accent | technical terms with known translations |

Set A separates accent from spontaneity. Without it, a drop in set B could be
either, and the conclusion would be unfalsifiable.

## Recording

- **16 kHz mono minimum**, lossless. QuickTime or `sox` is fine; phone voice
  memos are compressed and will confound the result.
- **Headset or lapel microphone**, consistent distance. Laptop microphones vary
  with head position and that variance enters the measurement.
- **One quiet room, one session.** Changing rooms mid-set adds a variable.
- **20 to 30 seconds per utterance.** Shorter than that and the policy barely
  makes a decision; longer than 30 s exceeds Whisper's window.
- Target **20 minutes total**, roughly 50 utterances.

### Set A — read, 15 utterances

Read sentences from a Spanish news site aloud, naturally, without performing.
Keep the source text: it becomes the transcript, and a human translation of it
becomes the reference.

### Set B — spontaneous, 25 utterances

Talk, unscripted, 20 to 30 seconds per take, on things you can speak about
without preparing: explain a technical decision, describe your week, argue a
position. **Do not re-record a take because it had a false start** — the false
starts are the data. Transcribe afterwards, verbatim, including the "o sea", the
"este", the repaired sentences.

### Set C — domain vocabulary, 10 utterances

Sentences containing terms whose correct English differs from the obvious
cognate, embedded in natural speech rather than listed. Candidates:

| Spanish | Correct in context | Likely error |
|---|---|---|
| ensayo clínico | clinical trial | rehearsal, essay |
| carpeta | folder | carpet |
| actualmente | currently | actually |
| sensible | sensitive | sensible |
| asistir a | to attend | to assist |
| fábrica | factory | fabric |
| compromiso | commitment | compromise |

## References

Each utterance needs a **Spanish transcript** and an **English translation**.
Translate for meaning, the way a competent interpreter would, and do not
translate toward what the model is likely to say — a reference written to be
easy inflates BLEU and measures nothing.

Write them as JSONL, one object per utterance:

```json
{"audio": "setB/012.wav", "set": "B", "sentence": "o sea, lo que pasa es que...",
 "translation": "I mean, what happens is that..."}
```

## Scoring

```bash
python -m simulst.evaluate --config configs/server.yaml \
    --checkpoint checkpoints/ft-best --manifest recordings/manifest.jsonl
```

Report BLEU per set against the CoVoST validation figure. **The interesting
outcome is a large drop**, and the honest framing is that a model trained on
read European Spanish was asked to handle spontaneous Ecuadorian speech, which
is a distribution it never saw — not that the model is bad.

Keep the failures. A table of specific mistranslations with their audio is worth
more to a reader than the aggregate, because it shows *what kind* of thing
breaks, which is what someone deciding whether to deploy actually needs.

## Licensing

These are recordings of a person. Decide before publishing whether the audio
ships with the repository or only the transcripts and scores do. If it ships,
say so in the README and pick a licence for it explicitly.
