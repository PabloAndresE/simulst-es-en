"""CoVoST 2 es->en loading, feature extraction and batching.

The dataset is downloaded and cached rather than streamed: its shards are 82 MB
each, and HTTP streaming stalls badly on a slow link. ``scripts/download_data.py``
materialises it once.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from simulst.audio import load, resample

COLUMNS = ("audio", "sentence", "translation")


@dataclass
class Example:
    """One utterance: Spanish audio, its transcript and the English reference."""

    audio: np.ndarray
    sentence: str
    translation: str
    duration_s: float
    #: Optional label, used by manifests to report results per recording set.
    group: str | None = None


def shard_paths(root: str | Path, split: str) -> list[Path]:
    """Parquet shards for a split, in deterministic order."""
    root = Path(root)
    paths = sorted(root.rglob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no {split} shards under {root}. Run scripts/download_data.py first."
        )
    return paths


def iter_examples(
    root: str | Path,
    split: str,
    target_rate: int = 16000,
    limit: int | None = None,
    on_corrupt: str = "skip",
) -> Iterator[Example]:
    """Stream decoded examples from the cached parquet shards.

    Two of the 79,015 training rows carry zero audio bytes, both in
    ``train-00005-of-00016``. That is an upstream defect, and crashing a
    multi-hour job on it is the wrong trade: ``on_corrupt="skip"`` drops them and
    reports the count at the end, so the loss is visible rather than silent.
    Pass ``on_corrupt="raise"`` to fail loudly instead.
    """
    if on_corrupt not in ("skip", "raise"):
        raise ValueError(f"on_corrupt must be 'skip' or 'raise', got {on_corrupt!r}")

    emitted = corrupt = 0
    for path in shard_paths(root, split):
        table = pq.read_table(path, columns=list(COLUMNS))
        for index, row in enumerate(table.to_pylist()):
            blob = row["audio"]
            raw = blob["bytes"] if isinstance(blob, dict) else blob
            try:
                if not raw:
                    raise ValueError("empty audio payload")
                audio = load(raw, target_rate)
            except Exception as exc:  # noqa: BLE001 - any decode failure is the same decision
                if on_corrupt == "raise":
                    raise RuntimeError(f"{path.name} row {index}: {exc}") from exc
                corrupt += 1
                continue

            yield Example(
                audio=audio,
                sentence=row["sentence"],
                translation=row["translation"],
                duration_s=len(audio) / target_rate,
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                break
        if limit is not None and emitted >= limit:
            break

    if corrupt:
        print(f"[data] skipped {corrupt} undecodable rows in split {split!r}")


class CoVoST2Dataset(torch.utils.data.Dataset):
    """Materialised split, held as decoded waveforms.

    CoVoST 2 clips average 6.3 s (p90 8.7 s), so the whole Spanish split fits in
    memory as float32 at 16 kHz: roughly 100 kB per second of audio.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        target_rate: int = 16000,
        limit: int | None = None,
        max_duration_s: float = 30.0,
        on_corrupt: str = "skip",
        concat: int = 1,
    ):
        examples = [
            ex
            for ex in iter_examples(root, split, target_rate, limit, on_corrupt)
            if ex.duration_s <= max_duration_s
        ]
        if concat > 1:
            examples = concatenate(examples, concat, sample_rate=target_rate)
            # Whisper's window is 30 s; anything past it is silently truncated.
            examples = [e for e in examples if e.duration_s <= 30.0]
        self.examples = examples
        if not self.examples:
            raise RuntimeError(f"no usable examples in split {split!r}")
        self.target_rate = target_rate

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]

    @property
    def total_hours(self) -> float:
        return sum(e.duration_s for e in self.examples) / 3600


class Collator:
    """Turn examples into Whisper input features and shifted label ids."""

    def __init__(self, processor, max_label_tokens: int = 160):
        self.processor = processor
        self.max_label_tokens = max_label_tokens

    def __call__(self, batch: list[Example]) -> dict[str, torch.Tensor]:
        features = self.processor.feature_extractor(
            [e.audio for e in batch],
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer(
            [e.translation for e in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_label_tokens,
        )
        label_ids = labels.input_ids.masked_fill(labels.attention_mask == 0, -100)
        return {
            "input_features": features.input_features,
            "labels": label_ids,
            "references": [e.translation for e in batch],
        }


def concatenate(
    examples: Sequence[Example], group: int, gap_s: float = 0.3, sample_rate: int = 16000
) -> list[Example]:
    """Join consecutive clips into longer synthetic utterances.

    Simultaneous translation only differs from offline translation when the
    utterance outlasts the policy's lookahead. CoVoST 2 clips average 5.15 s, so
    at a one-second chunk there are five or six decisions per clip and most of
    the output comes from the final flush, where every policy alike has the
    whole utterance. Comparisons made there measure almost nothing.

    Joining ``group`` clips with a short silence produces utterances long enough
    for the policies to diverge, with references joined in the same order.

    This is **synthetic** long-form and the README says so: real long-form
    speech has discourse flow, anaphora and prosodic continuity across sentence
    boundaries, and concatenated Common Voice readings have none of that. It
    buys length, not naturalness.
    """
    if group < 2:
        return list(examples)

    silence = np.zeros(int(gap_s * sample_rate), dtype=np.float32)
    joined = []
    for start in range(0, len(examples) - group + 1, group):
        batch = examples[start : start + group]
        pieces: list[np.ndarray] = []
        for index, example in enumerate(batch):
            if index:
                pieces.append(silence)
            pieces.append(example.audio)
        audio = np.concatenate(pieces)
        joined.append(
            Example(
                audio=audio,
                sentence=" ".join(e.sentence for e in batch),
                translation=" ".join(e.translation for e in batch),
                duration_s=len(audio) / sample_rate,
            )
        )
    return joined


def load_manifest(path: str | Path, target_rate: int = 16000) -> list[Example]:
    """Read a JSONL manifest of own recordings.

    One object per line, audio paths relative to the manifest's directory:

        {"audio": "setB/012.wav", "set": "B",
         "sentence": "o sea, lo que pasa es que...",
         "translation": "I mean, what happens is that..."}

    ``set`` is optional and carried through so results can be reported per set;
    see ``docs/RECORDING_PROTOCOL.md`` for why the sets are separated.
    """
    import json

    manifest = Path(path)
    root = manifest.parent
    examples = []

    for number, line in enumerate(manifest.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{manifest}:{number}: {exc}") from exc

        missing = {"audio", "translation"} - row.keys()
        if missing:
            raise ValueError(f"{manifest}:{number}: missing {sorted(missing)}")

        audio_path = root / row["audio"]
        if not audio_path.exists():
            raise FileNotFoundError(f"{manifest}:{number}: no such audio {audio_path}")

        raw, rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        audio = resample(raw.astype(np.float32), rate, target_rate)

        examples.append(
            Example(
                audio=audio,
                sentence=row.get("sentence", ""),
                translation=row["translation"],
                duration_s=len(audio) / target_rate,
                group=row.get("set"),
            )
        )

    if not examples:
        raise RuntimeError(f"{manifest} contains no usable entries")
    return examples
