"""CoVoST 2 es->en loading, feature extraction and batching.

The dataset is downloaded and cached rather than streamed: its shards are 82 MB
each, and HTTP streaming stalls badly on a slow link. ``scripts/download_data.py``
materialises it once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from simulst.audio import decode, load

COLUMNS = ("audio", "sentence", "translation")


@dataclass
class Example:
    """One utterance: Spanish audio, its transcript and the English reference."""

    audio: np.ndarray
    sentence: str
    translation: str
    duration_s: float


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
) -> Iterator[Example]:
    """Stream decoded examples from the cached parquet shards."""
    emitted = 0
    for path in shard_paths(root, split):
        table = pq.read_table(path, columns=list(COLUMNS))
        for row in table.to_pylist():
            blob = row["audio"]
            raw = blob["bytes"] if isinstance(blob, dict) else blob
            _, native_rate = decode(raw)
            audio = load(raw, target_rate)
            yield Example(
                audio=audio,
                sentence=row["sentence"],
                translation=row["translation"],
                duration_s=len(audio) / target_rate,
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return
            del native_rate


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
    ):
        self.examples = [
            ex
            for ex in iter_examples(root, split, target_rate, limit)
            if ex.duration_s <= max_duration_s
        ]
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
