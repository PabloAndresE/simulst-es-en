"""Typed experiment configuration, loadable from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    #: Preprocessed CoVoST 2 with audio bundled. The canonical facebook/covost2
    #: ships only a loading script and expects a manual Common Voice download.
    dataset: str = "fixie-ai/covost2"
    config_name: str = "es_en"
    cache_dir: str = "data"
    #: CoVoST 2 audio is stored at 48 kHz despite the dataset card claiming 16 kHz.
    #: Whisper expects 16 kHz; resampling is not optional.
    target_sample_rate: int = 16000
    max_duration_s: float = 30.0
    max_label_tokens: int = 160
    batch_size: int = 16
    num_workers: int = 8
    seed: int = 42
    #: Cap the training split for smoke runs; None uses all 79,015 examples.
    max_train_examples: int | None = None
    max_eval_examples: int | None = 1000


@dataclass
class ModelConfig:
    #: Whisper already performs X->English speech translation with task="translate",
    #: so the zero-shot model is a real baseline rather than a random starting point.
    name: str = "openai/whisper-small"
    language: str = "spanish"
    task: str = "translate"
    freeze_encoder: bool = False
    dropout: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    amp: bool = True
    eval_every: int = 1
    out_dir: str = "runs/baseline"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def with_overrides(self, **sections: dict[str, Any]) -> Config:
        updated = {
            name: replace(getattr(self, name), **sections.get(name, {}))
            for name in ("data", "model", "train")
        }
        return Config(**updated)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
