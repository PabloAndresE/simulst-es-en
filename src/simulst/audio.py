"""Audio decoding and resampling.

CoVoST 2 parquet rows carry encoded audio bytes, not arrays. Decoding them with
``soundfile`` avoids a hard dependency on ``torchcodec``, which newer versions of
``datasets`` demand for automatic audio decoding and which is awkward to install
on some platforms.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def decode(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode encoded audio bytes to mono float32 and its native sample rate."""
    array, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, sample_rate


def resample(array: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear-interpolation resampling.

    CoVoST 2 audio is 48 kHz despite the dataset card advertising 16 kHz, and
    Whisper's feature extractor assumes 16 kHz. Feeding it 48 kHz silently
    produces mel features for audio three times too fast, which trains without
    error and never converges.
    """
    if source_rate == target_rate:
        return array.astype(np.float32, copy=False)
    if array.size == 0:
        return array.astype(np.float32, copy=False)

    duration = array.shape[0] / source_rate
    target_length = int(round(duration * target_rate))
    source_positions = np.linspace(0.0, duration, num=array.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, array).astype(np.float32)


def load(raw: bytes, target_rate: int = 16000) -> np.ndarray:
    """Decode and resample in one step."""
    array, sample_rate = decode(raw)
    return resample(array, sample_rate, target_rate)
