import numpy as np
import pytest


@pytest.fixture
def tone():
    """One second of 440 Hz at 48 kHz, the rate CoVoST 2 actually ships."""
    sample_rate = 48000
    t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32), sample_rate


@pytest.fixture
def parquet_shards(tmp_path):
    """A miniature cache with the real column names and encoded audio."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf

    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        buffer = io.BytesIO()
        audio = rng.standard_normal(48000 * 2).astype(np.float32) * 0.01
        sf.write(buffer, audio, 48000, format="WAV")
        rows.append(
            {
                "audio": {"bytes": buffer.getvalue(), "path": f"{i}.wav"},
                "sentence": f"frase {i}",
                "translation": f"sentence {i}",
            }
        )

    for split in ("train", "validation"):
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, tmp_path / f"{split}-00000-of-00001.parquet")
    return tmp_path
