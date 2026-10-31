import pytest

from simulst.data import CoVoST2Dataset, iter_examples, shard_paths


def test_finds_shards(parquet_shards):
    assert len(shard_paths(parquet_shards, "train")) == 1


def test_missing_split_names_the_fix(parquet_shards):
    with pytest.raises(FileNotFoundError, match="download_data"):
        shard_paths(parquet_shards, "test")


def test_examples_are_resampled_to_the_target_rate(parquet_shards):
    example = next(iter_examples(parquet_shards, "train", target_rate=16000))
    # 2 s of 48 kHz audio becomes 2 s of 16 kHz audio.
    assert abs(example.duration_s - 2.0) < 0.01
    assert len(example.audio) == pytest.approx(32000, abs=2)


def test_limit_caps_the_number_of_examples(parquet_shards):
    assert len(list(iter_examples(parquet_shards, "train", limit=3))) == 3


def test_dataset_keeps_clips_within_the_cap(parquet_shards):
    assert len(CoVoST2Dataset(parquet_shards, "train", max_duration_s=30.0)) == 6


def test_empty_split_is_an_error(parquet_shards):
    with pytest.raises(RuntimeError, match="no usable examples"):
        CoVoST2Dataset(parquet_shards, "train", max_duration_s=0.1)


def test_total_hours(parquet_shards):
    dataset = CoVoST2Dataset(parquet_shards, "train")
    assert abs(dataset.total_hours - 6 * 2 / 3600) < 1e-6
