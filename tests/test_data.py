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


class TestCorruptRows:
    """Two of 79,015 CoVoST 2 training rows carry zero audio bytes."""

    def test_corrupt_rows_are_skipped_by_default(self, shards_with_corrupt_row):
        examples = list(iter_examples(shards_with_corrupt_row, "train"))
        assert len(examples) == 4  # 6 rows, 2 unusable

    def test_surviving_rows_keep_their_text(self, shards_with_corrupt_row):
        examples = list(iter_examples(shards_with_corrupt_row, "train"))
        assert [e.translation for e in examples] == [
            "sentence 0", "sentence 1", "sentence 3", "sentence 5"
        ]

    def test_raise_mode_fails_loudly(self, shards_with_corrupt_row):
        with pytest.raises(RuntimeError, match="row 2"):
            list(iter_examples(shards_with_corrupt_row, "train", on_corrupt="raise"))

    def test_unknown_policy_is_rejected(self, shards_with_corrupt_row):
        with pytest.raises(ValueError, match="on_corrupt must be"):
            list(iter_examples(shards_with_corrupt_row, "train", on_corrupt="ignore"))

    def test_limit_counts_usable_rows_only(self, shards_with_corrupt_row):
        assert len(list(iter_examples(shards_with_corrupt_row, "train", limit=3))) == 3

    def test_dataset_builds_despite_corrupt_rows(self, shards_with_corrupt_row):
        assert len(CoVoST2Dataset(shards_with_corrupt_row, "train")) == 4
