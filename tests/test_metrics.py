import pytest

from simulst.metrics import Emission, average_lagging, corpus_bleu, revision_rate


def test_bleu_is_perfect_for_identical_text():
    refs = ["the meeting was moved to three", "it lives in shallow water"]
    assert corpus_bleu(refs, refs) == pytest.approx(100.0)


def test_bleu_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="hypotheses against"):
        corpus_bleu(["a"], ["a", "b"])


def test_waiting_for_the_whole_utterance_lags_most():
    """A system that emits only at the end is maximally laggy."""
    duration, tokens = 6.0, 4
    offline = [Emission("w", duration) for _ in range(tokens)]
    streaming = [Emission("w", 1.5 * (i + 1)) for i in range(tokens)]
    assert average_lagging(offline, duration, tokens) > average_lagging(
        streaming, duration, tokens
    )


def test_lagging_is_near_zero_for_an_oracle_rate():
    duration, tokens = 4.0, 8
    emissions = [Emission("w", (i + 1) * duration / tokens) for i in range(tokens)]
    assert average_lagging(emissions, duration, tokens) < duration / tokens + 1e-6


def test_lagging_is_undefined_for_an_empty_hypothesis():
    assert average_lagging([], 5.0, 0) is None


def test_prefix_extensions_are_not_revisions():
    assert revision_rate(["the", "the meeting", "the meeting was"]) == 0.0


def test_rewrites_count_as_revisions():
    assert revision_rate(["the meeting", "a meeting", "a meeting was"]) == pytest.approx(0.5)


def test_single_output_has_no_revisions():
    assert revision_rate(["the meeting"]) == 0.0
