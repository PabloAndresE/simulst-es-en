"""Quality and latency metrics.

Quality uses sacreBLEU so the numbers are comparable with the speech-translation
literature. Latency uses Average Lagging, the standard measure in simultaneous
translation: how far behind an ideal wait-free system the policy runs, in
seconds of source audio.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def corpus_bleu(hypotheses: list[str], references: list[str]) -> float:
    """sacreBLEU with default tokenisation (13a), as reported by IWSLT."""
    import sacrebleu

    if len(hypotheses) != len(references):
        raise ValueError(
            f"{len(hypotheses)} hypotheses against {len(references)} references"
        )
    return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)


@dataclass
class Emission:
    """One emitted token and how much source audio had been consumed."""

    token: str
    source_time_s: float


def average_lagging(
    emissions: list[Emission], source_duration_s: float, target_length: int
) -> float | None:
    """Average Lagging (Ma et al., 2019), in seconds.

    Measures how far each emission lags behind an oracle that emits target
    tokens at a constant rate across the utterance. Lower is more responsive;
    a system that waits for the whole utterance before speaking scores roughly
    the utterance duration.

    Returns ``None`` for an empty hypothesis, where lagging is undefined rather
    than zero.
    """
    if not emissions or target_length <= 0 or source_duration_s <= 0:
        return None

    rate = target_length / source_duration_s
    # Sum until the step that first consumes the entire source, per the
    # original formulation; emissions after that carry no extra delay.
    cutoff = len(emissions)
    for index, emission in enumerate(emissions):
        if emission.source_time_s >= source_duration_s:
            cutoff = index + 1
            break

    total = sum(
        emission.source_time_s - index / rate
        for index, emission in enumerate(emissions[:cutoff])
    )
    return float(total / cutoff)


def revision_rate(partial_outputs: list[str]) -> float:
    """Share of updates that rewrote previously shown text.

    A re-translation system produces flicker: text appears, then changes. This
    counts how often a new partial output is not a prefix-extension of the one
    before it, which is what a viewer perceives as instability.
    """
    if len(partial_outputs) < 2:
        return 0.0
    revisions = sum(
        1
        for previous, current in zip(partial_outputs[:-1], partial_outputs[1:], strict=True)
        if not current.startswith(previous)
    )
    return revisions / (len(partial_outputs) - 1)


def token_revision_rate(histories: Sequence[Sequence[int]]) -> float:
    """Revisions measured on token ids rather than on decoded text.

    This is where a commit-once policy's guarantee actually lives. The text
    version can report a small non-zero rate even when no token was ever
    retracted, because byte-level BPE decodes an incomplete multi-byte
    character as U+FFFD and rewrites it once the remaining bytes arrive.
    Measured on CoVoST 2, that artefact alone accounts for a rate of about
    0.01, entirely on curly quotes and accented characters.
    """
    if len(histories) < 2:
        return 0.0
    revisions = sum(
        1
        for previous, current in zip(histories[:-1], histories[1:], strict=True)
        if list(current[: len(previous)]) != list(previous)
    )
    return revisions / (len(histories) - 1)
