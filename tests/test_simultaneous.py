"""The policy is tested against a stub model, so no weights are needed.

What matters is the decision rule: a token whose attention lands near the edge
of the received audio must be withheld, and a committed token must never be
taken back.
"""

import numpy as np
import pytest
import torch

from simulst.simultaneous import (
    FRAMES_PER_SECOND,
    REPLACEMENT,
    AlignAttConfig,
    SimultaneousTranslator,
    alignment_heads,
    attended_frame,
    run_stream,
)


class StubTokenizer:
    eos_token_id = 99

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(f"t{i}" for i in ids)


class StubFeatureExtractor:
    sampling_rate = 16000


class StubProcessor:
    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.feature_extractor = StubFeatureExtractor()

    def __call__(self, audio, sampling_rate, return_tensors):
        return type("Features", (), {"input_features": torch.zeros(1, 80, 3000)})()

    def get_decoder_prompt_ids(self, language, task):
        return [(1, 11), (2, 12), (3, 13)]


class StubModel:
    """Emits tokens whose attention sits at frames given by ``frames``."""

    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self, frames, vocab=100):
        self.frames = list(frames)
        self.vocab = vocab
        self.calls = 0
        self.config = type("C", (), {"decoder_start_token_id": 10})()
        self.generation_config = type("G", (), {"alignment_heads": None})()
        def encoder(features):
            return type("E", (), {"last_hidden_state": torch.zeros(1, 1500, 4)})()

        self.model = type("M", (), {"encoder": staticmethod(encoder)})()

    def __call__(self, **kwargs):
        # Each policy step restarts generation from the committed prefix, which
        # arrives as a multi-token decoder input. Reset there, or the stub keeps
        # a counter the real model does not have.
        if kwargs["decoder_input_ids"].shape[1] > 1:
            self.calls = 0

        # A real model stops on its own; a stub that never emits EOT would run
        # to max_new_tokens and make the policy look broken.
        if self.calls >= len(self.frames):
            logits = torch.zeros(1, 1, self.vocab)
            logits[0, 0, StubTokenizer.eos_token_id] = 10.0
            attention = torch.zeros(1, 2, 1, 1500)
            return type(
                "Out", (), {"logits": logits, "cross_attentions": (attention,),
                            "past_key_values": None}
            )()

        index = self.calls
        frame = self.frames[index]
        self.calls += 1

        logits = torch.zeros(1, 1, self.vocab)
        logits[0, 0, 20 + index] = 10.0  # a distinct token per step

        attention = torch.zeros(1, 2, 1, 1500)
        attention[0, :, 0, frame] = 1.0
        return type(
            "Out", (), {"logits": logits, "cross_attentions": (attention,),
                        "past_key_values": None}
        )()


def silence(seconds, rate=16000):
    return np.zeros(int(seconds * rate), dtype=np.float32)


class TestAttendedFrame:
    def test_picks_the_argmax_frame(self):
        attention = torch.zeros(1, 4, 1, 1500)
        attention[0, :, 0, 317] = 1.0
        assert attended_frame((attention,), None, -1) == 317

    def test_averages_the_declared_alignment_heads(self):
        attention = torch.zeros(1, 4, 1, 1500)
        attention[0, 0, 0, 100] = 1.0  # a head we are told to use
        attention[0, 3, 0, 900] = 5.0  # a louder head we are told to ignore
        assert attended_frame((attention,), [(0, 0)], -1) == 100

    def test_without_heads_every_head_counts(self):
        attention = torch.zeros(1, 4, 1, 1500)
        attention[0, 3, 0, 900] = 5.0
        assert attended_frame((attention,), None, -1) == 900


class TestAlignmentHeads:
    def test_absent_heads_report_none(self):
        model = StubModel([0])
        assert alignment_heads(model) is None

    def test_declared_heads_are_normalised_to_pairs(self):
        model = StubModel([0])
        model.generation_config.alignment_heads = [[5, 2], [6, 1]]
        assert alignment_heads(model) == [(5, 2), (6, 1)]


class TestPolicy:
    def test_a_token_at_the_edge_is_withheld(self):
        """One second buffered means frame 50; a token at 48 is inside the margin."""
        translator = SimultaneousTranslator(StubModel([48]), StubProcessor(),
                                            AlignAttConfig(frame_threshold=6))
        assert translator.push(silence(1.0)) == []

    def test_a_token_on_settled_audio_is_emitted(self):
        translator = SimultaneousTranslator(StubModel([10]), StubProcessor(),
                                            AlignAttConfig(frame_threshold=6))
        assert len(translator.push(silence(1.0))) == 1

    def test_the_threshold_is_what_decides(self):
        """Same audio, same attention: only the dial changes the outcome."""
        patient = SimultaneousTranslator(StubModel([40]), StubProcessor(),
                                         AlignAttConfig(frame_threshold=20))
        eager = SimultaneousTranslator(StubModel([40]), StubProcessor(),
                                       AlignAttConfig(frame_threshold=2))
        assert patient.push(silence(1.0)) == []
        assert len(eager.push(silence(1.0))) == 1

    def test_finish_flushes_what_the_policy_was_holding(self):
        translator = SimultaneousTranslator(StubModel([48]), StubProcessor(),
                                            AlignAttConfig(frame_threshold=6))
        translator.push(silence(1.0))
        assert translator.state.token_ids == []
        assert len(translator.finish()) >= 1

    def test_emissions_are_stamped_with_the_audio_they_had(self):
        translator = SimultaneousTranslator(StubModel([10]), StubProcessor(),
                                            AlignAttConfig(frame_threshold=6))
        emission = translator.push(silence(2.0))[0]
        assert emission.time_s == pytest.approx(2.0)

    def test_committed_tokens_are_never_rewritten(self):
        """The defining property: emitted text only ever grows."""
        translator = SimultaneousTranslator(StubModel([5, 10, 15, 20]), StubProcessor(),
                                            AlignAttConfig(frame_threshold=2))
        seen = []
        for _ in range(4):
            translator.push(silence(0.5))
            seen.append(translator.text)
        assert all(b.startswith(a) for a, b in zip(seen[:-1], seen[1:], strict=True))

    def test_eot_stops_generation(self):
        model = StubModel([10])
        model.vocab = 100
        translator = SimultaneousTranslator(model, StubProcessor(), AlignAttConfig())
        translator.push(silence(1.0))
        assert translator.state.token_ids  # produced something and stopped cleanly

    def test_reset_clears_the_stream(self):
        translator = SimultaneousTranslator(StubModel([10]), StubProcessor(), AlignAttConfig())
        translator.push(silence(1.0))
        translator.reset()
        assert translator.text == "" and translator.elapsed_s == 0.0

    def test_empty_audio_emits_nothing(self):
        translator = SimultaneousTranslator(StubModel([10]), StubProcessor(), AlignAttConfig())
        assert translator.push(np.zeros(0, dtype=np.float32)) == []


class TestFrameGeometry:
    def test_whisper_encodes_thirty_seconds_into_fifteen_hundred_frames(self):
        assert 30 * FRAMES_PER_SECOND == 1500


class TestRunStream:
    def test_replay_returns_text_and_emissions(self):
        text, emissions = run_stream(
            StubModel([5, 10, 15]), StubProcessor(), silence(1.5),
            AlignAttConfig(frame_threshold=2, chunk_s=0.5),
        )
        assert isinstance(text, str) and len(emissions) >= 1

    def test_emission_times_never_go_backwards(self):
        _, emissions = run_stream(
            StubModel([5, 10, 15, 20]), StubProcessor(), silence(2.0),
            AlignAttConfig(frame_threshold=2, chunk_s=0.5),
        )
        times = [e.time_s for e in emissions]
        assert times == sorted(times)


class TestMultibyteCharacters:
    """Byte-level BPE splits characters; the viewer must never see the pieces."""

    class SplitTokenizer(StubTokenizer):
        """Token 20 is half of a character, completed by token 21."""

        def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            body = [i for i in ids if i not in (10, 11, 12, 13)]
            if body == [20]:
                return "don" + REPLACEMENT
            if body == [20, 21]:
                return "don\u2019t"
            return "".join(f"t{i}" for i in body)

    def processor(self):
        processor = StubProcessor()
        processor.tokenizer = self.SplitTokenizer()
        return processor

    def test_an_incomplete_character_is_not_shown(self):
        translator = SimultaneousTranslator(
            StubModel([5]), self.processor(), AlignAttConfig(frame_threshold=2)
        )
        assert translator.push(silence(1.0)) == []

    def test_the_character_appears_whole_once_it_resolves(self):
        translator = SimultaneousTranslator(
            StubModel([5, 6]), self.processor(), AlignAttConfig(frame_threshold=2)
        )
        emitted = translator.push(silence(1.0))
        assert "".join(e.text for e in emitted) == "don\u2019t"

    def test_no_emission_ever_contains_the_replacement_glyph(self):
        translator = SimultaneousTranslator(
            StubModel([5, 6]), self.processor(), AlignAttConfig(frame_threshold=2)
        )
        translator.push(silence(1.0))
        translator.finish()
        assert all(REPLACEMENT not in e.text for e in translator.state.emissions)

    def test_concatenated_emissions_equal_the_committed_text(self):
        """The display and the committed text must not drift apart."""
        translator = SimultaneousTranslator(
            StubModel([5, 6]), self.processor(), AlignAttConfig(frame_threshold=2)
        )
        translator.push(silence(1.0))
        joined = "".join(e.text for e in translator.state.emissions)
        assert joined == translator.text
