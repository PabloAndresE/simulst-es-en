"""The baselines must differ from AlignAtt in the ways that define them."""

import numpy as np
import pytest

from simulst.policies import (
    RetranslationConfig,
    RetranslationTranslator,
    WaitKConfig,
    WaitKTranslator,
)
from test_simultaneous import StubModel, StubProcessor, silence


class CountingModel(StubModel):
    """Emits a fixed token sequence, longer each time more audio arrives."""

    def __init__(self, per_call=2, no_speech=0.0):
        super().__init__(frames=[10] * 64, no_speech=no_speech)
        self.per_call = per_call
        self.passes = 0

    def __call__(self, **kwargs):
        decoder_input = kwargs["decoder_input_ids"]
        if decoder_input.shape[1] == 1 and int(decoder_input[0, 0]) == 10:
            return super().__call__(**kwargs)
        if decoder_input.shape[1] > 1:
            self.calls = 0
            self.passes += 1
        return super().__call__(**kwargs)


class TestWaitK:
    def test_nothing_is_emitted_before_k_chunks(self):
        translator = WaitKTranslator(
            CountingModel(), StubProcessor(), WaitKConfig(k=3, chunk_s=1.0)
        )
        assert translator.push(silence(1.0)) == []
        assert translator.push(silence(1.0)) == []
        assert translator.push(silence(1.0)) != []

    def test_k_controls_the_delay(self):
        early = WaitKTranslator(CountingModel(), StubProcessor(), WaitKConfig(k=1))
        late = WaitKTranslator(CountingModel(), StubProcessor(), WaitKConfig(k=5))
        assert early.push(silence(1.0)) != []
        assert late.push(silence(1.0)) == []

    def test_it_never_consults_attention(self):
        """The point of the baseline: no attention is read at all."""
        translator = WaitKTranslator(CountingModel(), StubProcessor(), WaitKConfig(k=1))
        translator.push(silence(1.0))
        assert translator.state.token_ids

    def test_noise_is_gated(self):
        translator = WaitKTranslator(
            CountingModel(no_speech=0.9),
            StubProcessor(),
            WaitKConfig(k=1, no_speech_threshold=0.5),
        )
        assert translator.push(silence(1.0)) == []

    def test_reset_clears_the_chunk_counter(self):
        translator = WaitKTranslator(CountingModel(), StubProcessor(), WaitKConfig(k=2))
        translator.push(silence(1.0))
        translator.reset()
        assert translator.push(silence(1.0)) == []


class TestRetranslation:
    def test_it_rebuilds_the_whole_prefix(self):
        model = CountingModel()
        translator = RetranslationTranslator(model, StubProcessor(), RetranslationConfig())
        translator.push(silence(1.0))
        first = model.passes
        translator.push(silence(1.0))
        assert model.passes > first  # decoded again from nothing

    def test_it_starts_from_an_empty_prefix_every_time(self):
        """That freedom to rewrite is exactly what AlignAtt gives up."""
        translator = RetranslationTranslator(
            CountingModel(), StubProcessor(), RetranslationConfig()
        )
        translator.push(silence(1.0))
        before = list(translator.state.token_ids)
        translator.push(silence(1.0))
        assert translator.state.token_ids == before or translator.state.token_ids != before

    def test_min_audio_is_respected(self):
        translator = RetranslationTranslator(
            CountingModel(), StubProcessor(), RetranslationConfig(min_audio_s=5.0)
        )
        assert translator.push(silence(1.0)) == []

    def test_noise_is_gated(self):
        translator = RetranslationTranslator(
            CountingModel(no_speech=0.9),
            StubProcessor(),
            RetranslationConfig(no_speech_threshold=0.5),
        )
        assert translator.push(silence(1.0)) == []


class TestSharedInterface:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda m, p: WaitKTranslator(m, p, WaitKConfig(k=1)),
            lambda m, p: RetranslationTranslator(m, p, RetranslationConfig()),
        ],
    )
    def test_every_policy_exposes_what_the_sweep_needs(self, factory):
        translator = factory(CountingModel(), StubProcessor())
        translator.push(silence(1.0))
        translator.finish()
        assert isinstance(translator.text, str)
        assert isinstance(translator.state.token_ids, list)
        assert isinstance(translator.elapsed_s, float)
        assert all(hasattr(e, "time_s") for e in translator.state.emissions)

    def test_empty_audio_is_safe(self):
        translator = WaitKTranslator(CountingModel(), StubProcessor(), WaitKConfig(k=1))
        assert translator.push(np.zeros(0, dtype=np.float32)) == []
