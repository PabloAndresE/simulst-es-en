"""Two baselines AlignAtt has to beat.

A curve against the offline ceiling says how much quality a policy gives up,
but not whether the policy was worth implementing. These are the obvious
alternatives, sharing the streaming translator's interface so the sweep can run
all three over identical audio.

**wait-k** (Ma et al., 2019): wait for *k* chunks of audio, then emit one token
per chunk forever. It reads nothing from the model, which is the point — if a
rule this blunt matches AlignAtt, the attention machinery earned nothing.

**Re-translation**: every chunk, translate the whole prefix from scratch and
show the result. It is what a naive implementation does, it is allowed to
rewrite the screen, and its quality is strong for exactly that reason. The cost
lands in the revision rate, which is the column where it loses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from simulst.simultaneous import (
    REPLACEMENT,
    Emission,
    StreamState,
    no_speech_token_id,
)


@dataclass
class WaitKConfig:
    """Attributes:
    k: Chunks of audio to accumulate before the first token.
    chunk_s: Seconds per chunk; the latency floor.
    tokens_per_chunk: Tokens released per chunk once emission starts.
    max_new_tokens: Safety cap per step.
    no_speech_threshold: As in :class:`~simulst.simultaneous.AlignAttConfig`.
    """

    k: int = 3
    chunk_s: float = 1.0
    tokens_per_chunk: int = 2
    max_new_tokens: int = 24
    no_speech_threshold: float = 0.5


@dataclass
class RetranslationConfig:
    """Attributes:
    chunk_s: How often the whole prefix is retranslated.
    min_audio_s: Audio required before the first attempt.
    no_speech_threshold: As above.
    """

    chunk_s: float = 1.0
    min_audio_s: float = 1.0
    no_speech_threshold: float = 0.5


class _Base:
    """Shared plumbing: audio buffer, no-speech gate, decoded text."""

    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.config = config
        self.sample_rate = processor.feature_extractor.sampling_rate
        self.state = StreamState()

        prompt = processor.get_decoder_prompt_ids(language="spanish", task="translate")
        self._prompt_ids = [model.config.decoder_start_token_id] + [tok for _, tok in prompt]
        self._eot = processor.tokenizer.eos_token_id
        self._suppressed = torch.tensor(
            sorted(set(getattr(model.generation_config, "suppress_tokens", None) or [])),
            dtype=torch.long,
        )
        self._no_speech = no_speech_token_id(processor.tokenizer)

    def reset(self) -> None:
        self.state = StreamState()

    @property
    def elapsed_s(self) -> float:
        return len(self.state.audio) / self.sample_rate

    @property
    def text(self) -> str:
        return self.processor.tokenizer.decode(
            self.state.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _encode(self):
        features = self.processor(
            self.state.audio, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.to(self.model.device, dtype=self.model.dtype)
        return self.model.model.encoder(features)

    def _is_speech(self, encoder_outputs) -> bool:
        if self.config.no_speech_threshold >= 1.0:
            return True
        start = torch.tensor([[self._prompt_ids[0]]], device=self.model.device)
        probe = self.model(encoder_outputs=encoder_outputs, decoder_input_ids=start)
        probability = torch.softmax(probe.logits[0, -1].float(), -1)[self._no_speech]
        return float(probability) <= self.config.no_speech_threshold

    def _greedy(self, encoder_outputs, prefix: list[int], limit: int) -> list[int]:
        """Decode `limit` tokens greedily, continuing from `prefix`."""
        decoder_input = torch.tensor([self._prompt_ids + prefix], device=self.model.device)
        past, produced = None, []
        for _ in range(limit):
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            logits = outputs.logits[0, -1].clone()
            if self._suppressed.numel():
                logits[self._suppressed] = float("-inf")
            token = int(logits.argmax().item())
            if token == self._eot:
                break
            produced.append(token)
            decoder_input = torch.tensor([[token]], device=self.model.device)
        return produced


class WaitKTranslator(_Base):
    """Emit on a schedule, ignoring what the model is attending to."""

    def __init__(self, model, processor, config: WaitKConfig | None = None):
        super().__init__(model, processor, config or WaitKConfig())
        self._chunks = 0

    def reset(self) -> None:
        super().reset()
        self._chunks = 0

    @torch.no_grad()
    def push(self, chunk: np.ndarray) -> list[Emission]:
        self.state.audio = np.concatenate([self.state.audio, np.asarray(chunk, dtype=np.float32)])
        self._chunks += 1
        if self._chunks < self.config.k:
            return []
        return self._emit(self.config.tokens_per_chunk)

    @torch.no_grad()
    def finish(self) -> list[Emission]:
        return self._emit(self.config.max_new_tokens)

    def _emit(self, budget: int) -> list[Emission]:
        if len(self.state.audio) == 0:
            return []
        encoder_outputs = self._encode()
        if not self._is_speech(encoder_outputs):
            return []

        produced = self._greedy(encoder_outputs, self.state.token_ids, budget)
        if not produced:
            return []

        self.state.token_ids.extend(produced)
        decoded = self.text
        if decoded.endswith(REPLACEMENT):
            return []
        fresh = decoded[len(self.state.shown) :]
        self.state.shown = decoded
        if not fresh:
            return []
        # One emission per produced token, so Average Lagging counts the same
        # units it does for AlignAtt.
        emissions = [
            Emission(text=fresh if i == 0 else "", time_s=self.elapsed_s, frame=-1)
            for i in range(len(produced))
        ]
        self.state.emissions.extend(emissions)
        return emissions


class RetranslationTranslator(_Base):
    """Translate the whole prefix again on every chunk, rewriting freely."""

    def __init__(self, model, processor, config: RetranslationConfig | None = None):
        super().__init__(model, processor, config or RetranslationConfig())

    @torch.no_grad()
    def push(self, chunk: np.ndarray) -> list[Emission]:
        self.state.audio = np.concatenate([self.state.audio, np.asarray(chunk, dtype=np.float32)])
        if self.elapsed_s < self.config.min_audio_s:
            return []
        return self._retranslate()

    @torch.no_grad()
    def finish(self) -> list[Emission]:
        return self._retranslate()

    def _retranslate(self) -> list[Emission]:
        if len(self.state.audio) == 0:
            return []
        encoder_outputs = self._encode()
        if not self._is_speech(encoder_outputs):
            return []

        # From scratch every time: that is the definition of the baseline.
        previous = len(self.state.token_ids)
        self.state.token_ids = self._greedy(encoder_outputs, [], 200)
        decoded = self.text
        self.state.shown = decoded
        # Only the tokens beyond the previous length are "new"; the rest were
        # already on screen, possibly as different words.
        grown = max(len(self.state.token_ids) - previous, 0)
        emissions = [Emission(text="", time_s=self.elapsed_s, frame=-1) for _ in range(grown)]
        self.state.emissions.extend(emissions)
        return emissions


#: Name -> (translator class, config class). Used by the sweep.
POLICIES = {
    "waitk": (WaitKTranslator, WaitKConfig),
    "retranslate": (RetranslationTranslator, RetranslationConfig),
}
