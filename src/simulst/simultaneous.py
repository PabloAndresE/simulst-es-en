"""AlignAtt: emit a token only when it is not looking at the edge of the audio.

Papi et al., *AlignAtt: Using Attention-based Audio-Translation Alignments as a
Guide for Simultaneous Speech Translation* (2023).

The idea in one sentence: generate greedily on the audio received so far, and
for each candidate token ask which audio frame its cross-attention points at.
A token attending to the last few frames is leaning on sound that is still
arriving, so it is withheld; a token attending to settled audio is safe to emit
and is committed forever.

Committing forever is what makes this different from re-translation. A
re-translation system reruns the whole utterance each time and happily rewrites
what the viewer already read; AlignAtt never revises, so its revision rate is
zero by construction and its cost shows up as latency instead.

**Whisper's frame geometry.** The encoder always sees a 30 s window padded to
1500 frames, so one frame is 20 ms regardless of how much real audio arrived.
With ``t`` seconds buffered, real audio ends at frame ``t * 50`` and everything
past it is padding. The policy therefore compares against ``t * 50 - f``, not
against 1500.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

#: Whisper encodes 30 s into 1500 frames.
FRAMES_PER_SECOND = 50.0


@dataclass
class AlignAttConfig:
    """Policy knobs.

    Attributes:
        frame_threshold: How many frames at the edge of the audio count as
            "still arriving". Larger values wait longer and translate better.
            This is the quality-latency dial; the paper sweeps it.
        chunk_s: How much new audio arrives per step. The floor on latency.
        max_new_tokens: Safety cap per step.
        alignment_layer: Decoder layer whose cross-attention is read when the
            checkpoint declares no alignment heads. ``-1`` is the last layer.
    """

    frame_threshold: int = 6
    chunk_s: float = 0.5
    max_new_tokens: int = 24
    alignment_layer: int = -1
    #: Audio required before decoding starts at all. Whisper hallucinates on
    #: fragments, and this policy cannot retract what it emits.
    min_audio_s: float = 1.0


@dataclass
class Emission:
    """A token committed to the viewer, and when it was committed."""

    text: str
    time_s: float
    frame: int


@dataclass
class StreamState:
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    token_ids: list[int] = field(default_factory=list)
    emissions: list[Emission] = field(default_factory=list)
    finished: bool = False


def alignment_heads(model) -> list[tuple[int, int]] | None:
    """Whisper ships (layer, head) pairs whose attention tracks audio position.

    These are the heads the timestamp feature uses, so they are the right place
    to read alignment from. Not every checkpoint declares them.
    """
    heads = getattr(model.generation_config, "alignment_heads", None)
    if not heads:
        return None
    return [(int(layer), int(head)) for layer, head in heads]


def attended_frame(cross_attentions, heads: list[tuple[int, int]] | None, layer: int) -> int:
    """Which encoder frame the newest token is looking at.

    ``cross_attentions`` is one tuple entry per decoder layer, each shaped
    ``[batch, heads, query, frames]``. Averaging the declared alignment heads is
    more stable than any single head.

    Raises:
        RuntimeError: if the tuple is empty, which is what a model loaded with
            the SDPA kernel returns. That failure is otherwise a warning on
            stderr and an index error twenty lines later.
    """
    if not cross_attentions:
        raise RuntimeError(
            "the model returned no cross-attention; load it with "
            'attn_implementation="eager" (SDPA cannot expose attention weights)'
        )
    if heads:
        stacked = torch.stack([cross_attentions[layer][0, head, -1] for layer, head in heads])
        weights = stacked.mean(0)
    else:
        weights = cross_attentions[layer][0, :, -1].mean(0)
    return int(weights.argmax().item())


class SimultaneousTranslator:
    """Drive a Whisper checkpoint under the AlignAtt policy.

    The model must be loaded with ``attn_implementation="eager"``; see
    :func:`attended_frame`.

    Usage is push-based: feed audio as it arrives, read whatever became safe.

        translator = SimultaneousTranslator(model, processor)
        for chunk in microphone():
            for emission in translator.push(chunk):
                print(emission.text, end="", flush=True)
        translator.finish()
    """

    def __init__(self, model, processor, config: AlignAttConfig | None = None):
        self.model = model
        self.processor = processor
        self.config = config or AlignAttConfig()
        self.heads = alignment_heads(model)
        self.sample_rate = processor.feature_extractor.sampling_rate
        self.state = StreamState()

        prompt = processor.get_decoder_prompt_ids(language="spanish", task="translate")
        self._prompt_ids = [model.config.decoder_start_token_id] + [tok for _, tok in prompt]
        self._eot = processor.tokenizer.eos_token_id
        # Decoding by hand means none of generate()'s logits processors run, so
        # the suppression lists have to be applied here or they do nothing.
        self._suppressed = torch.tensor(
            sorted(set(getattr(model.generation_config, "suppress_tokens", None) or [])),
            dtype=torch.long,
        )

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

    def push(self, chunk: np.ndarray) -> list[Emission]:
        """Add audio and return whatever the policy now considers safe."""
        self.state.audio = np.concatenate([self.state.audio, np.asarray(chunk, dtype=np.float32)])
        return self._step(final=False)

    def finish(self) -> list[Emission]:
        """Flush the tail: no audio is still arriving, so the edge rule is off."""
        emitted = self._step(final=True)
        self.state.finished = True
        return emitted

    @torch.no_grad()
    def _step(self, final: bool) -> list[Emission]:
        if len(self.state.audio) == 0:
            return []
        if not final and self.elapsed_s < self.config.min_audio_s:
            return []

        features = self.processor(
            self.state.audio, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.to(self.model.device, dtype=self.model.dtype)

        encoder_outputs = self.model.model.encoder(features)
        # Frames beyond the real audio are padding; the policy must not trust them.
        limit = len(self.state.audio) / self.sample_rate * FRAMES_PER_SECOND
        cutoff = limit if final else limit - self.config.frame_threshold

        tokens = list(self._prompt_ids) + list(self.state.token_ids)
        decoder_input = torch.tensor([tokens], device=self.model.device)
        past, fresh = None, []

        for _ in range(self.config.max_new_tokens):
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input,
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
            )
            past = outputs.past_key_values
            logits = outputs.logits[0, -1].clone()
            if self._suppressed.numel():
                logits[self._suppressed] = float("-inf")
            token = int(logits.argmax().item())
            if token == self._eot:
                break

            frame = attended_frame(
                outputs.cross_attentions, self.heads, self.config.alignment_layer
            )
            # The token leans on audio still arriving: stop and wait for more.
            if not final and frame >= cutoff:
                break

            # Decoding the token alone is wrong for byte-level BPE: a token can
            # carry part of a multi-byte character and decodes to U+FFFD until
            # the rest arrives. Taking the delta of the decoded prefix instead
            # yields nothing while a character is incomplete and the whole
            # character once it is, so no replacement glyph ever reaches the
            # screen. The token ids themselves are append-only either way.
            before = self.text
            self.state.token_ids.append(token)
            emission = Emission(
                text=self.text[len(before) :],
                time_s=self.elapsed_s,
                frame=frame,
            )
            self.state.emissions.append(emission)
            fresh.append(emission)
            decoder_input = torch.tensor([[token]], device=self.model.device)

        return fresh


def translate_offline(model, processor, audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Wait for the whole utterance, then translate. The quality ceiling."""
    features = processor(
        audio, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        generated = model.generate(features, num_beams=1)
    return processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def run_stream(
    model,
    processor,
    audio: np.ndarray,
    config: AlignAttConfig | None = None,
    sample_rate: int = 16000,
) -> tuple[str, list[Emission]]:
    """Replay a recording through the policy as if it were arriving live."""
    translator = SimultaneousTranslator(model, processor, config)
    step = max(1, int(translator.config.chunk_s * sample_rate))
    for start in range(0, len(audio), step):
        translator.push(audio[start : start + step])
    translator.finish()
    return translator.text, translator.state.emissions
