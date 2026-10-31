"""Whisper as a Spanish-to-English speech translation model.

Whisper ships an X->English translation task, so `openai/whisper-small` with
``task="translate"`` already performs this job. That makes the zero-shot model a
genuine baseline to beat rather than an arbitrary initialisation, and it is the
number any fine-tune must be reported against.
"""

from __future__ import annotations

import torch

from simulst.config import ModelConfig


def build(cfg: ModelConfig, device: str = "cpu"):
    """Return ``(model, processor)`` configured for es->en translation."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(
        cfg.name, language=cfg.language, task=cfg.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg.name)

    # Without this the model falls back to transcription and "translates"
    # Spanish audio into Spanish text, which scores a plausible-looking but
    # meaningless BLEU against English references.
    forced = processor.get_decoder_prompt_ids(language=cfg.language, task=cfg.task)
    model.generation_config.forced_decoder_ids = forced
    model.generation_config.language = cfg.language
    model.generation_config.task = cfg.task
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # Whisper's own 448-token limit is the right cap; passing max_new_tokens as
    # well makes transformers warn on every generate() call, so generation
    # length is controlled here once instead of at each call site.
    model.generation_config.max_length = 448

    if cfg.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        model.model.encoder._freeze_parameters()

    return model.to(device), processor


def param_counts(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "frozen": total - trainable,
        "total": total,
        "trainable_pct": round(100 * trainable / total, 2),
    }


@torch.no_grad()
def translate(model, processor, input_features: torch.Tensor, num_beams: int = 1):
    """Greedy decode a batch of Whisper features into English text.

    ``clean_up_tokenization_spaces=False`` is deliberate: the clean-up step is
    built for WordPiece and strips spaces before punctuation on a BPE tokenizer
    like Whisper's, which silently alters the hypothesis and therefore the BLEU
    it is scored with.
    """
    generated = model.generate(input_features.to(model.device), num_beams=num_beams)
    return processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
