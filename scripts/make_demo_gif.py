#!/usr/bin/env python
"""Render the policy's output as a terminal animation.

The static README table says the lag is 1.75 s. This shows what 1.75 s feels
like: English appearing while the Spanish is still being spoken, never
rewritten once shown.

    python scripts/make_demo_gif.py --file docs/assets/sample_es.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from simulst.config import Config
from simulst.live import read_wav
from simulst.model import build
from simulst.simultaneous import AlignAttConfig, SimultaneousTranslator, translate_offline

WIDTH, HEIGHT, FPS = 900, 360, 10
BACKGROUND, FOREGROUND = (16, 18, 22), (232, 234, 238)
DIM, ACCENT, GREEN = (120, 126, 136), (122, 190, 255), (126, 211, 143)
MUTED = (78, 83, 92)


def load_font(size: int, bold: bool = False):
    for name in ("Menlo.ttc", "DejaVuSansMono.ttf", "Courier New.ttf"):
        for directory in ("/System/Library/Fonts", "/Library/Fonts",
                          "/usr/share/fonts/truetype/dejavu"):
            path = Path(directory) / name
            if path.exists():
                return ImageFont.truetype(str(path), size, index=1 if bold and
                                          name.endswith("ttc") else 0)
    return ImageFont.load_default()


#: Typographic punctuation the model emits but a monospace terminal font may
#: have no glyph for, which renders as a tofu box. Display only -- the metrics
#: and the committed text keep the characters the model actually produced.
PUNCTUATION = {
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2014": "--", "\u2013": "-", "\u2026": "...", "\u00a0": " ",
}


def displayable(text: str) -> str:
    for fancy, plain in PUNCTUATION.items():
        text = text.replace(fancy, plain)
    return text


def wrap(tokens: list[str], width: int) -> list[str]:
    lines, current = [], ""
    for token in tokens:
        if len(current) + len(token) > width:
            lines.append(current)
            current = token.lstrip()
        else:
            current += token
    if current:
        lines.append(current)
    return lines


def render(elapsed: float, total: float, tokens: list[str], offline: str, font, small, bold):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((28, 24), "simultaneous speech translation  es -> en", font=bold, fill=FOREGROUND)
    draw.text((28, 50), "AlignAtt policy, Whisper small fine-tuned on CoVoST 2",
              font=small, fill=DIM)

    # Progress bar: how much of the utterance has been spoken.
    bar_y, bar_w = 86, WIDTH - 56
    draw.rounded_rectangle([28, bar_y, 28 + bar_w, bar_y + 8], 4, fill=(38, 42, 50))
    played = int(bar_w * min(elapsed / total, 1.0))
    draw.rounded_rectangle([28, bar_y, 28 + played, bar_y + 8], 4, fill=ACCENT)
    draw.text((28, bar_y + 18), f"audio {min(elapsed, total):4.1f}s / {total:.1f}s",
              font=small, fill=DIM)

    draw.text((28, 140), "ENGLISH, AS IT IS DECIDED", font=small, fill=DIM)
    for index, line in enumerate(wrap([displayable(t) for t in tokens], 58)):
        draw.text((28, 166 + index * 30), line, font=font, fill=FOREGROUND)

    # The offline row is drawn from the first frame on purpose: an empty line
    # next to a filling one is the whole argument for the policy.
    draw.line([(28, HEIGHT - 104), (WIDTH - 28, HEIGHT - 104)], fill=(38, 42, 50))
    draw.text((28, HEIGHT - 88), "OFFLINE, WAITING FOR THE WHOLE UTTERANCE",
              font=small, fill=DIM)
    if elapsed >= total:
        draw.text((28, HEIGHT - 62), displayable(offline)[:64], font=small, fill=GREEN)
    else:
        draw.text((28, HEIGHT - 62), f"still listening, {total - elapsed:.1f}s to go",
                  font=small, fill=MUTED)

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="docs/assets/sample_es.wav")
    parser.add_argument("--out", default="docs/assets/demo.gif")
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--checkpoint", default=None, help="fine-tuned weights")
    parser.add_argument("--frame-threshold", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.checkpoint:
        cfg.model.name = args.checkpoint
    model, processor = build(cfg.model, args.device, attn_implementation="eager")
    model.eval()

    audio = read_wav(args.file)
    total = len(audio) / 16000

    translator = SimultaneousTranslator(
        model, processor, AlignAttConfig(frame_threshold=args.frame_threshold)
    )
    step = int(translator.config.chunk_s * 16000)
    timeline: list[tuple[float, str]] = []
    for start in range(0, len(audio), step):
        for emission in translator.push(audio[start : start + step]):
            timeline.append((emission.time_s, emission.text))
    for emission in translator.finish():
        timeline.append((emission.time_s, emission.text))

    offline = translate_offline(model, processor, audio).strip()
    print(f"streaming: {translator.text.strip()}")
    print(f"offline  : {offline}")

    font, small, bold = load_font(24), load_font(15), load_font(19, bold=True)
    frames = []
    for index in range(int((total + 2.0) * FPS)):
        elapsed = index / FPS
        shown = [text for time_s, text in timeline if time_s <= elapsed]
        frames.append(render(elapsed, total, shown, offline, font, small, bold))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
