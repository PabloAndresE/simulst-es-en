"""Watch the policy translate, token by token, as the audio arrives.

Two sources:

    python -m simulst.live --file clip.wav       # replay a recording
    python -m simulst.live --mic                 # a microphone, needs sounddevice

The display is the point. Every token appears at the moment the policy decided
it was safe, with the elapsed audio time next to it, so the delay between a
word being spoken and its translation appearing is visible rather than
summarised. Nothing on screen is ever rewritten: AlignAtt commits.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from simulst.audio import resample
from simulst.config import Config
from simulst.model import build
from simulst.simultaneous import AlignAttConfig, SimultaneousTranslator, translate_offline

SAMPLE_RATE = 16000


def read_wav(path: str) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample(audio, rate, SAMPLE_RATE)


def stream_file(translator: SimultaneousTranslator, audio: np.ndarray, realtime: bool) -> None:
    """Replay a recording at wall-clock speed, so the delay is felt, not read."""
    step = max(1, int(translator.config.chunk_s * SAMPLE_RATE))
    started = time.time()

    for start in range(0, len(audio), step):
        chunk = audio[start : start + step]
        if realtime:
            # Hold the chunk until its audio would really have arrived.
            due = (start + len(chunk)) / SAMPLE_RATE
            time.sleep(max(0.0, due - (time.time() - started)))

        for emission in translator.push(chunk):
            print(f"\033[2m[{emission.time_s:5.1f}s]\033[0m {emission.text}", flush=True)

    for emission in translator.finish():
        print(f"\033[2m[{emission.time_s:5.1f}s]\033[0m\033[33m{emission.text}\033[0m", flush=True)


def stream_microphone(translator: SimultaneousTranslator) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("microphone mode needs sounddevice: pip install sounddevice")

    block = int(translator.config.chunk_s * SAMPLE_RATE)
    print("listening in Spanish, Ctrl-C to stop\n")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=block) as stream:
        try:
            while True:
                chunk, overflowed = stream.read(block)
                if overflowed:
                    print("\033[31m[audio dropped]\033[0m", file=sys.stderr)
                for emission in translator.push(chunk[:, 0]):
                    print(f"\033[2m[{emission.time_s:5.1f}s]\033[0m {emission.text}", flush=True)
        except KeyboardInterrupt:
            for emission in translator.finish():
                print(f"\033[2m[{emission.time_s:5.1f}s]\033[0m {emission.text}", flush=True)
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="a wav/flac/mp3 to replay")
    source.add_argument("--mic", action="store_true", help="read the default input device")
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--frame-threshold", type=int, default=16)
    parser.add_argument("--chunk-s", type=float, default=0.5)
    parser.add_argument("--min-audio-s", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-realtime", action="store_true", help="replay as fast as possible")
    parser.add_argument("--compare", action="store_true", help="also show the offline output")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.checkpoint:
        cfg.model.name = args.checkpoint

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # eager, not SDPA: the policy reads attention weights.
    model, processor = build(cfg.model, device, attn_implementation="eager")
    model.eval()

    translator = SimultaneousTranslator(
        model,
        processor,
        AlignAttConfig(
            frame_threshold=args.frame_threshold,
            chunk_s=args.chunk_s,
            min_audio_s=args.min_audio_s,
        ),
    )

    print(f"\033[1mAlignAtt\033[0m f={args.frame_threshold} chunk={args.chunk_s}s "
          f"min-audio={args.min_audio_s}s on {device}\n")

    if args.mic:
        stream_microphone(translator)
        return

    audio = read_wav(args.file)
    stream_file(translator, audio, realtime=not args.no_realtime)

    print(f"\n\033[1mstreaming\033[0m {translator.text.strip()}")
    if args.compare:
        print(f"\033[1moffline  \033[0m {translate_offline(model, processor, audio).strip()}")


if __name__ == "__main__":
    main()
