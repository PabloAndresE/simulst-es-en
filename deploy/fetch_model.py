#!/usr/bin/env python
"""Resolve the model at build time so the container never needs the network.

Accepts a hub id or a local directory. Writes to /opt/model, which the image's
entrypoint passes as --checkpoint.
"""

import shutil
import sys
from pathlib import Path

from transformers import WhisperForConditionalGeneration, WhisperProcessor

TARGET = Path("/opt/model")


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "openai/whisper-small"
    TARGET.mkdir(parents=True, exist_ok=True)

    if Path(source).is_dir():
        for item in Path(source).iterdir():
            shutil.copy2(item, TARGET / item.name)
        print(f"copied {source} -> {TARGET}")
        return

    processor = WhisperProcessor.from_pretrained(source, language="spanish", task="translate")
    model = WhisperForConditionalGeneration.from_pretrained(source)
    processor.save_pretrained(TARGET)
    model.save_pretrained(TARGET)
    print(f"fetched {source} -> {TARGET}")


if __name__ == "__main__":
    main()
