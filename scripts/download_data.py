"""Materialise CoVoST 2 es->en locally.

Downloads rather than streams: shards are 82 MB and HTTP streaming stalls on a
slow link (measured 2.1 MB/s, so the full 4.53 GB takes about 36 minutes).

    python scripts/download_data.py --out data
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fixie-ai/covost2")
    parser.add_argument("--config", default="es_en")
    parser.add_argument("--out", default="data")
    parser.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    patterns = [f"{args.config}/{split}-*.parquet" for split in args.splits]
    started = time.time()
    path = snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=args.out,
    )

    shards = sorted(Path(path).rglob("*.parquet"))
    total = sum(p.stat().st_size for p in shards)
    print(f"{len(shards)} shards, {total / 1e9:.2f} GB in {time.time() - started:.0f}s")
    print(f"cached at {path}")


if __name__ == "__main__":
    main()
