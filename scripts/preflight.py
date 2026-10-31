"""Check everything a training run needs before queuing GPU time.

Every failure here is one that would otherwise appear minutes into a job: a
missing dataset, a model that cannot be fetched on an offline compute node, a
batch size that will not fit, or audio at the wrong sample rate.

    python scripts/preflight.py --config configs/server.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from simulst.audio import decode
from simulst.config import Config
from simulst.data import Collator, CoVoST2Dataset, shard_paths

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe-examples", type=int, default=64)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"preflight: config={args.config} device={device}\n")

    print("environment")
    check("torch", True, torch.__version__)
    if device == "cuda":
        check("cuda", torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            check("gpu", True, f"{props.name}, {props.total_memory / 1e9:.0f} GB")
    else:
        print("  [--  ] cuda - absent; dry run only, real training needs a GPU")

    print("\ndata")
    root = Path(cfg.data.cache_dir)
    if not check("dataset cached", root.is_dir(), str(root.resolve())):
        print("\nhint: python scripts/download_data.py --out data")
        return 1
    for split in ("train", "validation"):
        try:
            paths = shard_paths(root, split)
            check(f"{split} shards", True, f"{len(paths)} files")
        except FileNotFoundError as exc:
            check(f"{split} shards", False, str(exc))
            return 1

    # The dataset card claims 16 kHz; the files are 48 kHz. Catch that here
    # rather than after an epoch of silently mis-scaled mel features.
    import pyarrow.parquet as pq

    table = pq.read_table(shard_paths(root, "validation")[0], columns=["audio"])
    blob = table.to_pylist()[0]["audio"]
    _, native_rate = decode(blob["bytes"] if isinstance(blob, dict) else blob)
    check(
        "native sample rate",
        True,
        f"{native_rate} Hz -> resampled to {cfg.data.target_sample_rate} Hz"
        + (" (resampling required)" if native_rate != cfg.data.target_sample_rate else ""),
    )

    probe = CoVoST2Dataset(
        root, "validation", cfg.data.target_sample_rate, limit=args.probe_examples
    )
    durations = [e.duration_s for e in probe]
    check(
        "decode + resample",
        True,
        f"{len(probe)} examples, mean {statistics.mean(durations):.2f}s, "
        f"p90 {np.percentile(durations, 90):.2f}s",
    )
    check("references non-empty", all(e.translation.strip() for e in probe))

    print("\nmodel")
    try:
        from simulst.model import build, param_counts

        model, processor = build(cfg.model, device)
        check("weights load", True, str(param_counts(model)))
    except Exception as exc:  # noqa: BLE001 - the message is the point
        check("weights load", False, f"{type(exc).__name__}: {exc}")
        print("\nhint: pre-download the model on a login node, then export HF_HOME.")
        return 1

    forced = model.generation_config.forced_decoder_ids
    check("translate task forced", bool(forced), f"{len(forced or [])} decoder prompt ids")

    print("\nthroughput")
    collator = Collator(processor, cfg.data.max_label_tokens)
    loader = torch.utils.data.DataLoader(
        probe, batch_size=cfg.data.batch_size, collate_fn=collator, num_workers=0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device == "cuda")

    try:
        model.train()
        timings = []
        for step, batch in enumerate(loader):
            if step >= 5:
                break
            started = time.perf_counter()
            optimizer.zero_grad()
            with torch.autocast("cuda", enabled=cfg.train.amp and device == "cuda"):
                loss = model(
                    input_features=batch["input_features"].to(device),
                    labels=batch["labels"].to(device),
                ).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device == "cuda":
                torch.cuda.synchronize()
            if step >= 1:  # skip the warm-up step
                timings.append(time.perf_counter() - started)

        per_step = statistics.mean(timings) if timings else float("nan")
        train_examples = cfg.data.max_train_examples or 79015
        steps = -(-train_examples // cfg.data.batch_size)
        hours = per_step * steps * cfg.train.epochs / 3600
        check(
            f"train step at batch {cfg.data.batch_size}",
            True,
            f"{per_step * 1000:.0f} ms/step, {steps} steps/epoch "
            f"-> ~{hours:.1f} h for {cfg.train.epochs} epochs",
        )
        if device == "cuda":
            check("peak memory", True, f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    except torch.cuda.OutOfMemoryError:
        check("train step", False, f"OOM at batch_size {cfg.data.batch_size}; lower it")
        return 1

    print()
    if FAILURES:
        print(f"preflight FAILED: {', '.join(FAILURES)}")
        return 1
    print("preflight passed - safe to start training")
    print(
        "note: the epoch estimate excludes evaluation and assumes the data is\n"
        "      already decoded in memory. Budget wall-clock with headroom."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
