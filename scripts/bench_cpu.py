#!/usr/bin/env python
"""What the policy costs per chunk on CPU, against the chunk budget.

The README quotes latency in seconds of audio. That figure only holds while the
machine keeps up: if a chunk takes longer to process than it takes to arrive,
the policy falls behind the speaker and every latency number is optimistic by a
growing amount. This measures the margin.

    python scripts/bench_cpu.py --chunk-s 1.0 0.5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from simulst.config import Config
from simulst.live import read_wav
from simulst.model import build
from simulst.simultaneous import AlignAttConfig, SimultaneousTranslator


def measure(model, processor, audio, chunk_s: float, frame_threshold: int) -> dict:
    translator = SimultaneousTranslator(
        model, processor,
        AlignAttConfig(frame_threshold=frame_threshold, chunk_s=chunk_s),
    )
    step = int(chunk_s * 16000)

    # The first call compiles kernels and is not representative of the rest.
    translator.push(audio[:step])
    translator.reset()

    times = []
    for start in range(0, len(audio), step):
        began = time.perf_counter()
        translator.push(audio[start : start + step])
        times.append(time.perf_counter() - began)

    worst = max(times)
    return {
        "chunk_s": chunk_s,
        "median_ms": statistics.median(times) * 1000,
        "p90_ms": sorted(times)[int(len(times) * 0.9)] * 1000,
        "max_ms": worst * 1000,
        "budget_ms": chunk_s * 1000,
        "headroom_ms": (chunk_s - worst) * 1000,
        "realtime": worst < chunk_s,
        "chunks": len(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--file", default="docs/assets/sample_es.wav")
    parser.add_argument("--chunk-s", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--frame-threshold", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/cpu_latency.json")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.checkpoint:
        cfg.model.name = args.checkpoint
    model, processor = build(cfg.model, args.device, attn_implementation="eager")
    model.eval()
    audio = read_wav(args.file)

    print(f"{cfg.model.name} on {args.device}, {len(audio) / 16000:.1f}s of audio\n")
    print(f"{'chunk':>7}{'median':>10}{'p90':>9}{'max':>9}{'budget':>9}{'headroom':>11}  verdict")
    rows = []
    for chunk_s in args.chunk_s:
        row = measure(model, processor, audio, chunk_s, args.frame_threshold)
        rows.append(row)
        print(
            f"{row['chunk_s']:>6.1f}s{row['median_ms']:>9.0f}ms{row['p90_ms']:>7.0f}ms"
            f"{row['max_ms']:>7.0f}ms{row['budget_ms']:>7.0f}ms{row['headroom_ms']:>9.0f}ms"
            f"  {'keeps up' if row['realtime'] else 'FALLS BEHIND'}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": cfg.model.name, "device": args.device,
                               "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
