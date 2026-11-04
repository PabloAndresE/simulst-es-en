"""Sweep the policy's dial and plot quality against latency.

A simultaneous system has no single score. It has a curve: every point trades
BLEU for delay, and the only honest summary is the whole curve plus the offline
translation as the ceiling it is paying against.

    python -m simulst.stream_eval --config configs/server.yaml \
        --thresholds 2 4 6 10 16 --limit 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from simulst.config import Config
from simulst.data import CoVoST2Dataset
from simulst.metrics import Emission as MetricEmission
from simulst.metrics import average_lagging, corpus_bleu, revision_rate, token_revision_rate
from simulst.model import build
from simulst.simultaneous import AlignAttConfig, SimultaneousTranslator, translate_offline


def evaluate_threshold(
    model,
    processor,
    dataset,
    threshold: int,
    chunk_s: float,
    min_audio_s: float,
) -> dict:
    """Run every clip through the policy at one setting of the dial."""
    config = AlignAttConfig(frame_threshold=threshold, chunk_s=chunk_s, min_audio_s=min_audio_s)
    translator = SimultaneousTranslator(model, processor, config)

    hypotheses, references, lags, revisions, token_revisions = [], [], [], [], []
    started = time.time()

    for example in dataset:
        translator.reset()
        step = max(1, int(chunk_s * 16000))
        partials, token_history = [], []
        for start in range(0, len(example.audio), step):
            translator.push(example.audio[start : start + step])
            partials.append(translator.text)
            token_history.append(list(translator.state.token_ids))
        translator.finish()
        partials.append(translator.text)
        token_history.append(list(translator.state.token_ids))

        hypotheses.append(translator.text)
        references.append(example.translation)
        revisions.append(revision_rate(partials))
        token_revisions.append(token_revision_rate(token_history))

        tokens = len(translator.state.emissions)
        if tokens:
            lag = average_lagging(
                [MetricEmission(e.text, e.time_s) for e in translator.state.emissions],
                example.duration_s,
                tokens,
            )
            if lag is not None:
                lags.append(lag)

    return {
        "frame_threshold": threshold,
        "chunk_s": chunk_s,
        "min_audio_s": min_audio_s,
        "bleu": corpus_bleu(hypotheses, references),
        "average_lagging_s": sum(lags) / len(lags) if lags else None,
        "revision_rate": sum(revisions) / len(revisions) if revisions else 0.0,
        "token_revision_rate": (
            sum(token_revisions) / len(token_revisions) if token_revisions else 0.0
        ),
        "clips": len(hypotheses),
        "seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--checkpoint", default=None, help="fine-tuned weights")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[2, 4, 6, 10, 16])
    parser.add_argument("--chunk-s", type=float, default=0.5)
    parser.add_argument("--min-audio-s", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results/stream_sweep.json")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.checkpoint:
        cfg.model.name = args.checkpoint

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    # eager, not SDPA: the policy reads attention weights.
    model, processor = build(cfg.model, device, attn_implementation="eager")
    model.eval()

    dataset = CoVoST2Dataset(
        cfg.data.cache_dir, args.split, cfg.data.target_sample_rate, limit=args.limit
    )
    print(f"{len(dataset)} clips, {dataset.total_hours * 60:.1f} min of audio")

    # The ceiling: wait for everything, then translate.
    started = time.time()
    offline = [translate_offline(model, processor, ex.audio) for ex in dataset]
    references = [ex.translation for ex in dataset]
    ceiling = {
        "policy": "offline",
        "bleu": corpus_bleu(offline, references),
        "average_lagging_s": sum(ex.duration_s for ex in dataset) / len(dataset),
        "revision_rate": 0.0,
        "clips": len(dataset),
        "seconds": time.time() - started,
    }
    print(
        f"offline              BLEU {ceiling['bleu']:6.2f}  "
        f"lag {ceiling['average_lagging_s']:5.2f}s (whole utterance)"
    )

    rows = [ceiling]
    for threshold in args.thresholds:
        row = evaluate_threshold(
            model, processor, dataset, threshold, args.chunk_s, args.min_audio_s
        )
        row["policy"] = "alignatt"
        lag = row["average_lagging_s"]
        print(
            f"alignatt f={threshold:<3d}        BLEU {row['bleu']:6.2f}  "
            f"lag {lag:5.2f}s  rev {row['revision_rate']:.3f} "
            f"(tokens {row['token_revision_rate']:.3f})  "
            f"({row['seconds'] / 60:.1f} min)"
            if lag is not None
            else f"alignatt f={threshold:<3d}        BLEU {row['bleu']:6.2f}  lag n/a"
        )
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": cfg.model.name, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
