"""Run every policy over identical audio and print one table.

AlignAtt against the offline ceiling says what it costs. AlignAtt against
wait-k and re-translation says whether it was worth writing.

    python -m simulst.compare --checkpoint runs/dense-eval/best --limit 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from simulst.config import Config
from simulst.data import CoVoST2Dataset
from simulst.metrics import Emission as MetricEmission
from simulst.metrics import average_lagging, corpus_bleu, revision_rate, token_revision_rate
from simulst.model import build
from simulst.policies import (
    RetranslationConfig,
    RetranslationTranslator,
    WaitKConfig,
    WaitKTranslator,
)
from simulst.simultaneous import AlignAttConfig, SimultaneousTranslator, translate_offline


def run_policy(translator, dataset, chunk_s: float, label: str) -> dict:
    """Replay every clip through one policy and score it."""
    hypotheses, references = [], []
    lags, revisions, token_revisions = [], [], []
    started = time.time()

    for example in dataset:
        translator.reset()
        step = max(1, int(chunk_s * 16000))
        partials, histories = [], []

        for start in range(0, len(example.audio), step):
            translator.push(example.audio[start : start + step])
            partials.append(translator.text)
            histories.append(list(translator.state.token_ids))
        translator.finish()
        partials.append(translator.text)
        histories.append(list(translator.state.token_ids))

        hypotheses.append(translator.text)
        references.append(example.translation)
        revisions.append(revision_rate(partials))
        token_revisions.append(token_revision_rate(histories))

        emissions = translator.state.emissions
        if emissions:
            lag = average_lagging(
                [MetricEmission(e.text, e.time_s) for e in emissions],
                example.duration_s,
                len(emissions),
            )
            if lag is not None:
                lags.append(lag)

    return {
        "policy": label,
        "bleu": corpus_bleu(hypotheses, references),
        "average_lagging_s": sum(lags) / len(lags) if lags else None,
        "revision_rate": sum(revisions) / len(revisions),
        "token_revision_rate": sum(token_revisions) / len(token_revisions),
        "clips": len(hypotheses),
        "seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--chunk-s", type=float, default=1.0)
    parser.add_argument("--alignatt-f", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--waitk", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results/policy_comparison.json")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.checkpoint:
        cfg.model.name = args.checkpoint
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = build(cfg.model, device, attn_implementation="eager")
    model.eval()

    dataset = CoVoST2Dataset(
        cfg.data.cache_dir, args.split, cfg.data.target_sample_rate, limit=args.limit
    )
    print(f"{len(dataset)} clips, {dataset.total_hours * 60:.1f} min, chunk {args.chunk_s}s\n")

    started = time.time()
    offline = [translate_offline(model, processor, ex.audio) for ex in dataset]
    references = [ex.translation for ex in dataset]
    rows = [{
        "policy": "offline",
        "bleu": corpus_bleu(offline, references),
        "average_lagging_s": sum(ex.duration_s for ex in dataset) / len(dataset),
        "revision_rate": 0.0,
        "token_revision_rate": 0.0,
        "clips": len(dataset),
        "seconds": time.time() - started,
    }]

    header = f"{'policy':<22}{'BLEU':>7}{'lag':>8}{'revisions':>12}{'tokens':>9}"
    print(header)
    print("-" * len(header))

    def show(row: dict) -> None:
        lag = row["average_lagging_s"]
        print(
            f"{row['policy']:<22}{row['bleu']:>7.2f}"
            f"{(f'{lag:.2f}s' if lag is not None else 'n/a'):>8}"
            f"{row['revision_rate']:>12.3f}{row['token_revision_rate']:>9.3f}"
        )

    show(rows[0])

    for k in args.waitk:
        translator = WaitKTranslator(model, processor, WaitKConfig(k=k, chunk_s=args.chunk_s))
        row = run_policy(translator, dataset, args.chunk_s, f"wait-k k={k}")
        rows.append(row)
        show(row)

    translator = RetranslationTranslator(
        model, processor, RetranslationConfig(chunk_s=args.chunk_s)
    )
    row = run_policy(translator, dataset, args.chunk_s, "re-translation")
    rows.append(row)
    show(row)

    for f in args.alignatt_f:
        translator = SimultaneousTranslator(
            model, processor, AlignAttConfig(frame_threshold=f, chunk_s=args.chunk_s)
        )
        row = run_policy(translator, dataset, args.chunk_s, f"AlignAtt f={f}")
        rows.append(row)
        show(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": cfg.model.name, "chunk_s": args.chunk_s,
                               "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
