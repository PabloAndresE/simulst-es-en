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


def summarise(rows: list[dict], seeds: list[int]) -> None:
    """Mean and spread per policy, so no single slice is mistaken for a result."""
    import statistics

    print(f"{'policy':<22}{'BLEU':>16}{'lag':>16}{'revisions':>12}")
    print("-" * 66)
    for label in dict.fromkeys(row["policy"] for row in rows):
        scores = [row["bleu"] for row in rows if row["policy"] == label]
        lags = [
            row["average_lagging_s"]
            for row in rows
            if row["policy"] == label and row["average_lagging_s"] is not None
        ]
        revisions = [row["revision_rate"] for row in rows if row["policy"] == label]
        spread = f"+-{statistics.stdev(scores):.2f}" if len(scores) > 1 else ""
        lag_spread = f"+-{statistics.stdev(lags):.2f}" if len(lags) > 1 else ""
        print(
            f"{label:<22}{statistics.mean(scores):>10.2f} {spread:<5}"
            f"{statistics.mean(lags):>10.2f}s {lag_spread:<5}"
            f"{statistics.mean(revisions):>11.3f}"
        )
    if len(seeds) < 2:
        print("\nsingle seed: no spread, nothing here is a significance claim")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--concat", type=int, default=1, help="join N consecutive clips into one long utterance"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="disjoint slices of the split; several give a spread",
    )
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

    # Every seed is a DISJOINT slice of the split, so the spread across seeds
    # measures variation over material rather than re-measuring one sample.
    # Nothing here is random: the same seeds give the same clips every run.
    full = CoVoST2Dataset(
        cfg.data.cache_dir,
        args.split,
        cfg.data.target_sample_rate,
        limit=args.limit * len(args.seeds),
        concat=args.concat,
    )
    per_seed = len(full) // len(args.seeds)
    slices = {
        seed: full.examples[index * per_seed : (index + 1) * per_seed]
        for index, seed in enumerate(args.seeds)
    }
    mean_s = full.total_hours * 3600 / max(len(full), 1)
    print(
        f"{len(full)} utterances, mean {mean_s:.1f}s, {per_seed} per seed, "
        f"seeds {args.seeds}, chunk {args.chunk_s}s\n"
    )

    def policies(label_only: bool = False):
        """The arms under test, built fresh per seed."""
        yield "offline", None
        for k in args.waitk:
            yield (
                f"wait-k k={k}",
                lambda k=k: WaitKTranslator(
                    model, processor, WaitKConfig(k=k, chunk_s=args.chunk_s)
                ),
            )
        yield (
            "re-translation",
            lambda: RetranslationTranslator(
                model, processor, RetranslationConfig(chunk_s=args.chunk_s)
            ),
        )
        for f in args.alignatt_f:
            yield (
                f"AlignAtt f={f}",
                lambda f=f: SimultaneousTranslator(
                    model, processor, AlignAttConfig(frame_threshold=f, chunk_s=args.chunk_s)
                ),
            )

    rows = []
    for seed, examples in slices.items():
        print(f"seed {seed} ({len(examples)} utterances)")
        for label, factory in policies():
            if factory is None:
                started = time.time()
                hypotheses = [translate_offline(model, processor, ex.audio) for ex in examples]
                row = {
                    "policy": "offline",
                    "bleu": corpus_bleu(hypotheses, [e.translation for e in examples]),
                    "average_lagging_s": sum(e.duration_s for e in examples) / len(examples),
                    "revision_rate": 0.0,
                    "token_revision_rate": 0.0,
                    "clips": len(examples),
                    "seconds": time.time() - started,
                }
            else:
                row = run_policy(factory(), examples, args.chunk_s, label)
            row["seed"] = seed
            rows.append(row)
            lag = row["average_lagging_s"]
            print(
                f"  {label:<20}{row['bleu']:>7.2f}"
                f"{(f'{lag:.2f}s' if lag is not None else 'n/a'):>8}"
                f"  rev {row['revision_rate']:.3f}"
            )
        print()

    summarise(rows, args.seeds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": cfg.model.name,
                "chunk_s": args.chunk_s,
                "concat": args.concat,
                "seeds": args.seeds,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
