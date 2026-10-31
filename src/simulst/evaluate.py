"""Score a checkpoint, or the zero-shot model, on a split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from simulst.config import Config
from simulst.data import Collator, CoVoST2Dataset
from simulst.model import build
from simulst.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--checkpoint", default=None, help="omit to score the base model")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="required for --split test; a test read is one-shot and logged",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="runs/eval")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_final:
        raise SystemExit(
            "refusing to read the test split without --confirm-final.\n"
            "Tune on validation; touch test once, when the model is frozen."
        )

    cfg = Config.from_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg.model
    if args.checkpoint:
        model_cfg = cfg.with_overrides(model={"name": args.checkpoint}).model
    model, processor = build(model_cfg, device)

    dataset = CoVoST2Dataset(
        cfg.data.cache_dir,
        args.split,
        cfg.data.target_sample_rate,
        limit=args.limit or cfg.data.max_eval_examples,
        max_duration_s=cfg.data.max_duration_s,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        collate_fn=Collator(processor, cfg.data.max_label_tokens),
        num_workers=cfg.data.num_workers,
    )

    results = evaluate(model, processor, loader, device)
    results.update(
        split=args.split,
        checkpoint=args.checkpoint or f"{cfg.model.name} (zero-shot)",
        hours=round(dataset.total_hours, 2),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "zero_shot" if not args.checkpoint else "finetuned"
    (out_dir / f"metrics_{name}_{args.split}.json").write_text(json.dumps(results, indent=2))

    print(f"{results['checkpoint']}  split={args.split}")
    print(f"  BLEU {results['bleu']:.2f}   loss {results['loss']:.4f}   n={results['examples']}")


if __name__ == "__main__":
    main()
