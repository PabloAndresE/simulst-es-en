"""Fine-tune Whisper for Spanish-to-English speech translation.

The zero-shot model already performs this task, so training starts from a real
baseline. `evaluate.py --zero-shot` records it before any weights move; a
fine-tune that does not beat it is a result worth reporting, not a bug to hide.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from simulst.config import Config
from simulst.data import Collator, CoVoST2Dataset
from simulst.metrics import corpus_bleu
from simulst.model import build, param_counts, translate


def evaluate(model, processor, loader, device, max_batches: int | None = None):
    model.eval()
    hypotheses, references, losses = [], [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            features = batch["input_features"].to(device)
            losses.append(
                model(input_features=features, labels=batch["labels"].to(device)).loss.item()
            )
            hypotheses.extend(translate(model, processor, features))
            references.extend(batch["references"])
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "bleu": corpus_bleu(hypotheses, references),
        "examples": len(hypotheses),
    }


def run(cfg: Config, device: str, quiet: bool = False) -> dict:
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    torch.manual_seed(cfg.data.seed)

    model, processor = build(cfg.model, device)
    collator = Collator(processor, cfg.data.max_label_tokens)
    if not quiet:
        print(f"model={cfg.model.name} {param_counts(model)}")

    loaders = {}
    for split, limit in (
        ("train", cfg.data.max_train_examples),
        ("validation", cfg.data.max_eval_examples),
    ):
        dataset = CoVoST2Dataset(
            cfg.data.cache_dir,
            split,
            cfg.data.target_sample_rate,
            limit=limit,
            max_duration_s=cfg.data.max_duration_s,
        )
        if not quiet:
            print(f"{split}: {len(dataset)} examples, {dataset.total_hours:.1f} h")
        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=split == "train",
            collate_fn=collator,
            num_workers=cfg.data.num_workers,
            pin_memory=device == "cuda",
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    steps_per_epoch = math.ceil(len(loaders["train"]) / cfg.train.grad_accum)
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.train.lr,
        total_steps=max(total_steps, 1),
        pct_start=min(cfg.train.warmup_steps / max(total_steps, 1), 0.3),
    )
    use_amp = cfg.train.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_bleu = [], -1.0
    batches = len(loaders["train"])
    eval_interval = max(1, batches // max(cfg.train.evals_per_epoch, 1))

    def checkpoint(epoch: int, fraction: float, train_loss: float, started: float) -> None:
        nonlocal best_bleu
        metrics = evaluate(model, processor, loaders["validation"], device)
        metrics.update(
            epoch=epoch,
            progress=round(fraction, 3),
            train_loss=train_loss,
            seconds=time.time() - started,
        )
        history.append(metrics)
        if not quiet:
            print(
                f"epoch {epoch} @{fraction:.2f} | train {train_loss:.4f} "
                f"| val {metrics['loss']:.4f} | BLEU {metrics['bleu']:.2f} "
                f"| {metrics['seconds'] / 60:.1f} min"
            )
        if metrics["bleu"] > best_bleu:
            best_bleu = metrics["bleu"]
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
        model.train()

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running, started = 0.0, time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(loaders["train"], start=1):
            with torch.autocast("cuda", enabled=use_amp):
                loss = model(
                    input_features=batch["input_features"].to(device),
                    labels=batch["labels"].to(device),
                ).loss / cfg.train.grad_accum
            scaler.scale(loss).backward()

            if step % cfg.train.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running += loss.item() * cfg.train.grad_accum
            if not quiet and step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{batches} loss {running / step:.4f}")

            if step % eval_interval == 0 and step < batches:
                checkpoint(epoch, epoch - 1 + step / batches, running / step, started)

        checkpoint(epoch, float(epoch), running / batches, started)

    (out_dir / "history.json").write_text(
        json.dumps({"best_bleu": best_bleu, "history": history}, indent=2)
    )
    return {"best_bleu": best_bleu, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    data_over = {
        k: v
        for k, v in (
            ("batch_size", args.batch_size),
            ("max_train_examples", args.max_train_examples),
        )
        if v is not None
    }
    train_over = {
        k: v for k, v in (("epochs", args.epochs), ("out_dir", args.out_dir)) if v is not None
    }
    cfg = cfg.with_overrides(data=data_over, train=train_over)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    summary = run(cfg, device)
    print(f"best validation BLEU {summary['best_bleu']:.2f} -> {cfg.train.out_dir}/best")


if __name__ == "__main__":
    main()
