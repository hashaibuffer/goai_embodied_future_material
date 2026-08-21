#!/usr/bin/env python3
"""Train a hard terrain gate between a proven policy and a recovery policy."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .te_dataset import ShardSpec, load_shards, sha256_file
from .te_model import StudentConfig, StudentPolicy, load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--positive-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--negative-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-hidden", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-margin", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.85)
    return parser.parse_args()


def evaluate_logits(
    model: StudentPolicy,
    observations: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = torch.from_numpy(observations[start:start + batch_size]).to(device)
            outputs.append(model.gate_logits(batch).cpu().numpy().reshape(-1))
    logits = np.concatenate(outputs)
    positive = labels == 1.0
    negative = ~positive
    errors = int(np.count_nonzero((logits > 0.0) != positive))
    positive_min = float(logits[positive].min())
    negative_max = float(logits[negative].max())
    return {
        "samples": len(labels),
        "errors": errors,
        "accuracy": float(1.0 - errors / len(labels)),
        "positive_min_logit": positive_min,
        "negative_max_logit": negative_max,
        "margin": float(min(positive_min, -negative_max)),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise SystemExit("epochs, batch-size, and learning-rate must be positive")
    if not args.gate_hidden or any(width <= 0 for width in args.gate_hidden):
        raise SystemExit("gate hidden widths must be positive")
    if args.target_margin < 0:
        raise SystemExit("target-margin cannot be negative")
    if not 0.1 <= args.gpu_memory_fraction <= 0.95:
        raise SystemExit("gpu-memory-fraction must be in [0.1, 0.95]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    logical_cpus = os.cpu_count() or 1
    torch.set_num_threads(args.cpu_threads or max(1, logical_cpus - 2))
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True

    primary, statistics, primary_metadata = load_checkpoint(
        args.primary_checkpoint, map_location="cpu")
    recovery, recovery_statistics, recovery_metadata = load_checkpoint(
        args.recovery_checkpoint, map_location="cpu")
    if (primary.config.controller != "learned"
            or primary.config.architecture != "flat"
            or recovery.config != primary.config):
        raise ValueError("gate inputs must be matching learned flat checkpoints")
    if (not np.array_equal(statistics.mean, recovery_statistics.mean)
            or not np.array_equal(statistics.std, recovery_statistics.std)
            or not np.array_equal(
                statistics.normalized_mask, recovery_statistics.normalized_mask)):
        raise ValueError("primary and recovery checkpoints use different input statistics")
    teacher_sha256 = str(primary_metadata["teacher_onnx_sha256"])
    if recovery_metadata.get("teacher_onnx_sha256") != teacher_sha256:
        raise ValueError("primary and recovery checkpoints use different teachers")

    positive = load_shards(
        [ShardSpec(path=path) for path in args.positive_shards],
        expected_teacher_sha256=teacher_sha256)
    negative = load_shards(
        [ShardSpec(path=path) for path in args.negative_shards],
        expected_teacher_sha256=teacher_sha256)
    observations = np.concatenate((positive.observations, negative.observations))
    labels = np.concatenate((
        np.ones(len(positive), dtype=np.float32),
        np.zeros(len(negative), dtype=np.float32),
    ))

    config = StudentConfig(
        controller="learned",
        architecture="gated_ensemble",
        flat_hidden=primary.config.flat_hidden,
        gate_hidden=tuple(args.gate_hidden),
        activation=primary.config.activation,
        normalization_clip=primary.config.normalization_clip,
    )
    model = StudentPolicy(config, statistics).to(device)
    model.primary_policy.load_state_dict(primary.flat_policy.state_dict())
    model.recovery_policy.load_state_dict(recovery.flat_policy.state_dict())
    model.primary_policy.requires_grad_(False)
    model.recovery_policy.requires_grad_(False)

    class_counts = Counter(map(float, labels.tolist()))
    sample_weights = torch.as_tensor(
        [len(labels) / (2 * class_counts[float(label)]) for label in labels],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(observations), torch.from_numpy(labels[:, None])),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    started = time.monotonic()
    best_state = None
    best_score = None
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch_observations, batch_labels in loader:
            batch_observations = batch_observations.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model.gate_logits(batch_observations), batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("gate training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gate.parameters(), max_norm=10.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch_observations)
        metrics = evaluate_logits(
            model, observations, labels, device, args.batch_size)
        record = {
            "epoch": epoch,
            "loss": total / len(labels),
            **metrics,
            "elapsed_s": time.monotonic() - started,
        }
        history.append(record)
        score = (metrics["errors"], -metrics["margin"])
        if best_score is None or score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0 or metrics["errors"] == 0:
            print(json.dumps(record, sort_keys=True), flush=True)
        if metrics["errors"] == 0 and metrics["margin"] >= args.target_margin:
            break

    model.load_state_dict(best_state)
    final_metrics = evaluate_logits(
        model, observations, labels, device, args.batch_size)
    metadata = {
        "controller": "learned",
        "architecture": "gated_ensemble",
        "teacher_onnx_sha256": teacher_sha256,
        "primary_checkpoint": str(args.primary_checkpoint.resolve()),
        "primary_checkpoint_sha256": sha256_file(args.primary_checkpoint),
        "recovery_checkpoint": str(args.recovery_checkpoint.resolve()),
        "recovery_checkpoint_sha256": sha256_file(args.recovery_checkpoint),
        "positive_shards": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.positive_shards],
        "negative_shards": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.negative_shards],
        "positive_samples": len(positive),
        "negative_samples": len(negative),
        "best_epoch": best_epoch,
        "gate_metrics": final_metrics,
        "gate_hidden": args.gate_hidden,
        "seed": args.seed,
    }
    save_checkpoint(
        args.output, model=model, statistics=statistics, training_metadata=metadata)
    summary = {**metadata, "history": history, "output": str(args.output.resolve())}
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
