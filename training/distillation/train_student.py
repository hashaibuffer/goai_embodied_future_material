#!/usr/bin/env python3
"""Train a learned or proprio-only 441->16 TE student."""

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

from .te_dataset import (
    ShardSpec,
    accepted_training_specs,
    compute_input_statistics,
    evaluation_specs,
    load_shards,
    qc_excluded_paths,
    sha256_file,
)
from .te_model import (
    StudentConfig, StudentPolicy, load_checkpoint, parameter_count, save_checkpoint)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "results" / "datasets" / "model_44196_raw_v2"
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "QC_50K_MANIFEST.json"
DEFAULT_QC_MANIFEST = DEFAULT_DATASET_ROOT / "QC_MANIFEST.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def terrain_weights(terrains: np.ndarray) -> torch.Tensor:
    counts = Counter(map(str, terrains.tolist()))
    scale = len(terrains) / len(counts)
    return torch.as_tensor(
        [scale / counts[str(terrain)] for terrain in terrains], dtype=torch.double)


def validation_loss(
    model: StudentPolicy,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_total = absolute_total = 0.0
    value_count = sample_count = 0
    with torch.no_grad():
        for observations, actions in loader:
            observations = observations.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            prediction = model(observations)
            loss_total += float(loss_fn(prediction, actions)) * len(observations)
            absolute_total += float(torch.sum(torch.abs(prediction - actions)))
            value_count += prediction.numel()
            sample_count += len(observations)
    return loss_total / sample_count, absolute_total / value_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qc-manifest", type=Path, default=DEFAULT_QC_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--validation-groups", nargs="+", default=["VALIDATION", "VALIDATION_SEGMENTS"])
    parser.add_argument("--controller", choices=("learned", "proprio_clone"), required=True)
    parser.add_argument("--architecture", choices=("branch", "flat"), default="branch")
    parser.add_argument(
        "--flat-hidden", nargs="+", type=int, default=[256, 128],
        help="Hidden widths for the flat learned architecture")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--extra-shards", nargs="*", type=Path, default=[],
        help="Strict schema-v2 DAgger shards appended to the manifest training set")
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="Fine-tune from a student checkpoint while preserving its input statistics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="0 selects 4096 on CUDA or 2048 on CPU")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.2)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Keep 0 on Windows: workers duplicate the in-memory 50k dataset")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cpu-threads", type=int, default=0,
        help="0 uses all but two logical processors")
    parser.add_argument(
        "--gpu-memory-fraction", type=float, default=0.85,
        help="Maximum fraction of one CUDA device available to this process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size < 0:
        raise SystemExit("epochs/patience must be positive and batch-size cannot be negative")
    if not args.flat_hidden or any(width <= 0 for width in args.flat_hidden):
        raise SystemExit("--flat-hidden widths must be positive")
    if not 0.1 <= args.gpu_memory_fraction <= 0.95:
        raise SystemExit("gpu-memory-fraction must be in [0.1, 0.95]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = choose_device(args.device)
    logical_cpus = os.cpu_count() or 1
    cpu_threads = args.cpu_threads or max(1, logical_cpus - 2)
    batch_size = args.batch_size or (4096 if device.type == "cuda" else 2048)
    torch.set_num_threads(cpu_threads)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        cuda_device_index = (
            device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_memory_fraction, device=cuda_device_index)
        # Keep FP32 model/optimizer state while accelerating matrix multiplies
        # on Ampere-or-newer GPUs. Final ONNX parity is still checked in FP32.
        torch.backends.cuda.matmul.allow_tf32 = True

    train_specs, teacher_sha256 = accepted_training_specs(args.manifest)
    train_specs.extend(ShardSpec(path=path) for path in args.extra_shards)
    excluded = qc_excluded_paths(args.qc_manifest)
    validation_specs = evaluation_specs(
        args.dataset_root,
        groups=args.validation_groups,
        excluded_paths=excluded,
    )
    train = load_shards(train_specs, expected_teacher_sha256=teacher_sha256)
    validation = load_shards(validation_specs, expected_teacher_sha256=teacher_sha256)
    config = StudentConfig(
        controller=args.controller,
        architecture=args.architecture,
        flat_hidden=tuple(args.flat_hidden),
    )
    if args.init_checkpoint is not None:
        model, statistics, _ = load_checkpoint(
            args.init_checkpoint, map_location="cpu")
        if model.config != config:
            raise ValueError(
                f"initial checkpoint config {model.config} does not match requested {config}")
        model = model.to(device)
    else:
        statistics = compute_input_statistics(train.observations)
        model = StudentPolicy(config, statistics).to(device)
    loss_fn = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.05)

    train_dataset = TensorDataset(
        torch.from_numpy(train.observations), torch.from_numpy(train.actions))
    validation_dataset = TensorDataset(
        torch.from_numpy(validation.observations), torch.from_numpy(validation.actions))
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        terrain_weights(train.terrains), len(train), replacement=True, generator=generator)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    training_metadata = {
        "controller": args.controller,
        "architecture": args.architecture,
        "seed": args.seed,
        "teacher_onnx_sha256": teacher_sha256,
        "training_manifest": str(args.manifest.resolve()),
        "training_manifest_sha256": sha256_file(args.manifest),
        "extra_shards": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.extra_shards
        ],
        "init_checkpoint": (
            str(args.init_checkpoint.resolve()) if args.init_checkpoint else None),
        "init_checkpoint_sha256": (
            sha256_file(args.init_checkpoint) if args.init_checkpoint else None),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_terrain_samples": dict(sorted(Counter(map(str, train.terrains)).items())),
        "validation_terrain_samples": dict(
            sorted(Counter(map(str, validation.terrains)).items())),
        "validation_excluded_paths": sorted(excluded),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "smooth_l1_beta": args.smooth_l1_beta,
        "batch_size": batch_size,
        "parameter_count": parameter_count(model),
        "device": str(device),
        "cpu_threads": cpu_threads,
        "logical_cpus": logical_cpus,
        "gpu_memory_fraction": args.gpu_memory_fraction if device.type == "cuda" else None,
    }

    checkpoint = args.output_dir / "best.pt"
    history = []
    best_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.monotonic()
    if args.init_checkpoint is not None:
        initial_smooth_l1, initial_mae = validation_loss(
            model, validation_loader, loss_fn, device)
        best_mae = initial_mae
        initial_record = {
            "epoch": 0,
            "train_smooth_l1": None,
            "validation_smooth_l1": initial_smooth_l1,
            "validation_mae": initial_mae,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": time.monotonic() - started,
        }
        history.append(initial_record)
        print(json.dumps(initial_record, sort_keys=True), flush=True)
        save_checkpoint(
            checkpoint,
            model=model,
            statistics=statistics,
            training_metadata={
                **training_metadata,
                "best_epoch": best_epoch,
                "best_validation_mae": best_mae,
            },
        )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        for observations, actions in train_loader:
            observations = observations.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(observations)
            loss = loss_fn(prediction, actions)
            if not torch.isfinite(loss):
                raise RuntimeError("training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            train_loss_total += float(loss.detach()) * len(observations)
            train_samples += len(observations)
        validation_smooth_l1, validation_mae = validation_loss(
            model, validation_loader, loss_fn, device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "train_smooth_l1": train_loss_total / train_samples,
            "validation_smooth_l1": validation_smooth_l1,
            "validation_mae": validation_mae,
            "learning_rate": learning_rate,
            "elapsed_s": time.monotonic() - started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if validation_mae < best_mae - 1e-7:
            best_mae = validation_mae
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoint,
                model=model,
                statistics=statistics,
                training_metadata={
                    **training_metadata,
                    "best_epoch": best_epoch,
                    "best_validation_mae": best_mae,
                },
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= args.patience:
            break

    last_checkpoint = args.output_dir / "last.pt"
    save_checkpoint(
        last_checkpoint,
        model=model,
        statistics=statistics,
        training_metadata={
            **training_metadata,
            "final_epoch": int(history[-1]["epoch"]),
            "final_validation_mae": float(history[-1]["validation_mae"]),
            "selected_as_best": int(history[-1]["epoch"]) == best_epoch,
        },
    )

    summary = {
        **training_metadata,
        "best_checkpoint": str(checkpoint.resolve()),
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "last_checkpoint": str(last_checkpoint.resolve()),
        "epochs_completed": len(history),
        "elapsed_s": time.monotonic() - started,
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
