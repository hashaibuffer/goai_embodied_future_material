#!/usr/bin/env python3
"""Evaluate a TE checkpoint by terrain and test whether LiDAR affects actions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from .te_dataset import (
    accepted_training_specs,
    evaluation_specs,
    load_shards,
    qc_excluded_paths,
)
from .te_metrics import (
    metrics_by_terrain,
    predict_numpy,
    regression_metrics,
    terrain_ablation_metrics,
)
from .te_model import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "results" / "datasets" / "model_44196_raw_v2"
MANIFEST = DATASET_ROOT / "QC_50K_MANIFEST.json"
QC_MANIFEST = DATASET_ROOT / "QC_MANIFEST.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--qc-manifest", type=Path, default=QC_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--validation-groups", nargs="+", default=["VALIDATION", "VALIDATION_SEGMENTS"])
    parser.add_argument("--stress-groups", nargs="+", default=["ACCEPTED"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--require-gates", action="store_true")
    return parser.parse_args()


def torch_infer(model, device: torch.device):
    def infer(observations: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            values = model(torch.from_numpy(observations).to(device))
        return values.detach().cpu().numpy()
    return infer


def evaluate_dataset(model, data, device, batch_size: int) -> tuple[dict, np.ndarray]:
    infer = torch_infer(model, device)
    prediction = predict_numpy(data.observations, infer, batch_size=batch_size)
    return {
        "overall": regression_metrics(data.actions, prediction),
        "by_terrain": metrics_by_terrain(data.actions, prediction, data.terrains),
    }, prediction


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")
    _, teacher_sha256 = accepted_training_specs(args.manifest)
    excluded = qc_excluded_paths(args.qc_manifest)
    validation_specs = evaluation_specs(
        args.dataset_root,
        groups=args.validation_groups,
        excluded_paths=excluded,
    )
    stress_specs = evaluation_specs(
        args.dataset_root,
        groups=args.stress_groups,
        excluded_paths=excluded,
    )
    validation = load_shards(
        validation_specs, expected_teacher_sha256=teacher_sha256)
    stress = load_shards(stress_specs, expected_teacher_sha256=teacher_sha256)

    device = torch.device(args.device)
    model, _, training_metadata = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()
    validation_report, validation_prediction = evaluate_dataset(
        model, validation, device, args.batch_size)
    stress_report, _ = evaluate_dataset(model, stress, device, args.batch_size)
    infer = torch_infer(model, device)
    ablation = terrain_ablation_metrics(
        validation.observations,
        validation.actions,
        validation_prediction,
        infer,
        seed=20260820,
    )

    zero_change = ablation["zero_terrain"]["mean_action_l2"]
    baseline_mae = ablation["baseline"]["mae"]
    zero_mae = ablation["zero_terrain"]["prediction_metrics"]["mae"]
    if model.config.controller == "learned":
        height_gate = zero_change > 1e-2 and zero_mae > baseline_mae * 1.01
        height_gate_note = "learned must change actions and worsen when terrain is zeroed"
    else:
        height_gate = zero_change <= 1e-7
        height_gate_note = "proprio_clone must be exactly invariant to terrain"

    comparison = None
    comparison_gate = True
    if args.reference_checkpoint:
        reference, _, _ = load_checkpoint(args.reference_checkpoint, map_location=device)
        reference.to(device).eval()
        reference_report, _ = evaluate_dataset(
            reference, validation, device, args.batch_size)
        candidate_by_terrain = validation_report["by_terrain"]
        reference_by_terrain = reference_report["by_terrain"]
        better = [
            terrain for terrain in candidate_by_terrain
            if candidate_by_terrain[terrain]["mae"] < reference_by_terrain[terrain]["mae"]
        ]
        comparison_gate = (
            validation_report["overall"]["mae"] < reference_report["overall"]["mae"]
            and len(better) >= max(1, len(candidate_by_terrain) // 2)
        )
        comparison = {
            "reference_checkpoint": str(args.reference_checkpoint.resolve()),
            "reference": reference_report,
            "candidate_better_terrains": better,
            "candidate_better_terrain_count": len(better),
            "candidate_better_overall": (
                validation_report["overall"]["mae"]
                < reference_report["overall"]["mae"]),
        }

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "controller": model.config.controller,
        "architecture": model.config.architecture,
        "teacher_onnx_sha256": teacher_sha256,
        "training_metadata": training_metadata,
        "validation": validation_report,
        "stress": stress_report,
        "terrain_ablation": ablation,
        "quality_gates": {
            "height_behavior_passed": height_gate,
            "height_behavior_note": height_gate_note,
            "reference_comparison_passed": comparison_gate,
            "all_passed": height_gate and comparison_gate,
        },
        "reference_comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_gates and not report["quality_gates"]["all_passed"]:
        raise SystemExit("student failed one or more offline quality gates")


if __name__ == "__main__":
    main()
