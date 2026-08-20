"""Offline metrics and terrain ablations for TE students."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from .te_dataset import ACTION_DIM, HEIGHT_DIM, PROPRIO_DIM, VALIDITY_DIM


POLICY_TO_ROBOT = np.asarray(
    [0, 1, 2, 12, 3, 4, 5, 13, 6, 7, 8, 14, 9, 10, 11, 15],
    dtype=np.int64,
)
ACTION_SCALE_ROBOT = np.asarray([0.125, 0.25, 0.25, 5.0] * 4, dtype=np.float32)
LEG_ROBOT_INDICES = np.asarray([i for i in range(ACTION_DIM) if i % 4 != 3])
WHEEL_ROBOT_INDICES = np.asarray([i for i in range(ACTION_DIM) if i % 4 == 3])


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != ACTION_DIM:
        raise ValueError("target and prediction must both be [N,16]")
    error = prediction - target
    absolute = np.abs(error)
    physical_error = error[:, POLICY_TO_ROBOT] * ACTION_SCALE_ROBOT
    return {
        "samples": len(target),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "max_abs": float(absolute.max(initial=0.0)),
        "leg_target_mae_rad": float(
            np.abs(physical_error[:, LEG_ROBOT_INDICES]).mean()),
        "wheel_target_mae_rad_s": float(
            np.abs(physical_error[:, WHEEL_ROBOT_INDICES]).mean()),
        "per_action_mae": absolute.mean(axis=0).astype(float).tolist(),
    }


def metrics_by_terrain(
    target: np.ndarray,
    prediction: np.ndarray,
    terrains: np.ndarray,
) -> dict[str, dict]:
    grouped = {}
    for terrain in sorted(set(map(str, terrains.tolist()))):
        selected = terrains == terrain
        grouped[terrain] = regression_metrics(target[selected], prediction[selected])
    return grouped


def predict_numpy(
    observations: np.ndarray,
    infer: Callable[[np.ndarray], np.ndarray],
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(observations), batch_size):
        batch = np.asarray(observations[start:start + batch_size], dtype=np.float32)
        output = np.asarray(infer(batch), dtype=np.float32)
        if output.shape != (len(batch), ACTION_DIM) or not np.isfinite(output).all():
            raise RuntimeError(f"invalid inference result: {output.shape}")
        outputs.append(output)
    return np.concatenate(outputs)


def terrain_ablation_metrics(
    observations: np.ndarray,
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    infer: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int = 0,
) -> dict:
    observations = np.asarray(observations, dtype=np.float32)
    rng = np.random.default_rng(seed)
    zeroed = observations.copy()
    zeroed[:, PROPRIO_DIM:] = 0.0
    shuffled = observations.copy()
    permutation = rng.permutation(len(shuffled))
    shuffled[:, PROPRIO_DIM:] = shuffled[permutation, PROPRIO_DIM:]
    zero_prediction = predict_numpy(zeroed, infer)
    shuffled_prediction = predict_numpy(shuffled, infer)

    def change(candidate: np.ndarray) -> dict:
        delta = candidate - baseline_prediction
        row_l2 = np.linalg.norm(delta, axis=1)
        return {
            "mean_action_l2": float(row_l2.mean()),
            "p95_action_l2": float(np.quantile(row_l2, 0.95)),
            "max_action_l2": float(row_l2.max(initial=0.0)),
            "changed_fraction_gt_1e-3": float(np.mean(row_l2 > 1e-3)),
            "prediction_metrics": regression_metrics(target, candidate),
        }

    return {
        "baseline": regression_metrics(target, baseline_prediction),
        "zero_terrain": change(zero_prediction),
        "shuffled_terrain": change(shuffled_prediction),
    }
