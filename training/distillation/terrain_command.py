"""Numpy mirror of the TD C++ terrain command contract."""

from __future__ import annotations

import numpy as np

RISK_NAMES = (
    "max_step_m",
    "max_drop_m",
    "slope_m",
    "left_obstacle_m",
    "right_obstacle_m",
    "unknown_fraction",
    "valid_fraction",
    "risk_score",
)


def _slew(previous: np.ndarray, target: np.ndarray) -> np.ndarray:
    limits = np.asarray([0.04, 0.03, 0.06], dtype=np.float32)
    return previous + np.clip(target - previous, -limits, limits)


def rewrite_command(cmd_raw, heightmap, previous=(0.0, 0.0, 0.0)):
    """Return `(cmd_terrain, risk_features)` using the frozen C++ formula."""
    raw = np.asarray(cmd_raw, dtype=np.float32)
    grid = np.asarray(heightmap, dtype=np.float32).reshape(2, 16, 12)
    previous = np.asarray(previous, dtype=np.float32)
    if raw.shape != (3,) or previous.shape != (3,):
        raise ValueError("commands must have shape (3,)")
    if not np.isfinite(raw).all() or not np.isfinite(grid).all():
        raise ValueError("command and heightmap must be finite")

    heights = grid[0, 4:12] * np.float32(0.8)
    valid = grid[1, 4:12] > np.float32(0.5)
    values = heights[valid]
    max_step = max(float(values.max(initial=0.0)), 0.0)
    max_drop = max(float((-values).max(initial=0.0)), 0.0)
    near_values = heights[:4][valid[:4]]
    far_values = heights[4:][valid[4:]]
    near_mean = float(near_values.mean()) if near_values.size else 0.0
    far_mean = float(far_values.mean()) if far_values.size else near_mean
    slope = far_mean - near_mean
    left_values = heights[:, 6:][valid[:, 6:]]
    right_values = heights[:, :6][valid[:, :6]]
    left = max(float(left_values.max(initial=0.0)), 0.0)
    right = max(float(right_values.max(initial=0.0)), 0.0)
    valid_fraction = float(valid.mean())
    unknown_fraction = 1.0 - valid_fraction
    risk_score = float(np.clip(max(
        max_step / 0.25,
        max_drop / 0.20,
        abs(slope) / 0.20,
        max(0.0, (unknown_fraction - 0.35) / 0.65),
    ), 0.0, 1.0))

    target = np.clip(raw, [-1.0, -0.6, -1.0], [1.0, 0.6, 1.0]).astype(np.float32)
    active_risk = float(np.clip((risk_score - 0.15) / 0.85, 0.0, 1.0))
    target[0] *= np.float32(1.0 - 0.50 * active_risk)
    target[1] = np.clip(target[1] + np.clip(
        (right - left) * 1.2, -0.25, 0.25), -0.6, 0.6)
    target[2] *= np.float32(1.0 - 0.35 * active_risk)
    command = _slew(previous, target).astype(np.float32)
    risk = np.asarray([
        max_step, max_drop, slope, left, right,
        unknown_fraction, valid_fraction, risk_score,
    ], dtype=np.float32)
    return command, risk
