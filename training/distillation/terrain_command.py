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

    # Measure frontal risk along the wheel corridor. Absolute maxima over the
    # whole ROI made a side wall or one noisy ray look like a frontal step.
    corridor_h = heights[:, 3:9]
    corridor_v = valid[:, 3:9]
    profile = np.full(8, np.nan, dtype=np.float32)
    for x in range(8):
        values = corridor_h[x][corridor_v[x]]
        if values.size >= 3:
            profile[x] = np.median(values)
    differences = np.diff(profile)
    finite_differences = differences[np.isfinite(differences)]
    max_step = max(float(finite_differences.max(initial=0.0)), 0.0)
    max_drop = max(float((-finite_differences).max(initial=0.0)), 0.0)
    finite_profile = profile[np.isfinite(profile)]
    slope = (float(finite_profile[-1] - finite_profile[0])
             if finite_profile.size >= 4 else 0.0)

    # Side obstacle estimates use a robust percentile rather than one ray.
    left_values = heights[:, 9:][valid[:, 9:]]
    right_values = heights[:, :3][valid[:, :3]]
    left = max(float(np.sort(left_values)[int(0.75 * (left_values.size - 1))])
               if left_values.size else 0.0, 0.0)
    right = max(float(np.sort(right_values)[int(0.75 * (right_values.size - 1))])
                if right_values.size else 0.0, 0.0)
    valid_fraction = float(valid.mean())
    unknown_fraction = 1.0 - valid_fraction
    risk_score = float(np.clip(max(
        max_step / 0.25,
        max_drop / 0.20,
        0.45 * abs(slope) / 0.35,
        0.70 * max(0.0, (unknown_fraction - 0.45) / 0.55),
    ), 0.0, 1.0))

    target = np.clip(raw, [-1.0, -0.6, -1.0], [1.0, 0.6, 1.0]).astype(np.float32)
    active_risk = float(np.clip((risk_score - 0.25) / 0.75, 0.0, 1.0))
    speed_scale = np.float32(1.05 - 0.10 * active_risk)
    target[0] = np.clip(target[0] * speed_scale, -1.0, 1.0)
    imbalance = np.sign(right - left) * max(abs(right - left) - 0.12, 0.0)
    target[1] = np.clip(target[1] + np.clip(
        imbalance * 0.15, -0.03, 0.03), -0.6, 0.6)
    target[2] *= np.float32(1.0)
    command = _slew(previous, target).astype(np.float32)
    risk = np.asarray([
        max_step, max_drop, slope, left, right,
        unknown_fraction, valid_fraction, risk_score,
    ], dtype=np.float32)
    return command, risk
