"""MuJoCo implementation of the S10 forward-edge and overhead-clearance detector."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ClearanceRay:
    start_w: np.ndarray
    end_w: np.ndarray
    hit: bool


@dataclass(frozen=True)
class ClimbDetection:
    has_target: bool
    edge_candidate: bool
    overhead_rejected: bool
    upper_clearance_blocked: bool
    command_compatible: bool
    height_m: float
    lower_z_w: float
    upper_z_w: float
    edge_x_b: float
    edge_segment_w: np.ndarray
    landing_corners_w: np.ndarray
    corridor_corners_w: np.ndarray
    clearance_rays: tuple[ClearanceRay, ...]


def _yaw_transform(points_b: np.ndarray, base_pos_w, base_rotation_w) -> np.ndarray:
    rotation = np.asarray(base_rotation_w, np.float64).reshape(3, 3)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    c, s = np.cos(yaw), np.sin(yaw)
    points = np.asarray(points_b, np.float64).reshape(-1, 3).copy()
    xy = points[:, :2] @ np.asarray([[c, s], [-s, c]])
    points[:, :2] = xy + np.asarray(base_pos_w, np.float64)[:2]
    points[:, 2] += float(np.asarray(base_pos_w)[2])
    return points


class ForwardClearanceScanner:
    """Sparse upper-body ray fan; it is a Router guard, never an Actor input."""

    def __init__(self, model, body_exclude: int, contract: dict) -> None:
        self.model = model
        self.body_exclude = int(body_exclude)
        self.start_x = float(contract["clearance_start_x_m"])
        self.max_range = float(contract["clearance_range_m"])
        self.lateral = tuple(float(x) for x in contract["clearance_lateral_m"])
        self.heights = tuple(
            float(x) for x in contract["clearance_height_above_base_m"]
        )
        self.geomgroup = np.asarray([1, 0, 0, 1, 1, 1], dtype=np.uint8)

    def scan(self, data, base_pos_w, base_rotation_w) -> tuple[ClearanceRay, ...]:
        rotation = np.asarray(base_rotation_w, np.float64).reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        direction = np.asarray([np.cos(yaw), np.sin(yaw), 0.0], np.float64)
        starts_b = np.asarray(
            [[self.start_x, y, z] for z in self.heights for y in self.lateral],
            np.float64,
        )
        starts_w = _yaw_transform(starts_b, base_pos_w, base_rotation_w)
        geom_out = np.empty(1, dtype=np.int32)
        rays = []
        for start in starts_w:
            geom_out[0] = -1
            distance = float(
                mujoco.mj_ray(
                    self.model,
                    data,
                    start,
                    direction,
                    self.geomgroup,
                    True,
                    self.body_exclude,
                    geom_out,
                )
            )
            hit = geom_out[0] >= 0 and 0.0 <= distance <= self.max_range
            length = distance if hit else self.max_range
            rays.append(ClearanceRay(start.copy(), start + direction * length, hit))
        return tuple(rays)


class S10ClimbDetector:
    """Detect a forward landing patch from raw downward hits.

    The Actor still receives the frozen 1353-cell height map. This class only
    decides Router ownership and rejects a raised top surface when the upper
    forward fan proves that it belongs to a low overhang.
    """

    def __init__(self, model, body_exclude: int, contract: dict) -> None:
        self.model = model
        self.body_exclude = int(body_exclude)
        self.front_min, self.front_max = map(float, contract["front_x_range_m"])
        self.half_width = float(contract["corridor_half_width_m"])
        self.min_height = float(contract["min_height_m"])
        self.max_height = float(contract["max_height_m"])
        self.min_width = float(contract["min_landing_width_m"])
        self.min_depth = float(contract["min_landing_depth_m"])
        self.height_tolerance = float(contract["landing_height_tolerance_m"])
        self.coverage_min = float(contract["landing_coverage_min"])
        self.clearance_lateral_count = len(contract["clearance_lateral_m"])
        self.clearance = ForwardClearanceScanner(model, body_exclude, contract)
        self.x = np.linspace(-0.8, 3.2, 41)
        self.y = np.linspace(-1.6, 1.6, 33)

    def _empty(self, compatible, corridor, rays) -> ClimbDetection:
        zeros = np.zeros((0, 3), np.float64)
        upper_blocked = any(
            ray.hit for ray in rays[self.clearance_lateral_count :]
        )
        return ClimbDetection(
            False, False, False, upper_blocked, compatible, 0.0, 0.0, 0.0, 0.0,
            zeros, zeros, corridor, rays,
        )

    def detect(
        self,
        scan,
        data,
        base_pos_w,
        base_rotation_w,
        command,
    ) -> ClimbDetection:
        command = np.asarray(command, np.float64).reshape(3)
        compatible = bool(
            command[0] > 0.1
            and abs(command[1]) <= 0.1
            and abs(command[2]) <= 0.1
        )
        corridor_b = np.asarray(
            [
                [self.front_min, -self.half_width, 0.0],
                [self.front_max, -self.half_width, 0.0],
                [self.front_max, self.half_width, 0.0],
                [self.front_min, self.half_width, 0.0],
            ]
        )
        corridor = _yaw_transform(corridor_b, base_pos_w, base_rotation_w)
        ground_z = float(scan.base_ground_z_w)
        corridor[:, 2] = ground_z if np.isfinite(ground_z) else float(base_pos_w[2])
        rays = self.clearance.scan(data, base_pos_w, base_rotation_w)
        if not compatible or not np.isfinite(ground_z):
            return self._empty(compatible, corridor, rays)

        valid = np.asarray(scan.hit, bool).reshape(33, 41)
        z = np.asarray(scan.hit_z_w, np.float64).reshape(33, 41)
        dz = z[:, 1:] - z[:, :-1]
        valid_pair = valid[:, 1:] & valid[:, :-1]
        edge_x = 0.5 * (self.x[1:] + self.x[:-1])
        in_x = (edge_x >= self.front_min) & (edge_x <= self.front_max)
        in_y = np.abs(self.y) <= self.half_width + 1.0e-6
        rising = (
            valid_pair
            & (dz >= 0.40 * self.min_height)
            & in_y[:, None]
            & in_x[None, :]
        )
        edge_columns = np.flatnonzero(rising.any(axis=0))
        if edge_columns.size == 0:
            return self._empty(compatible, corridor, rays)
        column = int(edge_columns[0])
        x_edge = float(edge_x[column])
        edge_rows = np.flatnonzero(rising[:, column])

        lower_mask_x = (self.x < x_edge) & (self.x >= x_edge - 0.20)
        upper_mask_x = (self.x > x_edge) & (self.x <= x_edge + 0.20)
        corridor_rows = in_y
        lower_values = z[np.ix_(corridor_rows, lower_mask_x)]
        lower_valid = valid[np.ix_(corridor_rows, lower_mask_x)]
        upper_values = z[np.ix_(corridor_rows, upper_mask_x)]
        upper_valid = valid[np.ix_(corridor_rows, upper_mask_x)]
        if not lower_valid.any() or not upper_valid.any():
            return self._empty(compatible, corridor, rays)
        lower_z = float(np.median(lower_values[lower_valid]))
        upper_z = float(np.median(upper_values[upper_valid]))
        height = upper_z - lower_z

        landing_rows = np.abs(self.y) <= 0.5 * self.min_width + 1.0e-6
        landing_cols = (self.x >= x_edge) & (self.x <= x_edge + self.min_depth + 1.0e-6)
        landing_values = z[np.ix_(landing_rows, landing_cols)]
        landing_valid = valid[np.ix_(landing_rows, landing_cols)]
        on_upper = landing_valid & (
            np.abs(landing_values - upper_z) <= self.height_tolerance
        )
        coverage = float(on_upper.sum() / max(1, landing_valid.sum()))
        transverse_span = (
            float(self.y[edge_rows[-1]] - self.y[edge_rows[0]] + 0.10)
            if edge_rows.size else 0.0
        )
        geometry_ok = bool(
            self.min_height <= height <= self.max_height
            and coverage >= self.coverage_min
            and transverse_span + 1.0e-6 >= self.min_width
        )
        low_fan_blocked = any(
            ray.hit for ray in rays[: self.clearance_lateral_count]
        )
        upper_fan_blocked = any(
            ray.hit for ray in rays[self.clearance_lateral_count :]
        )
        overhead_rejected = bool(
            geometry_ok
            and upper_fan_blocked
            and not low_fan_blocked
        )

        edge_b = np.asarray(
            [
                [x_edge, -0.5 * self.min_width, upper_z - float(base_pos_w[2])],
                [x_edge, 0.5 * self.min_width, upper_z - float(base_pos_w[2])],
            ]
        )
        edge_segment = _yaw_transform(edge_b, base_pos_w, base_rotation_w)
        landing_b = np.asarray(
            [
                [x_edge, -0.5 * self.min_width, upper_z - float(base_pos_w[2])],
                [x_edge + self.min_depth, -0.5 * self.min_width, upper_z - float(base_pos_w[2])],
                [x_edge + self.min_depth, 0.5 * self.min_width, upper_z - float(base_pos_w[2])],
                [x_edge, 0.5 * self.min_width, upper_z - float(base_pos_w[2])],
            ]
        )
        landing = _yaw_transform(landing_b, base_pos_w, base_rotation_w)
        return ClimbDetection(
            has_target=geometry_ok and not overhead_rejected,
            edge_candidate=geometry_ok,
            overhead_rejected=overhead_rejected,
            upper_clearance_blocked=upper_fan_blocked,
            command_compatible=compatible,
            height_m=max(0.0, height),
            lower_z_w=lower_z,
            upper_z_w=upper_z,
            edge_x_b=x_edge,
            edge_segment_w=edge_segment,
            landing_corners_w=landing,
            corridor_corners_w=corridor,
            clearance_rays=rays,
        )
