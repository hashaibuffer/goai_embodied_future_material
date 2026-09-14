"""Layer-aware Low height selection from physical rays, never waypoint geometry.

Raw scans remain available for other actors. A connected layer starts below the
base, accepts upward-facing surfaces within one Low step, and checks the short
horizontal link above the higher tread. Unknown/disconnected cells retain the
raw obstacle instead of silently inventing a free floor. Head clearance remains
the independent Detector's responsibility.
"""
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_teacher import PrivilegedHeightScan, sanitize_privileged_height_grid


@dataclass(frozen=True)
class SurfaceSelection:
    scan: PrivilegedHeightScan
    connected: np.ndarray
    changed: np.ndarray
    xy_w: np.ndarray
    conversion_error_m: float


def scan_xy_world(scanner, position, rotation):
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return scanner.local_xy @ np.array([[c, s], [-s, c]]) + position[:2]


def wheel_tread_heights(scanner, data, wheel_positions):
    """Physical floor directly below each wheel centre (FL, FR, HL, HR).

    Start below overhead slabs, reject undersides and distant/missing floors.
    No XY extension or Actor encoding participates in this Router signal.
    """
    result = np.full(4, np.nan)
    geom = np.empty(1, np.int32)
    normal = np.empty(3, np.float64)
    for i, point in enumerate(np.asarray(wheel_positions, np.float64)):
        distance = mujoco.mj_ray(scanner.model, data, point, np.array([0., 0., -1.]),
                                scanner.geomgroup, True, scanner.body_exclude, geom, normal)
        if 0. <= distance <= .3 and normal[2] >= .5:
            result[i] = point[2] - distance
    return result


def select_support_surface(scanner, data, position, rotation, raw, max_step_m=.18,
                           *, query_x_range=None, query_half_width=0.):
    """Select a locally connected layer; bounded to eight intersections/ray.

    Requires MuJoCo mj_ray surface normals. Recasting below each intersection
    finds floors beneath slabs; downward-facing slab undersides are not support.
    No geom is removed or globally excluded (a mesh may contain both floors).
    """
    position = np.asarray(position)
    xy = scan_xy_world(scanner, position, np.asarray(rotation))
    geom = np.empty(1, np.int32)
    normal = np.empty(3, np.float64)
    down = np.array([0., 0., -1.])
    up = -down
    up_geom = np.empty(1, np.int32)
    up_normal = np.empty(3, np.float64)

    def layers(point_xy, start_z):
        start = np.array([*point_xy, start_z], np.float64)
        values = []
        for _ in range(8):
            geom[0] = -1
            distance = mujoco.mj_ray(scanner.model, data, start, down,
                                    scanner.geomgroup, True, scanner.body_exclude, geom, normal)
            if geom[0] < 0 or distance < 0:
                break
            z = float(start[2] - distance)
            if z < position[2] - 3.0:
                break
            if normal[2] >= .5:
                # A floor underneath a SOLID box is not a second free layer.
                # From free space an upward ray enters a ceiling (normal -Z);
                # an upward-facing exit means this clearance point is inside
                # solid terrain. This also prevents flood-fill tunnelling.
                above = np.array([*point_xy, z + .025])
                up_distance = mujoco.mj_ray(scanner.model, data, above, up,
                                           scanner.geomgroup, True, scanner.body_exclude,
                                           up_geom, up_normal)
                if up_distance < 0 or up_normal[2] <= 0.:
                    values.append((z, int(geom[0])))
            start[2] = z - 1.e-5
        return values

    seed_layers = layers(position[:2], position[2])
    if not seed_layers:
        return SurfaceSelection(raw, np.zeros(1353, bool), np.zeros(1353, bool), xy, 0.)
    ground = seed_layers[0][0]
    # Rescan only samples consumed by the XY clamp plus the Detector corridor.
    # Include the bracketing grid columns for non-grid-aligned boundaries.
    # Outside this mask the selected layer explicitly remains raw, not filled.
    needed = np.ones(1353, bool)
    if query_x_range is not None:
        lo, hi = min(query_x_range[0], 0.), max(query_x_range[1], 1.2)
        lo = scanner.X[max(0, np.searchsorted(scanner.X, lo, side="right") - 1)]
        hi = scanner.X[min(40, np.searchsorted(scanner.X, hi, side="left"))]
        needed &= (scanner.local_xy[:, 0] >= lo) & (scanner.local_xy[:, 0] <= hi)
    if query_half_width:
        width = max(query_half_width, .25)
        lo = scanner.Y[max(0, np.searchsorted(scanner.Y, -width, side="right") - 1)]
        hi = scanner.Y[min(32, np.searchsorted(scanner.Y, width, side="left"))]
        needed &= (scanner.local_xy[:, 1] >= lo) & (scanner.local_xy[:, 1] <= hi)
    candidates = [layers(p, position[2] + scanner.RAY_ORIGIN_OFFSET_Z) if use else []
                  for p, use in zip(xy, needed)]
    chosen = np.full(1353, np.nan)
    chosen_geom = raw.geom_ids.copy()

    def visible_link(a_xy, a_z, b_xy, b_z):
        start = np.array([*a_xy, max(a_z, b_z) + .025])
        delta = np.array([*(b_xy - a_xy), 0.])
        length = float(np.linalg.norm(delta))
        if length < 1.e-8:
            return True
        distance = mujoco.mj_ray(scanner.model, data, start, delta / length,
                                scanner.geomgroup, True, scanner.body_exclude, geom)
        return distance < 0 or distance >= length - 1.e-5

    seed = int(np.argmin(np.sum(scanner.local_xy ** 2, axis=1)))
    options = sorted(candidates[seed], key=lambda item: abs(item[0] - ground))
    if not options or abs(options[0][0] - ground) > max_step_m:
        return SurfaceSelection(raw, np.zeros(1353, bool), np.zeros(1353, bool), xy, 0.)
    chosen[seed], chosen_geom[seed] = options[0]
    queue = deque([seed])
    while queue:
        current = queue.popleft()
        row, col = divmod(current, 41)
        neighbors = []
        if row: neighbors.append(current - 41)
        if row < 32: neighbors.append(current + 41)
        if col: neighbors.append(current - 1)
        if col < 40: neighbors.append(current + 1)
        for neighbor in neighbors:
            if np.isfinite(chosen[neighbor]):
                continue
            for z, gid in sorted(candidates[neighbor], key=lambda item: abs(item[0] - chosen[current])):
                if abs(z - chosen[current]) > max_step_m:
                    continue
                if visible_link(xy[current], chosen[current], xy[neighbor], z):
                    chosen[neighbor], chosen_geom[neighbor] = z, gid
                    queue.append(neighbor)
                    break
    connected = np.isfinite(chosen)
    selected_z = np.where(connected, chosen, raw.hit_z_w)
    valid = np.isfinite(selected_z)
    height = position[2] - selected_z - scanner.HEIGHT_SCAN_OFFSET
    # Conversion check deliberately precedes clipping / missing-cell filling.
    error = float(np.max(np.abs((position[2] - scanner.HEIGHT_SCAN_OFFSET - height[valid]) - selected_z[valid]))) if valid.any() else 0.
    height = np.maximum(height, -1.)
    fallback = max(position[2] - ground - scanner.HEIGHT_SCAN_OFFSET, -1.)
    height = sanitize_privileged_height_grid(height.reshape(33, 41), valid.reshape(33, 41), fallback)
    scan = PrivilegedHeightScan(height.reshape(-1), valid, chosen_geom, selected_z, ground)
    changed = connected & (~raw.hit | (np.abs(selected_z - raw.hit_z_w) > 1.e-4))
    return SurfaceSelection(scan, connected, changed, xy, error)


def height_debug_record(raw, selection, actor_height, position):
    """Three distinct layers; final coordinates include synthetic extensions."""
    selected = selection.scan
    effective_z = float(position[2]) - .5 - np.asarray(actor_height).reshape(41, 33).T.reshape(-1)
    expected_raw_height = float(position[2]) - raw.hit_z_w - .5
    # Only compare cells untouched by the legacy height clipping/deep-drop
    # encoding; those encodings are intentionally not physical hit locations.
    comparable = raw.hit & (expected_raw_height >= -1.) & (expected_raw_height <= .52)
    raw_error = float(np.max(np.abs(raw.height[comparable] - expected_raw_height[comparable]))) if comparable.any() else None
    def finite_list(values):
        return [float(x) if np.isfinite(x) else None for x in values]
    return {
        "xy_w": selection.xy_w.tolist(),
        "actor_effective_is_virtual": True,
        "raw_hit_z_w": finite_list(raw.hit_z_w),
        "raw_geom_ids": raw.geom_ids.tolist(),
        "selected_hit_z_w": finite_list(selected.hit_z_w),
        "selected_geom_ids": selected.geom_ids.tolist(),
        "actor_effective_z_w": finite_list(effective_z),
        "connected": selection.connected.tolist(),
        "changed_surface_count": int(selection.changed.sum()),
        "postprocess_changed_count": int(np.sum(np.abs(effective_z - selected.hit_z_w) > 1.e-4)),
        "conversion_error_m": selection.conversion_error_m,
        "raw_conversion_error_m": raw_error,
    }
