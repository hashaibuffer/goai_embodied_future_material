"""ROS-independent MuJoCo lidar kernel shared by runtime and collectors."""
from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np


@dataclass(frozen=True)
class LidarScan:
    points_w: np.ndarray
    geom_ids: np.ndarray
    raw_hit_count: int


class MujocoLidarScanner:
    """Official fan lidar without ROS, timers, or simulator ownership."""

    def __init__(self, model: mujoco.MjModel, cfg: dict, *, rng=None):
        self.model = model
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg["site_name"])
        self.body_exclude = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, cfg["body_exclude"])
        if self.site_id < 0:
            raise RuntimeError(f"site {cfg['site_name']!r} is missing from MJCF")
        if self.body_exclude < 0:
            raise RuntimeError(f"body {cfg['body_exclude']!r} is missing from MJCF")
        self.site_offset = model.site_pos[self.site_id].copy()
        azimuth = np.deg2rad(np.linspace(
            cfg["azimuth_deg"][0], cfg["azimuth_deg"][1], int(cfg["azimuth_beams"])))
        elevation = np.deg2rad(np.linspace(
            cfg["elevation_deg"][0], cfg["elevation_deg"][1], int(cfg["elevation_beams"])))
        self.fan = np.asarray(
            [[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]
             for e in elevation for a in azimuth], dtype=np.float64)
        self.nray = len(self.fan)
        self.geomgroup = np.asarray(cfg["geomgroup"], dtype=np.uint8)

    def scan(self, data, base_pos_w, base_rotation_w, *, apply_noise=True) -> LidarScan:
        pos = np.asarray(base_pos_w, dtype=np.float64).reshape(3)
        rotation = np.asarray(base_rotation_w, dtype=np.float64).reshape(3, 3)
        origin = pos + rotation @ self.site_offset
        directions_w = (rotation @ self.fan.T).T
        distances = np.full(self.nray, -1.0, dtype=np.float64)
        geom_ids = np.full(self.nray, -1, dtype=np.int32)
        mujoco.mj_multiRay(
            self.model, data, origin, directions_w.reshape(-1), self.geomgroup,
            bool(self.cfg["flg_static"]), self.body_exclude, geom_ids, distances,
            None, self.nray, float(self.cfg["cutoff"]))
        hit = geom_ids >= 0
        hit &= distances <= float(self.cfg["cutoff"])
        range_min = float(self.cfg.get("range_min", 0.0))
        if range_min > 0:
            hit &= distances >= range_min
        raw_hit_count = int(hit.sum())
        points = origin[None, :] + directions_w[hit] * distances[hit, None]
        raw_geom_ids = geom_ids[hit].copy()
        if apply_noise and len(points):
            noise_std = float(self.cfg.get("range_noise_std_m", 0.0))
            if noise_std > 0:
                points += directions_w[hit] * self.rng.normal(
                    0.0, noise_std, size=(raw_hit_count, 1))
            dropout = float(self.cfg.get("dropout_probability", 0.0))
            if dropout > 0:
                keep = self.rng.random(raw_hit_count) >= dropout
                points = points[keep]
        return LidarScan(points, raw_geom_ids, raw_hit_count)


def load_lidar_yaml(path):
    """Load frozen configs/lidar.yaml into the pure scanner configuration."""
    import yaml
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    lidar = document["lidar"]
    geomgroup = [1, 1, 1, 1, 1, 1]
    for group in lidar.get("exclude_geom_groups", []):
        geomgroup[int(group)] = 0
    if lidar.get("exclude_robot_geoms", False):
        geomgroup[1] = 0
    horizontal_fov = float(lidar["horizontal_fov_deg"])
    return {
        "site_name": lidar["site_name"], "body_exclude": "base_link",
        "azimuth_deg": [-horizontal_fov / 2, horizontal_fov / 2],
        "azimuth_beams": int(lidar["horizontal_rays"]),
        "elevation_deg": [0.0, -float(lidar["vertical_fov_deg"])],
        "elevation_beams": int(lidar["vertical_rays"]),
        "range_min": float(lidar["range_min"]), "cutoff": float(lidar["range_max"]),
        "geomgroup": geomgroup, "flg_static": True,
        "range_noise_std_m": float(lidar.get("range_noise_std_m", 0.0)),
        "dropout_probability": float(lidar.get("dropout_probability", 0.0)),
        "rate_hz": float(lidar["update_rate_hz"]),
    }
