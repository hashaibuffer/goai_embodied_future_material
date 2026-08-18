#!/usr/bin/env python3
"""
heightmap.py — S10 T4 高度图编码（纯 numpy + yaml，不依赖 ROS）

按 T00 官方 configs/heightmap.yaml 语义实现：
- 点云投影到 yaw-only 水平系（去 roll/pitch，保留世界垂直 z）：
  x=前方、y=左侧、z=世界垂直（+Z up）
- full_grid (nx, ny) @ resolution 聚合：格内最高命中点 z（axis0=x, axis1=y）
- ground_reference (valid_median)：机器人附近有效格的 z 中位数作为零高基准
- height_normalized = clip((z - ground_ref) / height_divisor_m, 0, 1)
- policy_grid (nx, ny)：在 full 上 bilinear_center 降采样（每格取 cell 中心双线性插值高度，
  掩码取 cell 覆盖区域任一有效）
- 输出展平 (2, nx, ny) channel-major：先全部高度，再全部掩码
未知格：height = unknown_height_fill(0)，validity = unknown_validity(0)
"""
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "heightmap.yaml"

# 默认值（与官方 configs/heightmap.yaml 一致）
HEIGHTMAP_DEFAULTS = {
    "topic": "/S10_HEIGHTMAP",
    "frame": "robot_horizontal",
    "x_min": -0.8, "x_max": 3.2,
    "y_min": -1.5, "y_max": 1.5,
    "resolution": 0.10,
    "full_nx": 40, "full_ny": 30,
    "policy_nx": 16, "policy_ny": 12,
    "downsample": "bilinear_center",
    "clip_m": (-0.40, 0.80),
    "height_divisor_m": 0.80,
    "unknown_height_fill": 0.0,
    "unknown_validity": 0.0,
    "ground_ref_x_range": (-0.40, 0.40),
    "ground_ref_y_range": (-0.40, 0.40),
}


def load_config(path=None):
    """读取官方 heightmap.yaml（T00 冻结格式），返回 hm dict（带类型）。"""
    cfg = dict(HEIGHTMAP_DEFAULTS)
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_file():
        return cfg
    try:
        data = yaml.safe_load(p.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"[heightmap] 读取配置 {p} 失败，使用默认值: {e}")
        return cfg
    topics = data.get("topics") or {} if isinstance(data, dict) else {}
    h = data.get("heightmap") or {} if isinstance(data, dict) else {}
    if isinstance(topics, dict) and "heightmap" in topics:
        cfg["topic"] = topics["heightmap"]
    if isinstance(h, dict):
        for k in ("frame", "downsample"):
            if k in h:
                cfg[k] = h[k]
        for k in ("x_min", "x_max", "y_min", "y_max", "resolution",
                  "height_divisor_m", "unknown_height_fill", "unknown_validity"):
            if k in h:
                cfg[k] = float(h[k])
        if "full_grid" in h and len(h["full_grid"]) == 2:
            cfg["full_nx"], cfg["full_ny"] = int(h["full_grid"][0]), int(h["full_grid"][1])
        if "policy_grid" in h and len(h["policy_grid"]) == 2:
            cfg["policy_nx"], cfg["policy_ny"] = int(h["policy_grid"][0]), int(h["policy_grid"][1])
        if "clip_m" in h and len(h["clip_m"]) == 2:
            cfg["clip_m"] = (float(h["clip_m"][0]), float(h["clip_m"][1]))
        gr = h.get("ground_reference") or {}
        if isinstance(gr, dict):
            if "x_range" in gr and len(gr["x_range"]) == 2:
                cfg["ground_ref_x_range"] = (float(gr["x_range"][0]), float(gr["x_range"][1]))
            if "y_range" in gr and len(gr["y_range"]) == 2:
                cfg["ground_ref_y_range"] = (float(gr["y_range"][0]), float(gr["y_range"][1]))
    return cfg


def yaw_only_rotation(R):
    """从完整 body→world 旋转矩阵提取 yaw，返回仅绕世界 z 的旋转矩阵（纯 numpy）。

    R 对应 base_pose 的 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)，故 yaw = atan2(R[1,0], R[0,0])。
    返回矩阵只去 yaw（去掉 roll/pitch 倾斜），z 轴保持世界垂直。
    """
    yaw = np.arctan2(R[1, 0], R[0, 0])
    cz, sz = np.cos(yaw), np.sin(yaw)
    return np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _cell_ranges(x_min, x_max, y_min, y_max, nx, ny):
    res_x = (x_max - x_min) / nx
    res_y = (y_max - y_min) / ny
    return res_x, res_y


def _aggregate_full(p_h, hm):
    """水平系命中点 → (nx, ny) 高度(max z) + 掩码。axis0=x、axis1=y，越界丢弃。"""
    nx, ny = int(hm["full_nx"]), int(hm["full_ny"])
    x_min, x_max = float(hm["x_min"]), float(hm["x_max"])
    y_min, y_max = float(hm["y_min"]), float(hm["y_max"])
    res_x, res_y = _cell_ranges(x_min, x_max, y_min, y_max, nx, ny)

    height = np.zeros((nx, ny), np.float32)
    mask = np.zeros((nx, ny), np.float32)
    if len(p_h) == 0:
        return height, mask

    col = ((p_h[:, 0] - x_min) / res_x).astype(np.intp)   # x → axis0
    row = ((p_h[:, 1] - y_min) / res_y).astype(np.intp)   # y → axis1
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    xi, yi = col[ok], row[ok]
    z = p_h[ok, 2].astype(np.float32)
    # 注意：初始 -inf（不能 0），否则 max 会把负 z（地面在 pos.z 下方）丢成 0；
    # 无命中格在末尾回填 0（= unknown_height_fill 语义）
    tmp = np.full((nx, ny), -np.inf, np.float32)
    np.maximum.at(tmp, (xi, yi), z)                        # 格内最高命中点 z
    tmp[~np.isfinite(tmp)] = 0.0
    height[...] = tmp
    mask[xi, yi] = 1.0
    return height, mask


def _ground_reference_median(height, mask, hm):
    """机器人附近（ground_reference.x_range × y_range）有效格的 z 中位数。无有效格 → 0。"""
    nx, ny = height.shape
    x_min, x_max = float(hm["x_min"]), float(hm["x_max"])
    y_min, y_max = float(hm["y_min"]), float(hm["y_max"])
    res_x, res_y = _cell_ranges(x_min, x_max, y_min, y_max, nx, ny)
    gx0, gx1 = hm["ground_ref_x_range"]
    gy0, gy1 = hm["ground_ref_y_range"]

    c0 = int(np.floor((gx0 - x_min) / res_x)); c1 = int(np.ceil((gx1 - x_min) / res_x))
    r0 = int(np.floor((gy0 - y_min) / res_y)); r1 = int(np.ceil((gy1 - y_min) / res_y))
    c0, c1 = max(c0, 0), min(c1, nx)
    r0, r1 = max(r0, 0), min(r1, ny)
    if c1 <= c0 or r1 <= r0:
        return 0.0
    vals = height[c0:c1, r0:r1][mask[c0:c1, r0:r1] > 0]
    return float(np.median(vals)) if len(vals) else 0.0


def _downsample_bilinear_center(height, mask, hm):
    """policy cell 中心在 full grid 上双线性采样高度；掩码取 cell 覆盖区域任一有效。"""
    nx, ny = int(hm["policy_nx"]), int(hm["policy_ny"])
    f_nx, f_ny = height.shape
    x_min, x_max = float(hm["x_min"]), float(hm["x_max"])
    y_min, y_max = float(hm["y_min"]), float(hm["y_max"])
    res_fx, res_fy = _cell_ranges(x_min, x_max, y_min, y_max, f_nx, f_ny)
    res_px, res_py = _cell_ranges(x_min, x_max, y_min, y_max, nx, ny)

    # policy cell 中心在 full grid 上的浮点坐标（格心对齐：+0.5）
    fx = (x_min + (np.arange(nx, dtype=np.float64) + 0.5) * res_px - x_min) / res_fx
    fy = (y_min + (np.arange(ny, dtype=np.float64) + 0.5) * res_py - y_min) / res_fy

    # ---- 高度：双线性插值（边界 clip 到最后一个格）----
    x0 = np.clip(np.floor(fx).astype(np.intp), 0, f_nx - 1)
    y0 = np.clip(np.floor(fy).astype(np.intp), 0, f_ny - 1)
    x1 = np.clip(x0 + 1, 0, f_nx - 1)
    y1 = np.clip(y0 + 1, 0, f_ny - 1)
    wx1 = np.clip(fx - x0, 0.0, 1.0); wx0 = 1.0 - wx1
    wy1 = np.clip(fy - y0, 0.0, 1.0); wy0 = 1.0 - wy1

    pol_h = (
        wx0[:, None] * wy0[None, :] * height[np.ix_(x0, y0)]
        + wx1[:, None] * wy0[None, :] * height[np.ix_(x1, y0)]
        + wx0[:, None] * wy1[None, :] * height[np.ix_(x0, y1)]
        + wx1[:, None] * wy1[None, :] * height[np.ix_(x1, y1)]
    )

    # ---- 掩码：policy cell 覆盖的 full 区域任一有效（保守，不丢有效格）----
    half_x = 0.5 * res_px / res_fx
    half_y = 0.5 * res_py / res_fy
    pol_m = np.zeros((nx, ny), np.float32)
    for i in range(nx):
        lo = int(np.floor(fx[i] - half_x)); hi = int(np.ceil(fx[i] + half_x))
        lo, hi = max(lo, 0), min(hi, f_nx)
        for j in range(ny):
            jlo = int(np.floor(fy[j] - half_y)); jhi = int(np.ceil(fy[j] + half_y))
            jlo, jhi = max(jlo, 0), min(jhi, f_ny)
            if lo < hi and jlo < jhi:
                pol_m[i, j] = float(mask[lo:hi, jlo:jhi].max())
    return pol_h, pol_m


def build_heightmap(points, pos, R, hm):
    """world 系命中点 → policy 高度图 (2, nx, ny) float32（channel-major）。

    points: (N,3) world 系命中点（已含 site_offset，为加噪/丢点后的最终点）
    pos   : (3,)   位姿平移（world）
    R     : (3,3)  body→world 姿态矩阵（内部只取 yaw）
    hm    : dict   heightmap 配置节

    返回 (2, policy_nx, policy_ny)：通道 0 = 归一化高度、通道 1 = 有效性掩码。
    展平（C-order）= 先 policy_nx*policy_ny 高度，再同量掩码（flatten_order）。
    """
    R_yaw = yaw_only_rotation(R)
    p_h = (R_yaw.T @ (points - pos).T).T                  # (N,3) 水平系

    full_h, full_m = _aggregate_full(p_h, hm)             # (nx, ny)
    ground_ref = _ground_reference_median(full_h, full_m, hm)

    divisor = float(hm["height_divisor_m"])
    full_norm = np.clip((full_h - ground_ref) / divisor, 0.0, 1.0)

    pol_h, pol_m = _downsample_bilinear_center(full_norm, full_m, hm)

    # 未知格：高度填 unknown_height_fill、掩码 unknown_validity
    pol_h = np.where(pol_m > 0, pol_h, float(hm["unknown_height_fill"]))
    pol_m = np.where(pol_m > 0, 1.0, float(hm["unknown_validity"]))

    return np.stack([pol_h, pol_m], axis=0).astype(np.float32)   # (2, nx, ny) float32
