#!/usr/bin/env python3
"""tests/test_heightmap_step.py — S10 T6 台阶单元测试（不依赖 ROS）。

场景：把一块已知高度的台阶放进世界系（车系前方 0.8~2.0m、高 0.4m），
验证高度图（build_heightmap）的：
  - 方向：台阶落在车系正前方区域（x 正）；车转向后，同一世界台阶随车旋转
    到高度图对应方位（掩码质心旋转 = 车 yaw 旋转）；
  - 数值：台阶 z=0.4m 相对地面(z=0) 归一化 → 0.4/0.8 = 0.5（clip 上限 0.8m → 1.0）；
  - 掩码：台阶区域 1、地面 1、后方空白 0。

台阶固定在世界系（world = POS + local，不再乘 R），车朝向 R 变化 ⇒
build_heightmap 内部 yaw-only 转到水平系时台阶相对车的位置随之旋转（正是 lidar 真值位姿驱动）。

运行：python3 -m pytest tests/test_heightmap_step.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "s10_terrain_perception"))
import heightmap as hm  # noqa: E402

POS = np.array([0.0, -2.5, 0.2])          # 官方 TRACK_START_BASE_POS
STEP_Z = 0.4                              # 台阶高度 m（相对地面 0）


def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _scene_worldfixed(step_z=STEP_Z):
    """世界系点：车附近地面 z=0 + 前方台阶（世界固定，不随车转）。

    台阶 local 坐标 x∈[0.8,2.0]（车系 I 时前方）、y∈[-0.5,0.5]、z=step_z。
    返回 world 系点集。车朝向 R 由 build_heightmap 参数决定。
    """
    gx = np.linspace(-0.3, 0.3, 7); gy = np.linspace(-0.3, 0.3, 7)
    gX, gY = np.meshgrid(gx, gy)
    ground_local = np.stack([gX.ravel(), gY.ravel(), np.zeros(49)], axis=1)

    # 台阶面加密采样(0.05m):full 网格 0.1m,须保证每个台阶格都有命中点(模拟实体面)
    sx = np.linspace(0.8, 2.0, 25); sy = np.linspace(-0.5, 0.5, 21)
    sX, sY = np.meshgrid(sx, sy)
    step_local = np.stack([sX.ravel(), sY.ravel(), np.full(525, step_z)], axis=1)

    local = np.vstack([ground_local, step_local])
    return POS + local                    # world = pos + local（固定，R 不参与）


def _mask_centroid(g):
    """高度图掩码质心的 (x, y)，用 policy cell 中心。"""
    m = g[1]
    idx = np.argwhere(m > 0)              # (K,2) → (i, j)
    xs = -0.8 + (idx[:, 0] + 0.5) * 0.25
    ys = -1.5 + (idx[:, 1] + 0.5) * 0.25
    return xs.mean(), ys.mean()


@pytest.fixture(scope="module")
def cfg():
    return hm.load_config()


# ---------- T6-1 方向：台阶在车系正前方 ----------
def test_step_in_front(cfg):
    g = hm.build_heightmap(_scene_worldfixed(), POS, np.eye(3), cfg)
    h, m = g[0], g[1]

    # 台阶覆盖区域 x∈[0.8,2.0]→i∈[7,10]、y∈[-0.5,0.5]→j∈[4,7]：掩码 1
    assert m[7:11, 4:8].min() == 1.0, "台阶区域掩码应全 1"
    # 车后方最远端(i=0, 中心 x=-0.675,无地面无台阶)掩码 0
    assert m[0, 4:8].max() == 0.0, "车后方(无台阶)掩码应 0"


# ---------- T6-2 数值：0.4m 台阶 → 归一化 0.5 ----------
def test_step_normalized_height(cfg):
    g = hm.build_heightmap(_scene_worldfixed(), POS, np.eye(3), cfg)
    h, m = g[0], g[1]

    # 台阶中心 cell (i=8, 中心 x=1.0) 完全覆盖 → 高度 (0.4-0)/0.8 = 0.5
    assert abs(h[8, 5] - 0.5) < 0.05, f"台阶高度应≈0.5, got {h[8, 5]}"
    assert m[8, 5] == 1.0
    # 地面 cell (i=3, 中心 x≈0.075) → 高度≈0、掩码 1
    assert abs(h[3, 5]) < 0.05 and m[3, 5] == 1.0
    # 后方空白 cell (i=0) → 高度 0、掩码 0
    assert h[0, 5] == 0.0 and m[0, 5] == 0.0


# ---------- T6-3 数值上限：0.8m 台阶 → clip 到 1.0 ----------
def test_step_clip_upper(cfg):
    g = hm.build_heightmap(_scene_worldfixed(step_z=0.8), POS, np.eye(3), cfg)
    h, m = g[0], g[1]
    assert abs(h[8, 5] - 1.0) < 0.05, f"0.8m 台阶应 clip 到 1.0, got {h[8, 5]}"
    assert m[8, 5] == 1.0


# ---------- T6-4 方向随车旋转：掩码质心旋转 = 车 yaw 旋转 ----------
def test_step_rotates_with_robot(cfg):
    world = _scene_worldfixed()                     # 同一世界固定台阶

    g0 = hm.build_heightmap(world, POS, np.eye(3), cfg)          # 车朝 yaw=0
    g30 = hm.build_heightmap(world, POS, rot_z(30.0), cfg)       # 车逆时针转 30°

    c0x, c0y = _mask_centroid(g0)                   # 车系前方 → 质心角度 ≈ 0°
    c30x, c30y = _mask_centroid(g30)                # 台阶相对车左偏 → 质心角度 ≈ -30°

    a0 = np.degrees(np.arctan2(c0y, c0x))
    a30 = np.degrees(np.arctan2(c30y, c30x))
    assert abs(a0) < 8.0, f"yaw=0 台阶质心应在正前, got {a0:.1f}°"
    assert abs(a30 - (-30.0)) < 8.0, f"车转 30° 台阶质心应 -30°, got {a30:.1f}°"
    # 高度数值不随旋转变：旋转后台阶区域最高归一化高度仍 0.5
    assert abs(g30[0].max() - 0.5) < 0.05
