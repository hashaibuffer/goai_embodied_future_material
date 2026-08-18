#!/usr/bin/env python3
"""tests/test_heightmap.py — S10 T4 高度图编码单元测试（不依赖 ROS）。

build_heightmap 输入是 world 系命中点（lidar_node 直接喂扫描结果），
内部经 yaw-only 旋转转到水平系。因此本测试的所有构造点都用 POS+局部坐标生成。

验收：网格方向、数值、掩码逻辑、展平、旋转。
运行：python3 -m pytest tests/test_heightmap.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "s10_terrain_perception"))
import heightmap as hm  # noqa: E402

POS = np.array([0.0, -2.5, 0.2])          # 官方 TRACK_START_BASE_POS


def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def world(local):
    """水平系局部坐标 → world 系（R=I 时 world = pos + local）。"""
    return POS + np.asarray(local)


@pytest.fixture(scope="module")
def cfg():
    c = hm.load_config()
    assert c["policy_nx"] == 16 and c["policy_ny"] == 12
    return c


def _ground_obst():
    """机器人附近地面点(z=0) + 前方障碍(z=0.8)，world 系。"""
    gx = np.array([-0.3, -0.1, 0.1, 0.3])
    gy = np.array([-0.3, -0.1, 0.1, 0.3])
    ground = np.stack([np.repeat(gx, 4), np.tile(gy, 4), np.zeros(16)], axis=1)
    ground = ground + POS                                 # world = pos + 局部
    obst = world([1.0, 0.1, 0.8])[None, :]                # 局部 (1.0,0.1,0.8)
    return ground, obst


# ---------- T1 yaw-only 旋转 ----------
def test_yaw_only_rotation():
    R = rot_z(30.0) @ rot_x(10.0)                         # 含 roll 倾斜
    R_yaw = hm.yaw_only_rotation(R)
    np.testing.assert_allclose(R_yaw, rot_z(30.0), atol=1e-9)   # 只去 yaw
    # 往返验证：构造 body 系点，水平系投影后落在已知目标
    p_h_target = np.array([0.5, 0.5, 0.3])
    p_body = R.T @ (R_yaw @ p_h_target)
    world_pt = POS + R @ p_body
    p_h = R_yaw.T @ (world_pt - POS)
    np.testing.assert_allclose(p_h, p_h_target, atol=1e-9)     # roll 不影响 x/y 索引
    np.testing.assert_allclose(p_h[2], p_h_target[2], atol=1e-9)  # z 世界垂直


# ---------- T2/T3 地面参考 valid_median + 归一化 ----------
def test_ground_reference_and_normalize(cfg):
    ground, obst = _ground_obst()
    full_h, full_m = hm._aggregate_full(ground - POS, cfg)    # _aggregate_full 吃水平系
    ref = hm._ground_reference_median(full_h, full_m, cfg)
    assert abs(ref) < 1e-6                                    # 附近地面 median == 0
    assert float(full_m[5:12, 11:19].sum()) == 16.0           # 16 个地面点各占一格

    policy = hm.build_heightmap(np.vstack([ground, obst]), POS, np.eye(3), cfg)
    # 障碍 full 格(18,16) 归一化 = clip((0.8-0)/0.8)=1.0；
    # policy cell(7,6) 中心(1.075,0.125) 插值系数 wx0*wy0=0.25*0.75 → 0.1875
    assert abs(policy[0, 7, 6] - 0.1875) < 0.02


# ---------- T4 网格索引方向 + 边界 ----------
def test_grid_index_bounds(cfg):
    pts = np.array([
        [-0.8, 0.0, 0.0],   # x 左闭 → 有效 col0
        [3.2, 0.0, 0.0],    # x 右开 → 丢弃
        [0.0, -1.5, 0.0],   # y 左闭 → 有效 row0
        [0.0, 1.5, 0.0],    # y 右开 → 丢弃
        [0.5, 0.0, 0.0],    # 常规 → col13 row15
    ])
    _, fm = hm._aggregate_full(pts, cfg)
    assert fm[0, 15] == 1.0          # x=-0.8 左闭 col0 有效
    assert fm[39, 15] == 0.0         # x=3.2 右开丢弃
    assert fm[8, 0] == 1.0           # [0,-1.5] → col8 row0 有效
    assert fm[8, 29] == 0.0          # [0,1.5] → row30 右开丢弃
    assert fm[13, 15] == 1.0         # 常规点 [0.5,0] → col13 row15


# ---------- T5 格内 max z ----------
def test_max_z_per_cell(cfg):
    dup = np.array([[0.55, 0.05, 0.3], [0.55, 0.05, 0.7]])   # 水平系，同格
    fh2, _ = hm._aggregate_full(dup, cfg)
    assert abs(float(fh2[13, 15]) - 0.7) < 1e-6              # 同格取最高 z=0.7


def test_negative_z_preserved(cfg):
    """回归：地面在 pos.z 下方时命中点 z 为负，max 初值必须 -inf 而非 0，否则负 z 全丢成 0。"""
    dup = np.array([[0.55, 0.05, -0.2], [0.55, 0.05, -0.1]])  # 水平系，同格，负 z
    fh2, fm2 = hm._aggregate_full(dup, cfg)
    assert abs(float(fh2[13, 15]) - (-0.1)) < 1e-6            # 同格取最高(负)z=-0.1
    assert fm2[13, 15] == 1.0
    assert fh2[0, 0] == 0.0 and fm2[0, 0] == 0.0              # 无命中格仍为 0


# ---------- T6/T7 掩码 + bilinear_center 降采样 ----------
def test_mask_and_downsample(cfg):
    ground, obst = _ground_obst()
    gpol = hm.build_heightmap(np.vstack([ground, obst]), POS, np.eye(3), cfg)

    # 地面格 cell(3,6)：高度≈0 + 掩码 1
    assert abs(gpol[0, 3, 6]) < 0.01 and gpol[1, 3, 6] == 1.0
    # 空白格 cell(15,6)：高度 0 + 掩码 0
    assert gpol[0, 15, 6] == 0.0 and gpol[1, 15, 6] == 0.0
    # 障碍格 cell(7,6)：掩码 1（cell 覆盖任一有效）
    assert gpol[1, 7, 6] == 1.0


# ---------- T8 展平 channel-major ----------
def test_flatten_channel_major(cfg):
    ground, obst = _ground_obst()
    gpol = hm.build_heightmap(np.vstack([ground, obst]), POS, np.eye(3), cfg)

    assert gpol.shape == (2, 16, 12)
    assert gpol.dtype == np.float32
    n = 16 * 12
    flat = gpol.reshape(-1)
    assert np.array_equal(flat[:n], gpol[0].reshape(-1))     # 前 192 高度
    assert np.array_equal(flat[n:], gpol[1].reshape(-1))     # 后 192 掩码


# ---------- T9 空输入 ----------
def test_empty_input(cfg):
    empty = hm.build_heightmap(np.zeros((0, 3)), POS, np.eye(3), cfg)
    assert empty.shape == (2, 16, 12)
    assert float(empty.max()) == 0.0


# ---------- T10 官方配置加载 ----------
def test_config_loading():
    c = hm.load_config()
    assert c["topic"] == "/S10_HEIGHTMAP"
    assert c["frame"] == "robot_horizontal"
    assert (c["full_nx"], c["full_ny"]) == (40, 30)
    assert (c["policy_nx"], c["policy_ny"]) == (16, 12)
    assert abs(c["height_divisor_m"] - 0.80) < 1e-9
    assert c["clip_m"] == (-0.40, 0.80)
    assert c["ground_ref_x_range"] == (-0.40, 0.40)
    assert c["ground_ref_y_range"] == (-0.40, 0.40)
    assert c["unknown_validity"] == 0.0
    assert c["downsample"] == "bilinear_center"
