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
    # policy cell(7,6) 掩码加权：4 插值角中仅 (18,16) 有效（权重 wx0*wy0=0.25*0.75），
    # 无数据角 mask=0 不参与 → 高度不被稀释，仍为 1.0（旧双线性被稀释成 0.1875）
    assert abs(policy[0, 7, 6] - 1.0) < 0.02


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


# ---------- T11 掩码加权：负 ground_ref 无 phantom 高度（真实地形：地面在车下）----------
def test_mask_weighted_no_phantom_with_neg_ref(cfg):
    """回归：地面在 pos 下方（ground_ref=-0.4）时，无数据格回填 0 归一化后 = 0.5，
    旧双线性会把它混入部分有效格产生假台阶；掩码加权后无数据格不参与 → 无 phantom。"""
    gx = np.array([-0.3, -0.1, 0.1, 0.3]); gy = np.array([-0.3, -0.1, 0.1, 0.3])
    ground = np.stack([np.repeat(gx, 4), np.tile(gy, 4), np.full(16, -0.4)], axis=1) + POS
    obst = world([1.0, 0.1, 0.4])[None, :]          # 0.4m 障碍 → 归一化 (0.4+0.4)/0.8=1.0
    pts = np.vstack([ground, obst])
    g = hm.build_heightmap(pts, POS, np.eye(3), cfg); h, m = g[0], g[1]

    full_h, full_m = hm._aggregate_full(pts - POS, cfg)
    ref = hm._ground_reference_median(full_h, full_m, cfg)
    assert abs(ref - (-0.4)) < 1e-6                   # 负 ground_ref，复现真实地形场景
    assert m[7, 6] == 1.0 and abs(h[7, 6] - 1.0) < 0.02   # 障碍不被无数据角稀释（旧=0.59375）
    assert m[2, 6] == 1.0 and abs(h[2, 6]) < 0.02         # 地面边缘格无 phantom 台阶（旧=0.40625）
    assert np.isfinite(g).all()


# ---------- T12 掩码加权：部分遮挡时障碍高度不被稀释 ----------
def test_mask_weighted_partial_occlusion_height_preserved(cfg):
    """9 点障碍簇全落在 full 格(18,16)（0.4m），其余 3 插值角无数据：
    掩码加权保留 0.5，不被无数据角稀释（旧=0.09375）；相邻全未知格保持 0/0。"""
    gx = np.array([-0.3, -0.1, 0.1, 0.3]); gy = np.array([-0.3, -0.1, 0.1, 0.3])
    ground = np.stack([np.repeat(gx, 4), np.tile(gy, 4), np.zeros(16)], axis=1) + POS
    px = np.array([1.00, 1.02, 1.04]); py = np.array([0.10, 0.12, 0.14])
    cluster = np.array([[x, y, 0.40] for x in px for y in py], dtype=np.float64)
    pts = np.vstack([ground, POS + cluster])
    g = hm.build_heightmap(pts, POS, np.eye(3), cfg); h, m = g[0], g[1]

    assert m[7, 6] == 1.0 and abs(h[7, 6] - 0.5) < 0.02   # 0.4m→0.5，不被稀释（旧=0.09375）
    assert m[7, 8] == 0.0 and h[7, 8] == 0.0             # 相邻全未知格保持 0/0


# ---------- T13 掩码加权：全未知 cell 保持 0/0 + den 守卫无 NaN ----------
def test_mask_weighted_unknown_cell_zero_and_no_nan(cfg):
    """覆盖区角落有数据（full(17,16)）但 4 插值角全无数据 → 掩码 1 + 高度 0（den 守卫，
    不得产生 NaN）；完全无观测格 → 0/0。"""
    gx = np.array([-0.3, -0.1, 0.1, 0.3]); gy = np.array([-0.3, -0.1, 0.1, 0.3])
    ground = np.stack([np.repeat(gx, 4), np.tile(gy, 4), np.zeros(16)], axis=1) + POS
    far = world([0.95, 0.10, 0.0])[None, :]   # 落在 full(17,16)，不在 cell(7,6) 的 4 插值角
    pts = np.vstack([ground, far])
    g = hm.build_heightmap(pts, POS, np.eye(3), cfg); h, m = g[0], g[1]

    assert m[14, 6] == 0.0 and h[14, 6] == 0.0    # 完全无观测格
    assert m[7, 6] == 1.0 and h[7, 6] == 0.0      # 覆盖区角落有效但 4 插值角全无效 → 守卫给 0
    assert np.isfinite(g).all()                   # den==0 不得产生 0/0 NaN
