#!/usr/bin/env python3
"""tests/test_overlay_filter.py — S10 T4 overlay(group2) 过滤集成测试（不依赖 ROS）。

验收：overlay 过滤生效 → 平地上没有假障碍墙。
场景：track_overlay.xml 的绿色轨道边线胶囊（track_segment_000 从 start 前缘斜向延伸，
r=0.07、中心 z 0~0.475m）都是 group 2。lidar.yaml exclude_geom_groups=[2]
要求雷达不命中它们 → 高度图上不得出现这些胶囊造成的假墙。

做法：同一 start 场景，分别用过滤 geomgroup=[1,1,0,1,1,1]（lidar_node 行为）
与全开 geomgroup=[1,1,1,1,1,1] 扫描，比较 group2 命中数与前向高度图。
运行：python3 -m pytest tests/test_overlay_filter.py -v
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "src" / "s10_terrain_perception"))
import heightmap as hm  # noqa: E402

XML = WS / "models" / "mjcf" / "S10_track_lidar.xml"
TMP = XML.parent / "_ovl_test_tmp.xml"

GG_FILTERED = np.asarray([1, 1, 0, 1, 1, 1], dtype=np.uint8)   # lidar.yaml exclude group2
GG_ALL = np.asarray([1, 1, 1, 1, 1, 1], dtype=np.uint8)        # 对照组（不过滤）


def scan(gg):
    """start 位姿全扫，返回 (points, pos, R, group 分布)。"""
    TMP.write_text(
        XML.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(TMP))
    finally:
        TMP.unlink()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qpos[:7] = [0.0, -2.5, 0.2, 1.0, 0.0, 0.0, 0.0]      # 官方 TRACK_START_BASE_POS
    mujoco.mj_forward(model, data)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_front_site")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pnt = data.site_xpos[site].copy()
    az = np.linspace(-90, 90, 181) * np.pi / 180
    el = np.linspace(0, -55, 24) * np.pi / 180
    fan = np.asarray([[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]
                      for e in el for a in az])
    R = np.array(data.xmat[body]).reshape(3, 3)
    vec = (R @ fan.T).T
    dist = np.full(len(fan), -1.0)
    gid = np.full(len(fan), -1, dtype=np.int32)
    mujoco.mj_multiRay(model, data, pnt, vec.reshape(-1), gg, True, body,
                       gid, dist, None, len(fan), 6.0)
    hit = (gid >= 0) & (dist <= 6.0) & (dist >= 0.1)
    pts = pnt + vec[hit] * dist[hit, None]
    groups = np.array([model.geom(int(i)).group for i in gid[hit]])
    return pts, data.qpos[:3].copy(), R, groups


def _setup():
    import pytest
    cfg = hm.load_config()
    pts_f, pos, R, gf = scan(GG_FILTERED)
    pts_a, _, _, ga = scan(GG_ALL)
    pol_f = hm.build_heightmap(pts_f, pos, R, cfg)
    pol_a = hm.build_heightmap(pts_a, pos, R, cfg)
    return gf, ga, pol_f, pol_a


def test_overlay_group2_excluded():
    """过滤 geomgroup 关闭 group2：命中不含任何 group2（场景确有 overlay，对照组命中 41）。"""
    gf, ga, _, _ = _setup()
    assert int((gf == 2).sum()) == 0, "过滤后不得命中任何 group2 overlay"
    assert int((ga == 2).sum()) > 0, "对照组（不过滤）应能命中 overlay，证明过滤有意义"
    assert int((gf == 0).sum()) > 0, "过滤后仍命中真实地形（group0）"


def test_no_fake_wall_on_flat_ground():
    """平地上没有假障碍墙：高度图不被无数据格产生假墙（掩码加权修复后的正确行为）。

    旧双线性：胶囊命中集中在 full 格 col10（x∈[0.2,0.3]），不在 policy cell 的 4 个
    插值角内；该 cell 高度由无数据格回填 0 归一化（负 ground_ref 时）≈0.25 决定 → 假墙。
    掩码加权修复后无数据格不参与 → 高度≈0，假墙消除（正是修复目标）。
    """
    _, _, pol_f, pol_a = _setup()
    h_f, h_a = pol_f[0], pol_a[0]
    # 过滤与否，高度图都不应出现假墙（旧代码有 0.25 污染假墙，现应消除）
    assert h_a.max() < 0.15, f"对照组不应有假墙(max={h_a.max():.3f})"
    assert h_f.max() < 0.15, f"过滤后不应有假墙(max={h_f.max():.3f})"
