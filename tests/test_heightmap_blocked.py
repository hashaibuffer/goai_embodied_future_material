#!/usr/bin/env python3
"""tests/test_heightmap_blocked.py — S10 T4 高度图遮挡集成测试（不依赖 ROS）。

验收：雷达前方放一块板子 → Validity 掩码下降，而不是输出完整地形。
核心语义 unknown_validity=0.0：高度图是 mj_multiRay 射线的编码，
被遮挡区域必须掩码 0 / 高度 0，不得用 mesh 真值填充。

做法：读 S10_track_lidar.xml 字符串，在 worldbody 插入一块 box 板子
（雷达正前方 x=1.0，机器人 start y=-2.5），写临时 XML 到模型目录
（保证 include/meshdir 相对路径正确），编译后 mj_multiRay 扫描，
与无板子对照比较阴影区掩码。
运行：python3 -m pytest tests/test_heightmap_blocked.py -v
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "src" / "s10_terrain_perception"))
import heightmap as hm  # noqa: E402

XML = WS / "models" / "mjcf" / "S10_track_lidar.xml"
TMP = XML.parent / "_blocked_test_tmp.xml"

# 板子：雷达正前方 x=1.0（site x=0.223，约 0.8m 前），高 0.7m(z0~0.7)、宽 1.2m(y±0.6)
BOARD = ('  <geom name="block_board" type="box" pos="1.0 -2.5 0.35" '
         'size="0.02 0.6 0.35" rgba="0.4 0.4 0.5 1" group="0"/>\n')

# 阴影区（水平系）：x∈[1.2,3.0] → policy col 8:16；y∈[-0.5,0.5] → row 4:9
SHADOW = (slice(8, 16), slice(4, 9))


def blocked_xml():
    txt = XML.read_text(encoding="utf-8")
    assert txt.count("</worldbody>") == 1
    return txt.replace("</worldbody>", BOARD + "</worldbody>")


def scan(xml_str):
    """加载模型 → start 位姿 → mj_multiRay 全扫，返回 (points, pos, R, 命中板子数)。"""
    TMP.write_text(xml_str, encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(TMP))
    finally:
        TMP.unlink()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qpos[:7] = [0.0, -2.5, 0.2, 1.0, 0.0, 0.0, 0.0]   # 官方 TRACK_START_BASE_POS
    mujoco.mj_forward(model, data)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_front_site")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pnt = data.site_xpos[site].copy()
    az = np.linspace(-90, 90, 181) * np.pi / 180
    el = np.linspace(0, -55, 24) * np.pi / 180
    fan = np.asarray([[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]
                      for e in el for a in az])
    gg = np.asarray([1, 1, 0, 1, 1, 1], dtype=np.uint8)      # 同 lidar.yaml，排除 group2
    R = np.array(data.xmat[body]).reshape(3, 3)
    vec = (R @ fan.T).T
    dist = np.full(len(fan), -1.0)
    geomid = np.full(len(fan), -1, dtype=np.int32)
    mujoco.mj_multiRay(model, data, pnt, vec.reshape(-1), gg, True, body,
                       geomid, dist, None, len(fan), 6.0)
    hit = (geomid >= 0) & (dist <= 6.0) & (dist >= 0.1)
    pts = pnt + vec[hit] * dist[hit, None]
    n_board = int((geomid == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                               "block_board")).sum())
    return pts, data.qpos[:3].copy(), R, n_board


@pytest.fixture(scope="module")
def cfg():
    return hm.load_config()


def test_blocked_mask_drops_instead_of_full_terrain(cfg):
    """板子挡住雷达 → 阴影区掩码/高度全 0，不再输出完整地形。"""
    pts_b, pos, R, n_board = scan(blocked_xml())
    pts_n, _, _, _ = scan(XML.read_text(encoding="utf-8"))

    assert n_board > 0                                  # 板子确实被扫到
    pol_b = hm.build_heightmap(pts_b, pos, R, cfg)
    pol_n = hm.build_heightmap(pts_n, pos, R, cfg)
    mb, mn = pol_b[1], pol_n[1]

    # 阴影区（板子后方 1.2~3m 地形）：有板子必须掩码 0、高度 0
    assert mb[SHADOW].sum() == 0, "遮挡后方掩码应全 0"
    assert pol_b[0][SHADOW].sum() == 0.0, "遮挡后方高度应全 0（未知格）"
    # 无板子对照：同一区域能扫到地形
    assert mn[SHADOW].sum() > 0, "无板子时阴影区应扫到地形（对照）"

    # 板子表面（x≈1.0 → col7）保持有效
    assert mb[7].sum() > 0, "板子表面应保持掩码 1"
    # 总体掩码下降（不只是阴影区）
    assert int(mb.sum()) < int(mn.sum()), "遮挡后总有效格应下降"


def test_blocked_unknown_cells_zero_filled(cfg):
    """被遮挡格 = unknown（validity 0），高度也必须填 0 而非 mesh 真值。"""
    pts_b, pos, R, _ = scan(blocked_xml())
    pol_b = hm.build_heightmap(pts_b, pos, R, cfg)
    h, mask = pol_b[0], pol_b[1]
    # 所有掩码 0 的格高度必须为 0（unknown_height_fill）
    assert np.allclose(h[mask == 0], 0.0)
    # 掩码只有 0/1
    assert set(np.unique(mask).tolist()) <= {0.0, 1.0}
