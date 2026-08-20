from __future__ import annotations
import sys
from pathlib import Path
import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "s10_terrain_perception"))
from d_priv_dataset import DPrivRecorder, validate_d_priv
from mujoco_lidar import MujocoLidarScanner
from mujoco_teacher import (
    DEFAULT_ROBOT, JOINT_INIT_RAW, PrivilegedHeightScanner, S10PolicyState,
    assemble_official_57, assemble_teacher_1413, decode_action_raw,
    run_stand_up, stand_up_target_raw)


def plane_model():
    xml = """
    <mujoco><worldbody>
      <geom name="ground" type="plane" size="10 10 .1" group="0"/>
      <body name="base_link" pos="0 0 .4">
        <freejoint/><site name="lidar_front_site" pos="0 0 0"/>
        <geom type="sphere" size=".1" group="1"/>
      </body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    return model, data


def test_privileged_grid_matches_isaac_shape_order_and_flat_height():
    model, data = plane_model()
    scanner = PrivilegedHeightScanner(model)
    height, hit, geom_ids = scanner.scan(data, [0, 0, .4], np.eye(3))
    assert scanner.local_xy.shape == (1353, 2)
    np.testing.assert_allclose(scanner.local_xy[0], [-.8, -1.6])
    np.testing.assert_allclose(scanner.local_xy[1], [-.7, -1.6])
    np.testing.assert_allclose(scanner.local_xy[41], [-.8, -1.5])
    assert hit.all() and np.all(geom_ids >= 0)
    np.testing.assert_allclose(height, -.1, atol=2e-6)


def test_pure_lidar_filters_robot_and_returns_world_points():
    model, data = plane_model()
    cfg = {
        "site_name": "lidar_front_site", "body_exclude": "base_link",
        "azimuth_deg": [0, 0], "azimuth_beams": 1,
        "elevation_deg": [-90, -90], "elevation_beams": 1,
        "range_min": 0, "cutoff": 2, "geomgroup": [1, 0, 0, 1, 1, 1],
        "flg_static": True, "range_noise_std_m": 0, "dropout_probability": 0,
    }
    scan = MujocoLidarScanner(model, cfg).scan(data, [0, 0, .4], np.eye(3))
    assert scan.raw_hit_count == 1
    np.testing.assert_allclose(scan.points_w, [[0, 0, 0]], atol=1e-7)


def test_observation_and_action_contract_dimensions():
    state = S10PolicyState(
        np.asarray([0, 0, .4]), np.eye(3), np.asarray([1, 2, 3], np.float32),
        np.asarray([1, -.4, .8], np.float32), DEFAULT_ROBOT.copy(), np.zeros(16, np.float32))
    proprio = assemble_official_57(state, [.5, -.2, .3], np.zeros(16))
    assert proprio.shape == (57,)
    np.testing.assert_allclose(proprio[:9], [.25, -.1, .2, 0, 0, -1, .5, -.2, .3])
    assert assemble_teacher_1413(state, proprio, np.zeros(1353)).shape == (1413,)
    goal_pos, goal_vel = decode_action_raw(np.zeros(16))
    np.testing.assert_allclose(goal_pos[[0, 1, 2, 4, 5, 6]], [0, -.3, .6, 0, -.3, .6])
    np.testing.assert_allclose(goal_vel, 0)


def test_d_priv_roundtrip_and_fake_source(tmp_path):
    recorder = DPrivRecorder(teacher_model=None, teacher_source="fake_mujoco_smoke")
    recorder.append(
        student_obs=np.zeros(441), teacher_action_raw=np.zeros(16), command_raw=np.zeros(3),
        waypoint_id=-1, terrain_id="plane", episode_id=0, step_id=0,
        base_pose_wxyz=[0, 0, .4, 1, 0, 0, 0], privileged_hit_fraction=1.0)
    path = recorder.write(tmp_path / "fake.npz")
    assert validate_d_priv(path) == 1
    with np.load(path, allow_pickle=False) as data:
        assert data["teacher_source"].tolist() == ["fake_mujoco_smoke"]
        assert data["student_obs"].shape == (1, 441)


def test_wrong_teacher_width_is_rejected():
    state = S10PolicyState(
        np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3), np.zeros(16), np.zeros(16))
    with pytest.raises(ValueError):
        assemble_teacher_1413(state, np.zeros(57), np.zeros(100))


# ---------------------------------------------------------------------------
# StandUp 测试
# ---------------------------------------------------------------------------

def test_stand_up_target_raw_equals_default_robot():
    """stand_up_target_raw() 应直接返回 DEFAULT_ROBOT（raw 空间），
    不得再经过 published_targets_to_raw 做二次变换，否则会超出 MJCF 关节限位。"""
    target = stand_up_target_raw()
    np.testing.assert_allclose(target, DEFAULT_ROBOT, atol=1e-6)


def test_stand_up_target_within_joint_limits():
    """起立目标的腿关节角必须在 MJCF 硬限位内。"""
    xml_path = ROOT / "models" / "mjcf" / "S10_track_lidar.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    target = stand_up_target_raw()
    for leg in range(4):
        base = 4 * leg
        for dof in range(3):  # hipx, hipy, knee（轮关节是速度控制，跳过）
            jnt_idx = 1 + base + dof  # world joint 0 = freejoint
            lo, hi = model.jnt_range[jnt_idx]
            val = target[base + dof]
            assert lo <= val <= hi, (
                f"leg{leg} dof{dof}: target={val:.4f} out of [{lo:.4f}, {hi:.4f}]")


S10_TRACK_XML = ROOT / "models" / "mjcf" / "S10_track_lidar.xml"


@pytest.mark.skipif(not S10_TRACK_XML.exists(), reason="MJCF not present")
def test_run_stand_up_reaches_standing_height():
    """run_stand_up 结束后 base_z 应 ≥ 0.35m（官方 stand_height=0.48，
    地面支撑下稳态约 0.40，留余量至 0.35 避免误报）。"""
    model = mujoco.MjModel.from_xml_path(str(S10_TRACK_XML))
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    # 用默认 start 参数初始化：坐姿 JOINT_INIT_RAW，离地 0.20m
    data.qpos[:3] = [0.0, -2.5, 0.20]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:23] = JOINT_INIT_RAW
    mujoco.mj_forward(model, data)

    z = run_stand_up(model, data, base_id)
    assert z >= 0.35, f"stand-up failed: base_z={z:.3f} < 0.35"


def test_d_priv_write_returns_npz_path(tmp_path):
    """DPrivRecorder.write() 应返回实际存在的 .npz 文件路径。"""
    recorder = DPrivRecorder(teacher_model=None, teacher_source="test")
    recorder.append(
        student_obs=np.zeros(441), teacher_action_raw=np.zeros(16),
        command_raw=np.zeros(3), waypoint_id=0, terrain_id="test",
        episode_id=0, step_id=0, base_pose_wxyz=np.zeros(7),
        privileged_hit_fraction=1.0)
    # 故意用非 .npz 后缀，模拟 CLI 的 --output foo.h5 用法
    path = recorder.write(tmp_path / "shard.h5")
    assert path.exists(), f"write() returned {path} but file not found"
    assert path.suffix == ".npz", f"expected .npz suffix, got {path.suffix}"
    assert validate_d_priv(path) == 1


def test_dagger_collector_keeps_teacher_labels_separate_from_behavior():
    source = (ROOT / "scripts" / "collect_mujoco_d_priv.py").read_text(
        encoding="utf-8")
    assert "teacher_action = teacher_policy(teacher_obs)" in source
    assert "rollout_policy(student_obs) if rollout_policy else teacher_action" in source
    assert "teacher_action_raw=teacher_action" in source
    assert "last_action = behavior_action" in source
    assert "decode_action_raw(behavior_action)" in source
