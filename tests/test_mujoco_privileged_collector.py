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
    DEFAULT_ROBOT, PrivilegedHeightScanner, S10PolicyState,
    assemble_official_57, assemble_teacher_1413, decode_action_norm)


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
    goal_pos, goal_vel = decode_action_norm(np.zeros(16))
    np.testing.assert_allclose(goal_pos[[0, 1, 2, 4, 5, 6]], [0, -.3, .6, 0, -.3, .6])
    np.testing.assert_allclose(goal_vel, 0)


def test_d_priv_roundtrip_and_fake_source(tmp_path):
    recorder = DPrivRecorder(teacher_model=None, teacher_source="fake_mujoco_smoke")
    recorder.append(
        student_obs=np.zeros(441), teacher_action_norm=np.zeros(16), command_raw=np.zeros(3),
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
