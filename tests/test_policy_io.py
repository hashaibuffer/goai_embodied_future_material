"""Lock the official 57-D observation and 16-D action decode.

Reference constants and the cpp_* helpers below are a line-for-line reading of
S10PolicyRunner / RpyToRm. PolicyContract must match them, not the other way around.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from s10_terrain_policy import PolicyContract

OFFICIAL_OMEGA_SCALE = 0.25
OFFICIAL_DOF_VEL_SCALE = 0.05
OFFICIAL_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float64)

OFFICIAL_ROBOT_ORDER = [
    "fl_hipx_joint",
    "fl_hipy_joint",
    "fl_knee_joint",
    "fl_wheel_joint",
    "fr_hipx_joint",
    "fr_hipy_joint",
    "fr_knee_joint",
    "fr_wheel_joint",
    "hl_hipx_joint",
    "hl_hipy_joint",
    "hl_knee_joint",
    "hl_wheel_joint",
    "hr_hipx_joint",
    "hr_hipy_joint",
    "hr_knee_joint",
    "hr_wheel_joint",
]

OFFICIAL_POLICY_ORDER = [
    "fl_hipx_joint",
    "fl_hipy_joint",
    "fl_knee_joint",
    "fr_hipx_joint",
    "fr_hipy_joint",
    "fr_knee_joint",
    "hl_hipx_joint",
    "hl_hipy_joint",
    "hl_knee_joint",
    "hr_hipx_joint",
    "hr_hipy_joint",
    "hr_knee_joint",
    "fl_wheel_joint",
    "fr_wheel_joint",
    "hl_wheel_joint",
    "hr_wheel_joint",
]

OFFICIAL_ROBOT2POLICY = np.array(
    [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15], dtype=np.int64
)
OFFICIAL_POLICY2ROBOT = np.array(
    [0, 1, 2, 12, 3, 4, 5, 13, 6, 7, 8, 14, 9, 10, 11, 15], dtype=np.int64
)

OFFICIAL_DEFAULT_POSE_POLICY = np.array(
    [0.0, -0.3, 0.6, 0.0, -0.3, 0.6, 0.0, 0.3, -0.6, 0.0, 0.3, -0.6, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
OFFICIAL_DEFAULT_POSE_ROBOT = np.array(
    [0.0, -0.3, 0.6, 0.0, 0.0, -0.3, 0.6, 0.0, 0.0, 0.3, -0.6, 0.0, 0.0, 0.3, -0.6, 0.0],
    dtype=np.float64,
)
OFFICIAL_ACTION_SCALE_ROBOT = np.array(
    [0.125, 0.25, 0.25, 5.0] * 4,
    dtype=np.float64,
)
OFFICIAL_KP = np.array([80.0, 80.0, 80.0, 0.0] * 4, dtype=np.float64)
OFFICIAL_KD = np.array([2.0, 2.0, 2.0, 0.6] * 4, dtype=np.float64)

# Identity pose, default joints, unit-ish omega/command. Computed from cpp_assemble_57.
GOLDEN_IDENTITY_57 = np.array(
    [
        0.25,
        -0.10,
        0.20,
        0.0,
        0.0,
        -1.0,
        0.5,
        -0.2,
        0.3,
        *np.zeros(16),
        *np.zeros(16),
        *np.zeros(16),
    ],
    dtype=np.float64,
)


def _cpp_permutation(source: list[str], dest: list[str]) -> list[int]:
    index = {name: i for i, name in enumerate(source)}
    return [index[name] for name in dest]


def _cpp_rpy_to_rm(rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy_rad, dtype=np.float64).reshape(3)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def cpp_assemble_57(
    *,
    base_omega,
    base_rot_mat,
    command,
    joint_pos_robot,
    joint_vel_robot,
    last_action,
) -> np.ndarray:
    omega = np.asarray(base_omega, dtype=np.float64).reshape(3) * OFFICIAL_OMEGA_SCALE
    projected = np.asarray(base_rot_mat, dtype=np.float64).reshape(3, 3).T @ OFFICIAL_GRAVITY
    cmd = np.asarray(command, dtype=np.float64).reshape(3)
    joint_pos = np.asarray(joint_pos_robot, dtype=np.float64).reshape(16)[OFFICIAL_ROBOT2POLICY]
    joint_vel = np.asarray(joint_vel_robot, dtype=np.float64).reshape(16)[OFFICIAL_ROBOT2POLICY]
    joint_pos[12:16] = 0.0
    joint_pos = joint_pos - OFFICIAL_DEFAULT_POSE_POLICY
    joint_vel = joint_vel * OFFICIAL_DOF_VEL_SCALE
    last = np.asarray(last_action, dtype=np.float64).reshape(16)
    return np.concatenate([omega, projected, cmd, joint_pos, joint_vel, last])


def cpp_reconstruct_action(action_norm_policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action_norm_policy, dtype=np.float64).reshape(16)
    physical = action[OFFICIAL_POLICY2ROBOT] * OFFICIAL_ACTION_SCALE_ROBOT
    physical = physical + OFFICIAL_DEFAULT_POSE_ROBOT
    goal_pos = np.zeros(16, dtype=np.float64)
    goal_vel = np.zeros(16, dtype=np.float64)
    for leg in range(4):
        base = 4 * leg
        goal_pos[base : base + 3] = physical[base : base + 3]
        goal_vel[base + 3] = physical[base + 3]
    return goal_pos, goal_vel


@pytest.fixture(scope="module")
def contract(config_dir) -> PolicyContract:
    return PolicyContract.load(config_dir)


def test_yaml_files_exist(config_dir):
    for name in ("track.yaml", "lidar.yaml", "heightmap.yaml", "policy.yaml"):
        assert (config_dir / name).is_file()


def test_yaml_constants_match_official_runner(contract: PolicyContract):
    policy = contract.policy
    assert policy["proprio_dim"] == 57
    assert policy["obs_dim"] == 441
    assert policy["action_dim"] == 16
    assert policy["decimation"] == 4
    assert policy["omega_scale"] == OFFICIAL_OMEGA_SCALE
    assert policy["dof_vel_scale"] == OFFICIAL_DOF_VEL_SCALE
    assert policy["robot_order"] == OFFICIAL_ROBOT_ORDER
    assert policy["policy_order"] == OFFICIAL_POLICY_ORDER
    np.testing.assert_array_equal(contract.robot2policy_idx, OFFICIAL_ROBOT2POLICY)
    np.testing.assert_array_equal(contract.policy2robot_idx, OFFICIAL_POLICY2ROBOT)
    np.testing.assert_allclose(contract.default_pose_policy, OFFICIAL_DEFAULT_POSE_POLICY)
    np.testing.assert_allclose(contract.default_pose_robot, OFFICIAL_DEFAULT_POSE_ROBOT)
    np.testing.assert_allclose(contract.action_scale_robot, OFFICIAL_ACTION_SCALE_ROBOT)
    np.testing.assert_allclose(contract.kp_robot, OFFICIAL_KP)
    np.testing.assert_allclose(contract.kd_robot, OFFICIAL_KD)
    assert policy["imu_rpy_in_ros"] == "degrees"
    assert policy["imu_rpy_in_policy"] == "radians"
    assert policy["onnx"]["input_name"] == "obs"
    assert policy["onnx"]["output_name"] == "actions"
    assert policy["onnx"]["input_shape"] == [1, 441]


def test_permutation_matches_generate_permutation(contract: PolicyContract):
    assert _cpp_permutation(OFFICIAL_ROBOT_ORDER, OFFICIAL_POLICY_ORDER) == list(
        OFFICIAL_ROBOT2POLICY
    )
    assert _cpp_permutation(OFFICIAL_POLICY_ORDER, OFFICIAL_ROBOT_ORDER) == list(
        OFFICIAL_POLICY2ROBOT
    )
    assert list(contract.robot2policy_idx) == list(OFFICIAL_ROBOT2POLICY)
    assert list(contract.policy2robot_idx) == list(OFFICIAL_POLICY2ROBOT)


def test_command_limits_agree_across_yaml(config_dir, contract: PolicyContract):
    track = yaml.safe_load((config_dir / "track.yaml").read_text(encoding="utf-8"))
    assert track["command"] == {
        "vx": [-1.0, 1.0],
        "vy": [-0.6, 0.6],
        "wz": [-1.0, 1.0],
    }
    assert contract.policy_cfg["command"]["vx"] == track["command"]["vx"]
    assert contract.policy_cfg["command"]["vy"] == track["command"]["vy"]
    assert contract.policy_cfg["command"]["wz"] == track["command"]["wz"]
    assert contract.policy_cfg["command"]["student_command"] == "cmd_raw"


def test_obs_slices_cover_441_without_gaps(contract: PolicyContract):
    ordered = [
        "base_omega",
        "projected_gravity",
        "cmd_raw",
        "joint_pos",
        "joint_vel",
        "last_action",
        "height_normalized",
        "validity_mask",
    ]
    cursor = 0
    for name in ordered:
        start, end = contract.obs_slices[name]
        assert start == cursor
        assert end > start
        cursor = end
    assert cursor == 441
    assert contract.obs_slices["proprio"] == (0, 57)
    assert contract.obs_slices["heightmap"] == (57, 441)


def test_identity_pose_default_joints_golden_57(contract: PolicyContract):
    omega = np.array([1.0, -0.4, 0.8])
    command = np.array([0.5, -0.2, 0.3])
    joint_pos = OFFICIAL_DEFAULT_POSE_ROBOT.copy()
    joint_vel = np.zeros(16)
    last_action = np.zeros(16)
    rot = np.eye(3)

    expected = cpp_assemble_57(
        base_omega=omega,
        base_rot_mat=rot,
        command=command,
        joint_pos_robot=joint_pos,
        joint_vel_robot=joint_vel,
        last_action=last_action,
    )
    np.testing.assert_allclose(expected, GOLDEN_IDENTITY_57, atol=1e-7)

    got = contract.assemble_proprio_57(
        base_omega_rad=omega,
        base_rot_mat=rot,
        command_raw=command,
        joint_pos_robot=joint_pos,
        joint_vel_robot=joint_vel,
        last_action_norm=last_action,
    )
    np.testing.assert_allclose(got, GOLDEN_IDENTITY_57, atol=1e-6)


def test_zero_joints_subtract_default_pose(contract: PolicyContract):
    expected = cpp_assemble_57(
        base_omega=np.zeros(3),
        base_rot_mat=np.eye(3),
        command=np.zeros(3),
        joint_pos_robot=np.zeros(16),
        joint_vel_robot=np.zeros(16),
        last_action=np.zeros(16),
    )
    # Wheels already zero; remaining entries are -default_pose_policy.
    np.testing.assert_allclose(expected[9:25], -OFFICIAL_DEFAULT_POSE_POLICY)

    got = contract.assemble_proprio_57(
        base_omega_rad=np.zeros(3),
        base_rot_mat=np.eye(3),
        command_raw=np.zeros(3),
        joint_pos_robot=np.zeros(16),
        joint_vel_robot=np.zeros(16),
        last_action_norm=np.zeros(16),
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_wheel_position_is_zeroed_but_wheel_velocity_is_kept(contract: PolicyContract):
    joint_pos = OFFICIAL_DEFAULT_POSE_ROBOT.copy()
    joint_pos[3] = 1.25  # fl_wheel in robot_order
    joint_vel = np.zeros(16)
    joint_vel[3] = 10.0

    expected = cpp_assemble_57(
        base_omega=np.zeros(3),
        base_rot_mat=np.eye(3),
        command=np.zeros(3),
        joint_pos_robot=joint_pos,
        joint_vel_robot=joint_vel,
        last_action=np.zeros(16),
    )
    # policy index 12 is fl_wheel
    assert expected[9 + 12] == 0.0
    np.testing.assert_allclose(expected[25 + 12], 10.0 * OFFICIAL_DOF_VEL_SCALE)

    got = contract.assemble_proprio_57(
        base_omega_rad=np.zeros(3),
        base_rot_mat=np.eye(3),
        command_raw=np.zeros(3),
        joint_pos_robot=joint_pos,
        joint_vel_robot=joint_vel,
        last_action_norm=np.zeros(16),
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_single_leg_joint_lands_in_policy_order_slot(contract: PolicyContract):
    joint_pos = OFFICIAL_DEFAULT_POSE_ROBOT.copy()
    joint_pos[4] = OFFICIAL_DEFAULT_POSE_ROBOT[4] + 0.2  # fr_hipx robot index 4
    expected = cpp_assemble_57(
        base_omega=np.zeros(3),
        base_rot_mat=np.eye(3),
        command=np.zeros(3),
        joint_pos_robot=joint_pos,
        joint_vel_robot=np.zeros(16),
        last_action=np.zeros(16),
    )
    # policy_order index of fr_hipx is 3
    np.testing.assert_allclose(expected[9 + 3], 0.2)

    got = contract.assemble_proprio_57(
        base_omega_rad=np.zeros(3),
        base_rot_mat=np.eye(3),
        command_raw=np.zeros(3),
        joint_pos_robot=joint_pos,
        joint_vel_robot=np.zeros(16),
        last_action_norm=np.zeros(16),
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_imu_degrees_match_radians_and_reject_raw_degree_rpy(contract: PolicyContract):
    rpy_deg = np.array([90.0, 0.0, 0.0])
    rpy_rad = rpy_deg * np.pi / 180.0
    rot = _cpp_rpy_to_rm(rpy_rad)
    # +90 deg roll: world -Z is body -Y.
    np.testing.assert_allclose(rot.T @ OFFICIAL_GRAVITY, [0.0, -1.0, 0.0], atol=1e-7)

    kwargs = dict(
        base_omega_rad=np.zeros(3),
        command_raw=np.zeros(3),
        joint_pos_robot=OFFICIAL_DEFAULT_POSE_ROBOT,
        joint_vel_robot=np.zeros(16),
        last_action_norm=np.zeros(16),
    )
    from_deg = contract.assemble_proprio_57(imu_rpy_deg=rpy_deg, **kwargs)
    from_rad = contract.assemble_proprio_57(base_rpy_rad=rpy_rad, **kwargs)
    from_mat = contract.assemble_proprio_57(base_rot_mat=rot, **kwargs)
    np.testing.assert_allclose(from_deg, from_rad, atol=1e-6)
    np.testing.assert_allclose(from_deg, from_mat, atol=1e-6)
    np.testing.assert_allclose(from_deg[3:6], [0.0, -1.0, 0.0], atol=1e-6)

    wrong = contract.assemble_proprio_57(base_rpy_rad=rpy_deg, **kwargs)
    assert not np.allclose(wrong[3:6], from_deg[3:6], atol=1e-2)


def test_forgetting_omega_or_vel_scale_is_detectable(contract: PolicyContract):
    omega = np.array([2.0, 0.0, -1.0])
    vel = np.zeros(16)
    vel[1] = 8.0
    got = contract.assemble_proprio_57(
        base_omega_rad=omega,
        base_rot_mat=np.eye(3),
        command_raw=np.zeros(3),
        joint_pos_robot=OFFICIAL_DEFAULT_POSE_ROBOT,
        joint_vel_robot=vel,
        last_action_norm=np.zeros(16),
    )
    np.testing.assert_allclose(got[0:3], omega * OFFICIAL_OMEGA_SCALE)
    # fl_hipy is policy index 1
    np.testing.assert_allclose(got[25 + 1], 8.0 * OFFICIAL_DOF_VEL_SCALE)
    assert not np.allclose(got[0:3], omega)
    assert not np.allclose(got[25 + 1], 8.0)


def test_last_action_is_unscaled_onnx_output(contract: PolicyContract):
    last = np.linspace(-1.0, 1.0, 16)
    got = contract.assemble_proprio_57(
        base_omega_rad=np.zeros(3),
        base_rot_mat=np.eye(3),
        command_raw=np.zeros(3),
        joint_pos_robot=OFFICIAL_DEFAULT_POSE_ROBOT,
        joint_vel_robot=np.zeros(16),
        last_action_norm=last,
    )
    np.testing.assert_allclose(got[41:57], last)


def test_student_obs_keeps_cmd_raw_not_cmd_terrain(contract: PolicyContract):
    cmd_raw = np.array([0.8, 0.0, 0.2])
    cmd_terrain = np.array([0.3, 0.1, 0.0])
    proprio = contract.assemble_proprio_57(
        base_omega_rad=np.zeros(3),
        base_rot_mat=np.eye(3),
        command_raw=cmd_raw,
        joint_pos_robot=OFFICIAL_DEFAULT_POSE_ROBOT,
        joint_vel_robot=np.zeros(16),
        last_action_norm=np.zeros(16),
    )
    height = np.zeros((16, 12), dtype=np.float32)
    validity = np.ones((16, 12), dtype=np.float32)
    student = contract.assemble_student_441(proprio, height, validity)
    assert student.shape == (441,)
    np.testing.assert_allclose(contract.slice_obs(student, "cmd_raw"), cmd_raw)
    assert not np.allclose(contract.slice_obs(student, "cmd_raw"), cmd_terrain)


def test_command_is_clipped_to_keyboard_range(contract: PolicyContract):
    got = contract.assemble_proprio_57(
        base_omega_rad=np.zeros(3),
        base_rot_mat=np.eye(3),
        command_raw=[2.0, -1.0, 4.0],
        joint_pos_robot=OFFICIAL_DEFAULT_POSE_ROBOT,
        joint_vel_robot=np.zeros(16),
        last_action_norm=np.zeros(16),
    )
    np.testing.assert_allclose(got[6:9], [1.0, -0.6, 1.0])


def test_heightmap_flatten_is_channel_major_x_then_y(contract: PolicyContract):
    height = np.arange(16 * 12, dtype=np.float32).reshape(16, 12)
    validity = np.where(height % 2 == 0, 1.0, 0.0).astype(np.float32)
    flat = contract.flatten_heightmap(height, validity)
    assert flat.shape == (384,)
    np.testing.assert_array_equal(flat[:192], height.reshape(-1, order="C"))
    np.testing.assert_array_equal(flat[192:], validity.reshape(-1, order="C"))
    height_b, validity_b = contract.unflatten_heightmap(flat)
    np.testing.assert_array_equal(height_b, height)
    np.testing.assert_array_equal(validity_b, validity)

    proprio = np.zeros(57, dtype=np.float32)
    student = contract.assemble_student_441(proprio, height, validity)
    np.testing.assert_array_equal(student[57:249], height.reshape(-1, order="C"))
    np.testing.assert_array_equal(student[249:441], validity.reshape(-1, order="C"))


def test_reconstruct_action_zero_and_ones(contract: PolicyContract):
    pos0, vel0 = cpp_reconstruct_action(np.zeros(16))
    got0 = contract.reconstruct_robot_action(np.zeros(16))
    np.testing.assert_allclose(pos0, OFFICIAL_DEFAULT_POSE_ROBOT)
    np.testing.assert_allclose(vel0, 0.0)
    np.testing.assert_allclose(got0.goal_joint_pos, pos0)
    np.testing.assert_allclose(got0.goal_joint_vel, vel0)

    ones = np.ones(16)
    pos1, vel1 = cpp_reconstruct_action(ones)
    got1 = contract.reconstruct_robot_action(ones)
    np.testing.assert_allclose(got1.goal_joint_pos, pos1, atol=1e-6)
    np.testing.assert_allclose(got1.goal_joint_vel, vel1, atol=1e-6)
    # fl_hipx / fl_hipy / fl_knee / fl_wheel in robot_order
    np.testing.assert_allclose(got1.goal_joint_pos[0], 0.125, atol=1e-6)
    np.testing.assert_allclose(got1.goal_joint_pos[1], -0.05, atol=1e-6)
    np.testing.assert_allclose(got1.goal_joint_pos[2], 0.85, atol=1e-6)
    np.testing.assert_allclose(got1.goal_joint_vel[3], 5.0, atol=1e-6)
    np.testing.assert_allclose(got1.kp, OFFICIAL_KP)
    np.testing.assert_allclose(got1.kd, OFFICIAL_KD)


def test_reconstruct_does_not_scale_twice(contract: PolicyContract):
    action = np.zeros(16)
    action[0] = 2.0  # fl_hipx in policy_order
    got = contract.reconstruct_robot_action(action)
    np.testing.assert_allclose(got.goal_joint_pos[0], 2.0 * 0.125)
    assert not np.isclose(got.goal_joint_pos[0], 2.0 * 0.125 * 0.125)


def test_lidar_and_heightmap_contracts_are_sensor_first(config_dir):
    lidar = yaml.safe_load((config_dir / "lidar.yaml").read_text(encoding="utf-8"))["lidar"]
    height = yaml.safe_load((config_dir / "heightmap.yaml").read_text(encoding="utf-8"))[
        "heightmap"
    ]
    assert lidar["implementation"] == "mj_multiRay"
    assert lidar["scan_on_sim_thread"] is True
    assert 2 in lidar["exclude_geom_groups"]
    assert height["flatten_order"] == "channel_major_x_then_y"
    assert height["policy_grid"] == [16, 12]
    assert height["full_grid"] == [40, 30]
    assert height["unknown_validity"] == 0.0
    np.testing.assert_allclose(lidar["body_frame_origin_m"], [0.22335, 0.0, -0.0001])


def test_track_has_33_waypoints_and_startup_heading(config_dir):
    track = yaml.safe_load((config_dir / "track.yaml").read_text(encoding="utf-8"))
    assert track["track"]["waypoint_count"] == 33
    assert len(track["waypoints"]) == 33
    assert [item["id"] for item in track["waypoints"]] == list(range(33))
    assert track["start"]["heading_axis"] == "+X"
    np.testing.assert_allclose(track["start"]["base_pos_xyz"], [0.0, -2.5, 0.2])
    np.testing.assert_allclose(track["start"]["waypoint0_xyz"], [0.0, -1.725, 0.0])
    assert track["start"]["min_stand_wait_s"] == 4.0
