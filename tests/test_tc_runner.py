from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from s10_terrain_policy import PolicyContract


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "src/s10_terrain_policy/cpp/terrain_policy_runner.hpp"
MATH = REPO_ROOT / "src/s10_terrain_policy/cpp/terrain_policy_math.hpp"
MODEL = REPO_ROOT / "models/terrain_locomotion.onnx"
CPP_MATH_SMOKE = REPO_ROOT / "tests/terrain_policy_math_smoke.cpp"


def _array_from_header(source: str, name: str, dtype=float) -> np.ndarray:
    match = re.search(rf"{name}\{{([^}}]+)\}};", source, re.DOTALL)
    assert match, f"{name} not found in C++ header"
    values = re.findall(r"-?\d+(?:\.\d+)?", match.group(1).replace("F", ""))
    return np.asarray([dtype(value) for value in values])


def test_cpp_runner_mirrors_frozen_t00_contract(config_dir):
    contract = PolicyContract.load(config_dir)
    runner = RUNNER.read_text(encoding="utf-8")
    math = MATH.read_text(encoding="utf-8")

    assert "kProprioDim = 57" in runner
    assert "kHeightmapDim = 384" in runner
    assert "kObservationDim = 441" in runner
    assert "kActionDim = 16" in runner
    assert "kDecimation = 4" in runner
    assert "kOmegaScale = 0.25F" in runner
    assert "kDofVelScale = 0.05F" in runner
    for unused in (
        "kRobotToPolicy",
        "kPolicyToRobot",
        "kDefaultPosePolicy",
        "kDefaultPoseRobot",
        "kActionScaleRobot",
    ):
        assert unused not in runner
    np.testing.assert_array_equal(
        _array_from_header(math, "kRobotToPolicy", int), contract.robot2policy_idx
    )
    np.testing.assert_array_equal(
        _array_from_header(math, "kPolicyToRobot", int), contract.policy2robot_idx
    )
    np.testing.assert_allclose(
        _array_from_header(math, "kDefaultPosePolicy"), contract.default_pose_policy
    )
    np.testing.assert_allclose(
        _array_from_header(math, "kDefaultPoseRobot"), contract.default_pose_robot
    )
    np.testing.assert_allclose(
        _array_from_header(math, "kActionScaleRobot"), contract.action_scale_robot
    )
    np.testing.assert_allclose(_array_from_header(runner, "kKpRobot"), contract.kp_robot)
    np.testing.assert_allclose(_array_from_header(runner, "kKdRobot"), contract.kd_robot)


def test_runner_uses_t00_heightmap_topic_and_nonblocking_zero_fallback(config_dir):
    source = RUNNER.read_text(encoding="utf-8")
    policy = yaml.safe_load((config_dir / "policy.yaml").read_text(encoding="utf-8"))
    assert policy["topics"]["heightmap"] == "/S10_HEIGHTMAP"
    assert 'heightmap_topic = "/S10_HEIGHTMAP"' in source
    assert "msg->data.size() != kHeightmapDim" in source
    assert "if (fresh) heightmap = heightmap_;" in source
    assert "kHeightmapTimeout = std::chrono::milliseconds(250)" in source
    assert "Controller::kProprioClone" in source


def test_runner_does_not_convert_robot_basic_state_rpy_twice():
    source = RUNNER.read_text(encoding="utf-8")
    assert "Deg2Rad" not in source
    assert "rotation[row * 3 + col] = robot.base_rot_mat(row, col);" in source
    assert "TerrainPolicyMath::AssembleObservation" in source


def test_placeholder_model_is_deterministic(tmp_path):
    generated = tmp_path / "terrain_locomotion.onnx"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/export_zero_policy.py"), str(generated)],
        check=True,
    )
    assert generated.read_bytes() == MODEL.read_bytes()
    payload = generated.read_bytes()
    assert b"obs" in payload
    assert b"actions" in payload
    assert b"Constant" in payload


def test_cpp_observation_and_action_contract_executes(tmp_path):
    executable = tmp_path / "terrain_policy_math_smoke"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(CPP_MATH_SMOKE),
            "-I",
            str(REPO_ROOT / "src/s10_terrain_policy/cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run([str(executable)], check=True, text=True, capture_output=True)
    assert result.stdout.strip() == "C++ observation and action contract verified"


def test_heightmap_invalid_and_stale_data_fall_back_to_zero():
    source = RUNNER.read_text(encoding="utf-8")
    assert "kHeightmapTimeout = std::chrono::milliseconds(250)" in source
    assert source.count("ClearHeightmap();") == 2
    assert "else has_heightmap_ = false;" in source


def test_ros_smoke_publishes_encoder_space_joints():
    source = (REPO_ROOT / "scripts/tc_joint_cmd_smoke.py").read_text(
        encoding="utf-8"
    )
    assert "POS_OFFSET_RAD" in source
    assert "(q_raw + excitation - offset) * direction" in source
    assert "acc_z = 9.81" in source
    assert "stdout=subprocess.PIPE" in source
    assert "DEVNULL" not in source


def test_runner_accepts_dynamic_onnx_batch_dim():
    source = RUNNER.read_text(encoding="utf-8")
    assert "actual.back() != expected.back()" in source
    assert "(actual[0] > 0 && actual[0] != expected[0])" in source
    assert "actual[0] != expected[0] || actual[1] != expected[1]" not in source


def test_sdk_uses_repository_local_runner_and_model():
    sdk = REPO_ROOT / "src/S10_sdk_deploy"
    cmake = (sdk / "CMakeLists.txt").read_text(encoding="utf-8")
    main = (sdk / "main.cpp").read_text(encoding="utf-8")
    rl_state = (
        sdk / "state_machine/quadruped_wheel/rl_control_state.hpp"
    ).read_text(encoding="utf-8")

    assert "../s10_terrain_policy/cpp" in cmake
    assert "../../models/terrain_locomotion.onnx" in cmake
    assert "S10_TERRAIN_POLICY_ROOT" not in cmake
    assert "terrain_policy_runner.hpp" in rl_state
    assert 'policy" / "policy.onnx' not in rl_state
    assert "S10_TERRAIN_DEFAULT_MODEL" in main
    assert '--controller' in main
    assert 'arg == "--ros-args"' in main
    assert 'arg.rfind("__", 0) == 0' in main
    assert 'arg == "-r" || arg == "--remap"' in main
