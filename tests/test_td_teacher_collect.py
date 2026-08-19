from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/distillation"))

from schema import ChunkedDatasetWriter, validate_dataset  # noqa: E402
from terrain_command import rewrite_command  # noqa: E402
from dataset_coverage import audit_coverage  # noqa: E402
from focus_segments import (  # noqa: E402
    load_fail_segments, official_focus_waypoints, requires_privileged,
)


def full_valid_flat():
    grid = np.zeros((2, 16, 12), dtype=np.float32)
    grid[1] = 1.0
    return grid


def test_flat_ground_preserves_command_after_smoother_is_settled():
    raw = np.asarray([1.0, 0.2, 0.4], dtype=np.float32)
    command, risk = rewrite_command(raw, full_valid_flat(), raw)
    np.testing.assert_allclose(command, raw)
    assert risk[7] == 0.0


def test_clear_ground_can_accelerate_smoothly_within_official_limit():
    raw = np.asarray([0.7, 0.0, 0.0], dtype=np.float32)
    settled = np.asarray([0.84, 0.0, 0.0], dtype=np.float32)
    command, risk = rewrite_command(raw, full_valid_flat(), settled)
    np.testing.assert_allclose(command, settled, atol=1e-6)
    assert risk[7] == 0.0 and command[0] <= 1.0


def test_step_slows_and_left_obstacle_steers_right_with_slew_limit():
    grid = full_valid_flat()
    grid[0, 8, 7] = 0.5  # 0.4m obstacle in the left half
    raw = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    command, risk = rewrite_command(raw, grid, raw)
    assert risk[0] == np.float32(0.4)
    assert risk[7] == 1.0
    assert 0.959 <= command[0] < 1.0
    assert -0.031 <= command[1] < 0.0
    assert command[0] <= 1.0 and abs(command[1]) <= 0.6


def test_unknown_heightmap_is_conservative_and_finite():
    raw = np.asarray([0.7, 0.0, 0.0], dtype=np.float32)
    command, risk = rewrite_command(raw, np.zeros(384, np.float32), raw)
    assert risk[5] == 1.0 and risk[6] == 0.0 and risk[7] == 1.0
    assert command[0] < raw[0]
    assert np.isfinite(command).all()


def make_record(*, success=True, pre_failure=False, danger=False):
    raw = np.asarray([0.7, 0.0, 0.0], dtype=np.float32)
    terrain = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    student = np.zeros(441, dtype=np.float32)
    teacher = np.zeros(57, dtype=np.float32)
    student[6:9] = raw
    teacher[6:9] = terrain
    risk = np.zeros(8, dtype=np.float32)
    if danger:
        risk[0] = 0.4
        risk[7] = 1.0
    return {
        "obs_student": student,
        "obs_teacher": teacher,
        "heightmap": np.zeros((2, 16, 12), dtype=np.float32),
        "action_teacher": np.zeros(16, dtype=np.float32),
        "cmd_raw": raw,
        "cmd_terrain": terrain,
        "risk_features": risk,
        "pose": np.asarray([0, 0, 0.2, 0, 0, 0, 1], dtype=np.float32),
        "timestamp_ns": 1,
        "sequence": 0,
        "episode_id": 0,
        "wp_id": 0,
        "next_wp_id": 1,
        "teacher_source": 0,
        "failure_code": 0,
        "success": success,
        "pre_failure": pre_failure,
        "focus_segment": True,
        "post_teleport": False,
        "contrast_label": 2 if success else (1 if pre_failure else 0),
        "heightmap_valid": True,
        "heightmap_age_ms": 10.0,
    }


def test_npz_is_pickle_free_and_cross_platform_readable(tmp_path):
    output = tmp_path / "20260819_seed0.npz"
    writer = ChunkedDatasetWriter(output, {"git_commit": "abc"}, chunk_size=10)
    writer.append(make_record())
    paths = writer.close()
    assert paths == [output]
    assert validate_dataset(output) == 1
    with np.load(output, allow_pickle=False) as data:
        assert data["obs_student"].shape == (1, 441)
        assert data["action_teacher"].shape == (1, 16)
        assert not any(data[name].dtype.hasobject for name in data.files)


def test_ta_fail_segments_are_real_input_and_deduplicated(tmp_path):
    handoff = tmp_path / "fail_segments.md"
    handoff.write_text("wp~=6, stall\nwp~=1, tumble\nwp~=6, stall\nwp~=28, stall\n")
    assert load_fail_segments(handoff) == (1, 6, 28)


def test_difficult_segments_are_handed_to_privileged_teacher():
    assert requires_privileged(16, 17)
    assert requires_privileged(27, 28)
    assert requires_privileged(31, 32)
    assert not requires_privileged(15, 16)
    assert official_focus_waypoints([6, 16, 27, 28, 31, 32]) == (6,)


def test_coverage_requires_flat_danger_failure_and_success_control(tmp_path):
    output = tmp_path / "focus.npz"
    writer = ChunkedDatasetWriter(output, {"git_commit": "abc"}, chunk_size=10)
    writer.append(make_record(success=True, danger=False))
    writer.append(make_record(success=False, pre_failure=True, danger=True))
    writer.close()
    report = audit_coverage([output], [0], require_full_route=False)
    assert report["complete"] is True
    assert report["per_waypoint"]["0"] == {
        "samples": 2, "pre_failure": 1, "success": 1,
    }


def test_coverage_rejects_missing_success_control(tmp_path):
    output = tmp_path / "failure_only.npz"
    writer = ChunkedDatasetWriter(output, {"git_commit": "abc"}, chunk_size=10)
    writer.append(make_record(success=False, pre_failure=True, danger=True))
    writer.close()
    report = audit_coverage([output], [0], require_full_route=False)
    assert report["complete"] is False
    assert report["missing_success"] == [0]


def test_cpp_terrain_command_contract_executes(tmp_path):
    executable = tmp_path / "terrain_command_math_smoke"
    subprocess.run([
        "g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
        str(ROOT / "tests/terrain_command_math_smoke.cpp"),
        "-I", str(ROOT / "src/s10_terrain_policy/cpp"),
        "-o", str(executable),
    ], check=True)
    result = subprocess.run([str(executable)], check=True, text=True, capture_output=True)
    assert result.stdout.strip() == "TD terrain command contract verified"


def test_teacher_runner_prevents_future_action_leakage_and_uses_official_shape():
    source = (ROOT / "src/s10_terrain_policy/cpp/teacher_collect_runner.hpp").read_text()
    assert "Both observations are assembled before last_action_ is updated" in source
    assert source.index("Publish(student") < source.index("last_action_ = action")
    assert "input.back() != 57" in source
    assert "output.back() != 16" in source
    assert '"/S10_TD_SAMPLE"' in source


def test_autonav_is_reused_for_status_and_failure_labels():
    source = (ROOT / "src/S10_sdk_deploy/interface/user_command/autonav_interface.hpp").read_text()
    assert '"/S10_AUTONAV_STATUS"' in source
    assert "detect_stall" in source and "request_teleport" in source
    assert "publish_status(teleported, failure_code(reason))" in source


def test_teleport_assisted_waypoint_is_not_a_success_control():
    source = (ROOT / "training/distillation/collect.py").read_text()
    assert 'and not record["post_teleport"]' in source


def test_simulator_supports_seeded_initial_pose_jitter():
    source = (
        ROOT / "src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py"
    ).read_text()
    assert "S10_START_JITTER_X" in source
    assert "S10_START_JITTER_Y" in source
    assert "S10_START_JITTER_YAW" in source
