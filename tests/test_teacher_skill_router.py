from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "s10_terrain_perception"))
sys.path.insert(0, str(ROOT / "src" / "s10_terrain_policy"))

from mujoco_climb_detector import S10ClimbDetector
from mujoco_teacher import (
    DEFAULT_ROBOT,
    PrivilegedHeightScanner,
    S10PolicyState,
    assemble_asymmetric_teacher_inputs,
    assemble_official_57,
)
from teacher_skill_router import PolicyMode, S10TeacherSkillRouter, TeacherSkillRuntime


DETECTOR_CONTRACT = {
    "front_x_range_m": [0.30, 1.20],
    "corridor_half_width_m": 0.25,
    "min_height_m": 0.04,
    "max_height_m": 0.45,
    "min_landing_width_m": 0.42,
    "min_landing_depth_m": 0.20,
    "landing_height_tolerance_m": 0.04,
    "landing_coverage_min": 0.80,
    "clearance_start_x_m": 0.25,
    "clearance_range_m": 1.20,
    "clearance_lateral_m": [-0.22, 0.0, 0.22],
    "clearance_height_above_base_m": [-0.25, 0.0, 0.15],
}


def _model(obstacle_xml: str):
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco><worldbody>
        <geom name="ground" type="plane" size="10 10 .1" group="0"/>
        {obstacle_xml}
        <body name="base_link" pos="0 0 .4">
          <freejoint/><geom type="sphere" size=".05" group="1"/>
        </body>
        </worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return model, data, base_id


def _detect(obstacle_xml: str):
    model, data, base_id = _model(obstacle_xml)
    scanner = PrivilegedHeightScanner(model, body_exclude=base_id)
    geometry = scanner.scan_geometry(data, [0, 0, .4], np.eye(3))
    detector = S10ClimbDetector(model, base_id, DETECTOR_CONTRACT)
    return detector.detect(geometry, data, [0, 0, .4], np.eye(3), [.3, 0, 0])


def test_asymmetric_observation_reorders_command_before_proprio():
    state = S10PolicyState(
        np.asarray([0, 0, .4]),
        np.eye(3),
        np.asarray([1, 2, 3], np.float32),
        np.asarray([.4, -.8, 1.2], np.float32),
        DEFAULT_ROBOT.copy(),
        np.zeros(16, np.float32),
    )
    official = assemble_official_57(state, [.5, -.2, .3], np.zeros(16))
    command, proprio, height = assemble_asymmetric_teacher_inputs(
        state, official, np.zeros(1353)
    )
    np.testing.assert_allclose(command, [.5, -.2, .3])
    np.testing.assert_allclose(proprio[:3], [1, 2, 3])
    np.testing.assert_allclose(proprio[3:9], official[:6])
    np.testing.assert_allclose(proprio[9:], official[9:])
    assert height.shape == (1, 41, 33)


def test_solid_step_is_a_target_not_an_overhang():
    detection = _detect(
        '<geom name="step" type="box" pos=".9 0 .1" size=".5 .6 .1" group="0"/>'
    )
    assert detection.edge_candidate
    assert detection.has_target
    assert not detection.overhead_rejected
    assert abs(detection.height_m - .2) < .02


def test_suspended_slab_is_rejected_by_layered_clearance_rays():
    detection = _detect(
        '<geom name="overhang" type="box" pos=".9 0 .4" size=".5 .6 .05" group="0"/>'
    )
    assert detection.edge_candidate
    assert detection.upper_clearance_blocked
    assert detection.overhead_rejected
    assert not detection.has_target


def test_router_uses_confirmed_per_wheel_support_and_operator_override():
    contract = {
        "height_split_m": .18,
        "forward_min_x": .1,
        "max_abs_y": .1,
        "max_abs_yaw": .1,
        "detector_confirm_s": .04,
        "attempt_timeout_s": 12.0,
        "successor_search_s": .20,
        "recovery_hold_s": .20,
        "recovery_timeout_s": 5.0,
        "recovery_base_height_m": .35,
    }
    router = S10TeacherSkillRouter(contract, .02)
    detection = _detect(
        '<geom name="step" type="box" pos=".9 0 .05" size=".5 .6 .05" group="0"/>'
    )
    state = S10PolicyState(
        np.asarray([0, 0, .4]), np.eye(3), np.zeros(3), np.zeros(3),
        DEFAULT_ROBOT.copy(), np.zeros(16),
    )
    no_contact = np.zeros(4, bool)
    for _ in range(2):
        mode = router.select([.3, 0, 0], detection, state, np.zeros(4), no_contact)
    assert mode == PolicyMode.LOW_STEP_SEQUENCE

    wheel_z = np.full(4, detection.upper_z_w + .11)
    for _ in range(12):
        mode = router.select(
            [.3, 0, 0], detection, state, wheel_z, np.ones(4, bool)
        )
    assert mode == PolicyMode.RECOVERY
    mode = router.select([.3, 0, 0], detection, state, wheel_z, np.ones(4, bool))
    assert mode == PolicyMode.NORMAL


def test_runtime_executes_only_selected_actor_and_resets_dormant_grus():
    class FakeRouter:
        mode = PolicyMode.HIGH_CLIMB

        def select(self, *_args):
            return self.mode

    class FakeActor:
        def __init__(self, value):
            self.value = value
            self.calls = []
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def __call__(self, command, proprio, height_map):
            self.calls.append(np.asarray(command).copy())
            return np.full(16, self.value, np.float32)

    runtime = object.__new__(TeacherSkillRuntime)
    runtime.router = FakeRouter()
    runtime.actors = {
        role: FakeActor(index) for index, role in enumerate(
            ("NORMAL", "HIGH_CLIMB", "LOW_STEP_SEQUENCE", "RECOVERY")
        )
    }
    unused = object()
    action = runtime.step(
        np.asarray([.3, 0, 0]), np.zeros(57), np.zeros((1, 41, 33)),
        unused, unused, np.zeros(4), np.zeros(4, bool),
    )
    np.testing.assert_allclose(action, 1.0)
    assert len(runtime.actors["HIGH_CLIMB"].calls) == 1
    assert runtime.actors["HIGH_CLIMB"].reset_count == 0
    for role in ("NORMAL", "LOW_STEP_SEQUENCE", "RECOVERY"):
        assert not runtime.actors[role].calls
        assert runtime.actors[role].reset_count == 1

    runtime.router.mode = PolicyMode.RECOVERY
    runtime.step(
        np.asarray([.3, 0, 0]), np.zeros(57), np.zeros((1, 41, 33)),
        unused, unused, np.zeros(4), np.zeros(4, bool),
    )
    np.testing.assert_array_equal(
        runtime.actors["RECOVERY"].calls[-1], np.zeros(3)
    )
    assert runtime.actors["HIGH_CLIMB"].reset_count == 1
