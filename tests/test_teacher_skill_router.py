from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest


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
from teacher_skill_router import clamp_height_corridor, clamp_height_window


def test_corridor_preserves_forward_rows_and_input_and_interpolates_boundary():
    y = np.linspace(-1.6, 1.6, 33)
    raw = (np.arange(41)[:, None] + y[None, :]).astype(np.float32)[None]
    original = raw.copy()
    assert clamp_height_corridor(raw, 0) is raw
    actual = clamp_height_corridor(raw, .25)
    expected = np.arange(41)[:, None] + np.clip(y, -.25, .25)[None, :]
    np.testing.assert_allclose(actual[0], expected, atol=2e-6)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(actual[:, :, 14:19], raw[:, :, 14:19])


def test_three_locomotion_actors_share_selected_surface_and_xy_clamp():
    from types import SimpleNamespace
    runtime = object.__new__(TeacherSkillRuntime)
    runtime.low_forward_command_max_mps = .6
    runtime.low_height_corridor_half_width_m = .4
    runtime.low_height_x_range = [-.4, 1.2]
    raw = np.arange(1353, dtype=np.float32).reshape(1, 41, 33)
    original = raw.copy()
    class Actor:
        def reset(self):
            pass
        def __call__(self, command, proprio, height):
            self.height = height.copy()
            return np.zeros(16)
    runtime.actors = {mode.name: Actor() for mode in PolicyMode}
    for mode in PolicyMode:
        runtime.router = SimpleNamespace(select=lambda *args: mode)
        runtime.step([.4, 0, 0], np.zeros(57), raw, None, None, None, None,
                     low_height_map=raw + 99)
        expected = raw if mode == PolicyMode.RECOVERY else clamp_height_window(raw + 99, .4, [-.4, 1.2])
        np.testing.assert_array_equal(runtime.actors[mode.name].height, expected)
        np.testing.assert_array_equal(raw, original)
        # Unrelated far-away Z must not leak into ANY locomotion Actor.
        polluted = raw + 99
        polluted[:, 24:, :] = 12345
        polluted[:, :, :10] = -12345
        polluted[:, :, 23:] = 12345
        runtime.step([.4, 0, 0], np.zeros(57), raw, None, None, None, None,
                     low_height_map=polluted)
        np.testing.assert_array_equal(runtime.actors[mode.name].height, expected)


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
        "height_split_m": .16,
        "low_forward_command_max_mps": .6,
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


@pytest.mark.parametrize("pause_frames", [3, 40, 300])
@pytest.mark.parametrize("resume_treads,command,resumes", [
    ([.1, .1, 0., 0.], [.3, 0, 0], True),
    ([.1, .1, .1, 0.], [.3, 0, 0], True),
    ([.1, .1, .1, .1], [.3, 0, 0], False),
    ([0., 0., .1, .1], [.3, 0, 0], False),
    ([np.nan, .1, 0., 0.], [.3, 0, 0], False),
    ([.1, .1, 0., 0.], [-.3, 0, 0], False),
    ([.1, .1, 0., 0.], [.3, 0, .5], False),
])
def test_resume_straddled_tread_without_forward_target(pause_frames, resume_treads, command, resumes):
    from dataclasses import replace
    from types import SimpleNamespace
    contract = dict(height_split_m=.16, forward_min_x=.1, max_abs_y=.1, max_abs_yaw=.1,
                    detector_confirm_s=.04, attempt_timeout_s=12., successor_search_s=.2,
                    recovery_hold_s=.2, recovery_timeout_s=5., recovery_base_height_m=.35)
    router = S10TeacherSkillRouter(contract, .02)
    detection = _detect('<geom type="box" pos=".9 0 .05" size=".5 .6 .05"/>')
    state = SimpleNamespace(base_rotation_w=np.eye(3), base_pos_w=np.array([0, 0, .4]),
                            base_lin_vel_b=np.zeros(3), base_ang_vel_b=np.zeros(3))
    for _ in range(2):
        router.select([.3, 0, 0], detection, state, [.21, .21, .11, .11], [True]*4,
                      [.1, .1, 0., 0.])
    assert router.mode == PolicyMode.LOW_STEP_SEQUENCE
    invisible = replace(detection, has_target=False)
    for _ in range(pause_frames):
        router.select([0, 0, 0], invisible, state, [.21, .21, .11, .11], [True]*4,
                      resume_treads)
        assert router.mode in (PolicyMode.RECOVERY, PolicyMode.NORMAL)
    mode = router.select(command, invisible, state, [.21, .21, .11, .11], [True]*4, resume_treads)
    assert (mode == PolicyMode.LOW_STEP_SEQUENCE) == resumes
    if resumes:
        assert router.locked_edge_center_w is not None
        assert router.mode_steps == 0
    router.reset()
    assert router.paused_climb_mode is None


@pytest.mark.parametrize("treads,command,contacts,expected", [
    ([.075, .075, 0., 0.], [.4, 0, 0], [True]*4, True),
    ([.15, .15, .075, 0.], [.4, 0, 0], [True]*4, True),
    ([.075, .075, .075, 0.], [.4, 0, 0], [True]*4, True),
    ([0., 0., 0., 0.], [.4, 0, 0], [True]*4, False),
    ([0., 0., .075, .075], [.4, 0, 0], [True]*4, False),
    ([.3, .3, 0., 0.], [.4, 0, 0], [True]*4, False),
    ([.075, .075, 0., 0.], [0, 0, 0], [True]*4, False),
    ([.075, .075, 0., 0.], [-.4, 0, 0], [True]*4, False),
    ([.075, .075, 0., 0.], [.4, 0, .5], [True]*4, False),
    ([.075, .075, 0., 0.], [.4, 0, 0], [False, False, True, True], False),
    ([.075, .075, np.nan, 0.], [.4, 0, 0], [True]*4, False),
])
def test_normal_enters_low_from_treads_without_history_or_forward_target(treads, command, contacts, expected):
    from dataclasses import replace
    from types import SimpleNamespace
    contract = dict(height_split_m=.16, forward_min_x=.1, max_abs_y=.1, max_abs_yaw=.1,
                    detector_confirm_s=.04, attempt_timeout_s=12., successor_search_s=.2,
                    recovery_hold_s=.2, recovery_timeout_s=5., recovery_base_height_m=.35)
    router = S10TeacherSkillRouter(contract, .02)
    detection = replace(_detect(''), has_target=False)
    state = SimpleNamespace(base_rotation_w=np.eye(3), base_pos_w=np.array([0, 0, .4]))
    assert router.paused_climb_mode is None
    for _ in range(2):
        assert router.select(command, detection, state, np.asarray(treads)+.11, contacts, treads) == PolicyMode.NORMAL
    mode = router.select(command, detection, state, np.asarray(treads)+.11, contacts, treads)
    assert (mode == PolicyMode.LOW_STEP_SEQUENCE) == expected
    if expected:
        assert router.last_entry_reason == "normal_treads"
        assert router.locked_upper_z_w == max(treads[:2])
        assert router.locked_direction_w is not None
        router.select(command, detection, state, np.asarray(treads)+.11, contacts, treads)


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
    runtime.low_forward_command_max_mps = 0.6
    runtime.low_height_corridor_half_width_m = 0.0
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

    runtime.router.mode = PolicyMode.LOW_STEP_SEQUENCE
    runtime.step(
        np.asarray([1.0, .05, -.1]), np.zeros(57), np.zeros((1, 41, 33)),
        unused, unused, np.zeros(4), np.zeros(4, bool),
    )
    np.testing.assert_allclose(
        runtime.actors["LOW_STEP_SEQUENCE"].calls[-1], [.6, .05, -.1]
    )
