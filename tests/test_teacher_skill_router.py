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
from teacher_skill_router import (
    PolicyMode,
    S10LowCommandAdapter,
    S10TeacherSkillRouter,
    TeacherSkillRuntime,
)
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


def test_high_and_low_receive_their_own_selected_surface_with_same_xy_clamp():
    from types import SimpleNamespace
    runtime = object.__new__(TeacherSkillRuntime)
    runtime.low_forward_command_max_mps = .6
    runtime.low_height_corridor_half_width_m = .4
    runtime.low_height_x_range = [-.4, 1.2]
    raw = np.arange(1353, dtype=np.float32).reshape(1, 41, 33)
    high = raw + 199
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
                     low_height_map=raw + 99, high_height_map=high)
        expected = (
            raw if mode == PolicyMode.RECOVERY
            else clamp_height_window(high, .4, [-.4, 1.2])
            if mode == PolicyMode.HIGH_CLIMB
            else clamp_height_window(raw + 99, .4, [-.4, 1.2])
        )
        np.testing.assert_array_equal(runtime.actors[mode.name].height, expected)
        np.testing.assert_array_equal(raw, original)
        # Polluting LOW's selected surface must not alter HIGH's separately
        # selected surface; each still receives the same XY clamp.
        polluted = raw + 99
        polluted[:, 24:, :] = 12345
        polluted[:, :, :10] = -12345
        polluted[:, :, 23:] = 12345
        runtime.step([.4, 0, 0], np.zeros(57), raw, None, None, None, None,
                     low_height_map=polluted, high_height_map=high)
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


def test_router_returns_normal_directly_after_confirmed_success():
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
    assert mode == PolicyMode.NORMAL
    assert not router.just_entered_recovery


def test_low_released_command_hands_directly_to_normal():
    contract = {
        "height_split_m": .16, "forward_min_x": .1,
        "max_abs_y": .1, "max_abs_yaw": .1,
        "detector_confirm_s": .04, "attempt_timeout_s": 12.,
        "successor_search_s": .20, "recovery_hold_s": .20,
        "recovery_timeout_s": 5., "recovery_base_height_m": .35,
    }
    router = S10TeacherSkillRouter(contract, .02)
    detection = _detect(
        '<geom name="step" type="box" pos=".9 0 .05" size=".5 .6 .05" group="0"/>'
    )
    state = S10PolicyState(
        np.asarray([0, 0, .4]), np.eye(3), np.zeros(3), np.zeros(3),
        DEFAULT_ROBOT.copy(), np.zeros(16),
    )
    for _ in range(2):
        router.select([.3, 0, 0], detection, state, np.zeros(4), np.zeros(4, bool))
    assert router.select(
        [0, 0, 0], detection, state, np.zeros(4), np.zeros(4, bool),
    ) == PolicyMode.NORMAL
    assert router.last_transition_reason == "low_command_handoff"
    assert not router.just_entered_recovery


@pytest.mark.parametrize("handoff_command", [
    [.3, .2, 0.],
    [.3, 0., .2],
    [0., 0., 0.],
    [-.3, 0., 0.],
])
def test_low_non_forward_command_hands_directly_to_normal(handoff_command):
    contract = {
        "height_split_m": .16, "forward_min_x": .1,
        "max_abs_y": .1, "max_abs_yaw": .1,
        "detector_confirm_s": .04, "attempt_timeout_s": 12.,
        "successor_search_s": .20, "recovery_hold_s": .20,
        "recovery_timeout_s": 5., "recovery_base_height_m": .35,
    }
    router = S10TeacherSkillRouter(contract, .02)
    detection = _detect(
        '<geom name="step" type="box" pos=".9 0 .05" size=".5 .6 .05" group="0"/>'
    )
    state = S10PolicyState(
        np.asarray([0, 0, .4]), np.eye(3), np.zeros(3), np.zeros(3),
        DEFAULT_ROBOT.copy(), np.zeros(16),
    )
    for _ in range(2):
        router.select([.3, 0, 0], detection, state, np.zeros(4), np.zeros(4, bool))
    assert router.select(
        handoff_command, detection, state, np.zeros(4), np.zeros(4, bool),
    ) == PolicyMode.NORMAL
    assert router.last_transition_reason == "low_command_handoff"
    assert not router.just_entered_recovery


def test_continuous_low_stairs_advance_successor_during_split_support():
    from dataclasses import replace

    contract = {
        "height_split_m": .16, "forward_min_x": .1,
        "max_abs_y": .1, "max_abs_yaw": .1,
        "detector_confirm_s": .04, "attempt_timeout_s": 12.,
        "successor_search_s": .20, "recovery_hold_s": .20,
        "recovery_timeout_s": 5., "recovery_base_height_m": .35,
    }
    router = S10TeacherSkillRouter(contract, .02)
    first = _detect(
        '<geom name="step" type="box" pos=".9 0 .05" size=".5 .6 .05" group="0"/>'
    )
    state = S10PolicyState(
        np.asarray([0, 0, .4]), np.eye(3), np.zeros(3), np.zeros(3),
        DEFAULT_ROBOT.copy(), np.zeros(16),
    )
    for _ in range(2):
        router.select([.3, 0, 0], first, state, np.zeros(4), np.zeros(4, bool))
    next_upper = first.upper_z_w + .075
    successor = replace(
        first,
        edge_segment_w=first.edge_segment_w + np.asarray([.25, 0., .075]),
        upper_z_w=next_upper,
        height_m=.075,
    )
    treads = np.asarray([next_upper, next_upper, first.upper_z_w, first.upper_z_w])
    wheel_z = treads + .11
    for _ in range(3):
        mode = router.select(
            [.3, 0, 0], successor, state, wheel_z, np.ones(4, bool), treads
        )
    assert mode == PolicyMode.LOW_STEP_SEQUENCE
    assert router.locked_upper_z_w == pytest.approx(next_upper)
    assert router.last_transition_reason == "confirmed_low_successor"
    assert router.mode_steps == 0
    assert not router.wheel_confirmed.any()


def test_low_stairs_handoff_directly_to_detected_high_successor():
    from dataclasses import replace

    contract = {
        "height_split_m": .16, "forward_min_x": .1,
        "max_abs_y": .1, "max_abs_yaw": .1,
        "detector_confirm_s": .04, "attempt_timeout_s": 12.,
        "successor_search_s": .20, "recovery_hold_s": .20,
        "recovery_timeout_s": 5., "recovery_base_height_m": .35,
    }
    router = S10TeacherSkillRouter(contract, .02)
    first = _detect(
        '<geom name="step" type="box" pos=".9 0 .05" size=".5 .6 .05" group="0"/>'
    )
    state = S10PolicyState(
        np.asarray([0, 0, .4]), np.eye(3), np.zeros(3), np.zeros(3),
        DEFAULT_ROBOT.copy(), np.zeros(16),
    )
    for _ in range(2):
        router.select([.3, 0, 0], first, state, np.zeros(4), np.zeros(4, bool))

    high_upper = first.upper_z_w + .23
    successor = replace(
        first,
        edge_segment_w=first.edge_segment_w + np.asarray([.25, 0., .23]),
        upper_z_w=high_upper,
        height_m=.23,
    )
    # Detection alone must not transfer control, even after LOW completion.
    low_treads = np.full(4, first.upper_z_w)
    for _ in range(5):
        assert router.select(
            [.3, 0, 0], successor, state, low_treads + .11,
            np.ones(4, bool), low_treads,
        ) == PolicyMode.LOW_STEP_SEQUENCE
    missing = replace(successor, has_target=False)
    high_treads = np.array([high_upper, high_upper, 0., 0.])
    # Historical, alternating contacts must not count as joint support.
    for contacts in ([True, False, False, False], [False, True, False, False]):
        for _ in range(4):
            assert router.select(
                [.3, 0, 0], missing, state, high_treads + .11,
                contacts, high_treads,
            ) == PolicyMode.LOW_STEP_SEQUENCE
    # A ray below a lifted wheel is insufficient, even with a side contact.
    for _ in range(4):
        assert router.select(
            [.3, 0, 0], missing, state, high_treads + .3,
            np.ones(4, bool), high_treads,
        ) == PolicyMode.LOW_STEP_SEQUENCE
    # Detector loss is harmless; rear contacts/height do not gate handoff.
    for i in range(3):
        mode = router.select(
            [.3, 0, 0], missing, state, high_treads + .11,
            [True, True, False, False], high_treads,
        )
        assert mode == (PolicyMode.HIGH_CLIMB if i == 2
                        else PolicyMode.LOW_STEP_SEQUENCE)
    assert router.locked_upper_z_w == pytest.approx(high_upper)
    assert router.last_transition_reason == "confirmed_high_front_support"
    assert router.mode_steps == 0


def test_low_does_not_handoff_to_high_from_one_frame_detection_spike():
    from dataclasses import replace

    contract = {
        "height_split_m": .16, "forward_min_x": .1,
        "max_abs_y": .1, "max_abs_yaw": .1,
        "detector_confirm_s": .04, "attempt_timeout_s": 12.,
        "successor_search_s": .20, "recovery_hold_s": .20,
        "recovery_timeout_s": 5., "recovery_base_height_m": .35,
    }
    router = S10TeacherSkillRouter(contract, .02)
    first = _detect(
        '<geom name="step" type="box" pos=".9 0 .05" size=".5 .6 .05" group="0"/>'
    )
    state = S10PolicyState(
        np.asarray([0, 0, .4]), np.eye(3), np.zeros(3), np.zeros(3),
        DEFAULT_ROBOT.copy(), np.zeros(16),
    )
    for _ in range(2):
        router.select([.3, 0, 0], first, state, np.zeros(4), np.zeros(4, bool))
    high = replace(
        first,
        edge_segment_w=first.edge_segment_w + np.asarray([.25, 0., .23]),
        upper_z_w=first.upper_z_w + .23,
        height_m=.23,
    )
    missing = replace(high, has_target=False)
    assert router.select(
        [.3, 0, 0], high, state, np.zeros(4), np.zeros(4, bool)
    ) == PolicyMode.LOW_STEP_SEQUENCE
    assert router.select(
        [.3, 0, 0], missing, state, np.zeros(4), np.zeros(4, bool)
    ) == PolicyMode.LOW_STEP_SEQUENCE
    assert router.pending_successor is None


def test_runtime_prewarms_pending_high_successor_without_using_its_action():
    from types import SimpleNamespace

    class FakeRouter:
        mode = PolicyMode.LOW_STEP_SEQUENCE
        normal_handoff_settling = False
        pending = True

        def select(self, *_args):
            return self.mode

        def high_successor_pending(self):
            return self.pending

    class FakeActor:
        def __init__(self, value):
            self.value = value
            self.calls = []
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def __call__(self, command, _proprio, height):
            self.calls.append((np.asarray(command).copy(), np.asarray(height).copy()))
            return np.full(16, self.value, np.float32)

    runtime = object.__new__(TeacherSkillRuntime)
    runtime.router = FakeRouter()
    runtime.low_command_adapter = None
    runtime.low_forward_command_max_mps = .6
    runtime.high_forward_command_max_mps = .6
    runtime.low_height_corridor_half_width_m = 0.
    runtime.low_height_x_range = None
    runtime.actors = {
        mode.name: FakeActor(index) for index, mode in enumerate(PolicyMode)
    }
    low_height = np.full((1, 41, 33), 1., np.float32)
    high_height = np.full((1, 41, 33), 2., np.float32)
    args = (
        np.asarray([1., 0., 0.]), np.zeros(57), np.zeros((1, 41, 33)),
        SimpleNamespace(), SimpleNamespace(base_rotation_w=np.eye(3)),
        np.zeros(4), np.zeros(4, bool),
    )
    action = runtime.step(
        *args, low_height_map=low_height, high_height_map=high_height
    )
    np.testing.assert_allclose(action, 2.)
    assert len(runtime.actors["LOW_STEP_SEQUENCE"].calls) == 1
    assert len(runtime.actors["HIGH_CLIMB"].calls) == 1
    np.testing.assert_allclose(runtime.actors["HIGH_CLIMB"].calls[0][0], [.6, 0., 0.])
    np.testing.assert_array_equal(runtime.actors["HIGH_CLIMB"].calls[0][1], high_height)
    assert runtime.actors["HIGH_CLIMB"].reset_count == 0

    # The warmed hidden state remains owned by HIGH on the transfer frame.
    runtime.router.mode = PolicyMode.HIGH_CLIMB
    runtime.router.pending = False
    runtime.step(*args, low_height_map=low_height, high_height_map=high_height)
    assert len(runtime.actors["HIGH_CLIMB"].calls) == 2
    assert runtime.actors["HIGH_CLIMB"].reset_count == 0


@pytest.mark.parametrize("vx", [1.0, 0.6, 0.4, 0.2, 0.0, -0.3])
def test_high_command_cap_preserves_router_normal_shadow_and_user_command(vx):
    from types import SimpleNamespace
    raw = np.asarray([vx, .05, -.08], np.float32)
    before = raw.copy()
    class Actor:
        def reset(self):
            pass
        def __call__(self, command, *_):
            self.command = np.asarray(command).copy()
            return np.zeros(16, np.float32)
    class Router:
        mode = PolicyMode.HIGH_CLIMB
        normal_handoff_settling = True
        def select(self, command, *_):
            self.command = np.asarray(command).copy()
            return self.mode
    runtime = object.__new__(TeacherSkillRuntime)
    runtime.router = Router()
    runtime.high_forward_command_max_mps = .4
    runtime.low_height_corridor_half_width_m = 0.
    runtime.actors = {mode.name: Actor() for mode in PolicyMode}
    args = (raw, np.zeros(57), np.zeros((1, 41, 33)), None, SimpleNamespace(), None, None)
    runtime.step(*args)
    expected = before.copy()
    expected[0] = min(vx, .4)
    np.testing.assert_allclose(runtime.actors['HIGH_CLIMB'].command, expected)
    np.testing.assert_array_equal(runtime.router.command, before)
    np.testing.assert_array_equal(runtime.actors['NORMAL'].command, before)
    np.testing.assert_array_equal(raw, before)
    runtime.router.mode = PolicyMode.NORMAL
    runtime.router.normal_handoff_settling = False
    runtime.step(*args)
    np.testing.assert_array_equal(runtime.actors['NORMAL'].command, before)


def test_model99_low_command_adapter_tracks_latched_world_direction():
    adapter = S10LowCommandAdapter(.02, max_speed_mps=.6, yaw_gain=.5,
                                   yaw_limit=.5, smoothing_tau_s=.2)
    yaw = np.deg2rad(10.)
    rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw), 0.],
        [np.sin(yaw), np.cos(yaw), 0.],
        [0., 0., 1.],
    ])
    actual = adapter.update([.4, 0., 0.], rotation, True, [1., 0.])
    expected = np.asarray([
        .4 * np.cos(yaw), -.4 * np.sin(yaw), -.5 * yaw,
    ])
    np.testing.assert_allclose(actual, expected, atol=1.e-6)
    adapter.update([.4, 0., 0.], rotation, False, None)
    assert not adapter.active
    np.testing.assert_array_equal(adapter.command_b, np.zeros(3))


def test_runtime_wires_adapted_command_only_to_model99_low_actor():
    from types import SimpleNamespace

    class FakeRouter:
        mode = PolicyMode.LOW_STEP_SEQUENCE
        locked_direction_w = np.asarray([1., 0.])
        normal_handoff_settling = False

        def select(self, *_args):
            return self.mode

    class FakeActor:
        def __init__(self):
            self.calls = []
        def reset(self):
            pass
        def __call__(self, command, _proprio, _height):
            self.calls.append(np.asarray(command).copy())
            return np.zeros(16, np.float32)

    runtime = object.__new__(TeacherSkillRuntime)
    runtime.router = FakeRouter()
    runtime.low_forward_command_max_mps = .6
    runtime.low_command_adapter = S10LowCommandAdapter(.02)
    runtime.low_height_corridor_half_width_m = 0.
    runtime.low_height_x_range = None
    runtime.actors = {mode.name: FakeActor() for mode in PolicyMode}
    yaw = np.deg2rad(10.)
    state = SimpleNamespace(base_rotation_w=np.asarray([
        [np.cos(yaw), -np.sin(yaw), 0.],
        [np.sin(yaw), np.cos(yaw), 0.],
        [0., 0., 1.],
    ]))
    raw = np.asarray([.4, 0., 0.], np.float32)
    raw_before = raw.copy()
    runtime.step(raw, np.zeros(57), np.zeros((1, 41, 33)),
                 object(), state, np.zeros(4), np.zeros(4, bool))
    expected = [.4 * np.cos(yaw), -.4 * np.sin(yaw), -.5 * yaw]
    np.testing.assert_allclose(
        runtime.actors["LOW_STEP_SEQUENCE"].calls[-1], expected, atol=1.e-6
    )
    np.testing.assert_array_equal(raw, raw_before)
    assert not runtime.actors["NORMAL"].calls


def test_router_locks_actual_command_xy_in_world_not_only_body_forward():
    from types import SimpleNamespace

    contract = dict(
        height_split_m=.16, forward_min_x=.1, max_abs_y=.1,
        max_abs_yaw=.1, detector_confirm_s=.02, attempt_timeout_s=12.,
        successor_search_s=.2, recovery_hold_s=.2,
        recovery_timeout_s=5., recovery_base_height_m=.35,
    )
    router = S10TeacherSkillRouter(contract, .02)
    yaw = np.deg2rad(30.)
    rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw), 0.],
        [np.sin(yaw), np.cos(yaw), 0.],
        [0., 0., 1.],
    ])
    state = SimpleNamespace(base_rotation_w=rotation)
    detection = _detect(
        '<geom name="step" type="box" pos=".9 0 .05" size=".5 .6 .05" group="0"/>'
    )
    command = np.asarray([.3, .05, 0.])
    assert router.select(
        command, detection, state, np.zeros(4), np.zeros(4, bool)
    ) == PolicyMode.LOW_STEP_SEQUENCE
    expected = rotation[:2, :2] @ command[:2]
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(router.locked_direction_w, expected, atol=1.e-12)


def test_recovery_blocks_normal_handoff_while_yawing_or_sliding_laterally():
    from types import SimpleNamespace

    contract = dict(
        height_split_m=.16, forward_min_x=.1, max_abs_y=.1,
        max_abs_yaw=.1, detector_confirm_s=.04, attempt_timeout_s=12.,
        successor_search_s=.2, recovery_hold_s=.2,
        recovery_timeout_s=5., recovery_base_height_m=.35,
    )
    router = S10TeacherSkillRouter(contract, .02)
    router.mode = PolicyMode.RECOVERY
    unstable = SimpleNamespace(
        base_rotation_w=np.eye(3), base_pos_w=np.array([0., 0., .4]),
        base_lin_vel_b=np.array([.3, .2, 0.]),
        base_ang_vel_b=np.array([0., 0., .2]),
    )
    unused = object()
    for _ in range(router.recovery_hold_steps + 2):
        assert router.select(
            [.3, 0, 0], unused, unstable, np.zeros(4), np.zeros(4, bool)
        ) == PolicyMode.RECOVERY
    stable = SimpleNamespace(
        base_rotation_w=np.eye(3), base_pos_w=np.array([0., 0., .4]),
        base_lin_vel_b=np.array([.3, 0., 0.]),
        base_ang_vel_b=np.zeros(3),
    )
    for _ in range(router.recovery_hold_steps - 1):
        assert router.select(
            [.3, 0, 0], unused, stable, np.zeros(4), np.zeros(4, bool)
        ) == PolicyMode.RECOVERY
    assert router.select(
        [.3, 0, 0], unused, stable, np.zeros(4), np.zeros(4, bool)
    ) == PolicyMode.NORMAL


@pytest.mark.parametrize("pause_frames", [3, 40, 300])
@pytest.mark.parametrize("resume_treads,command,reenters", [
    ([.1, .1, 0., 0.], [.3, 0, 0], True),
    ([.1, .1, .1, 0.], [.3, 0, 0], True),
    ([.1, .1, .1, .1], [.3, 0, 0], False),
    ([0., 0., .1, .1], [.3, 0, 0], False),
    ([np.nan, .1, 0., 0.], [.3, 0, 0], False),
    ([.1, .1, 0., 0.], [-.3, 0, 0], False),
    ([.1, .1, 0., 0.], [.3, 0, .5], False),
])
def test_normal_reenters_straddled_tread_without_forward_target(
    pause_frames, resume_treads, command, reenters
):
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
        assert router.select(
            [0, 0, 0], invisible, state, [.21, .21, .11, .11], [True]*4,
            resume_treads,
        ) == PolicyMode.NORMAL
    for _ in range(3):
        mode = router.select(
            command, invisible, state, [.21, .21, .11, .11], [True]*4,
            resume_treads,
        )
    assert (mode == PolicyMode.LOW_STEP_SEQUENCE) == reenters
    if reenters:
        assert router.locked_edge_center_w is not None
        assert router.mode_steps == 0
    router.reset()


@pytest.mark.parametrize("treads,command,contacts,expected", [
    ([.075, .075, 0., 0.], [.4, 0, 0], [True]*4, PolicyMode.LOW_STEP_SEQUENCE),
    ([.15, .15, .075, 0.], [.4, 0, 0], [True]*4, PolicyMode.LOW_STEP_SEQUENCE),
    ([.075, .075, .075, 0.], [.4, 0, 0], [True]*4, PolicyMode.LOW_STEP_SEQUENCE),
    ([.16, .16, 0., 0.], [.4, 0, 0], [True]*4, PolicyMode.HIGH_CLIMB),
    ([.30, .30, .15, 0.], [.4, 0, 0], [True]*4, PolicyMode.HIGH_CLIMB),
    ([0., 0., 0., 0.], [.4, 0, 0], [True]*4, PolicyMode.NORMAL),
    ([0., 0., .075, .075], [.4, 0, 0], [True]*4, PolicyMode.NORMAL),
    ([.075, .075, 0., 0.], [0, 0, 0], [True]*4, PolicyMode.NORMAL),
    ([.075, .075, 0., 0.], [-.4, 0, 0], [True]*4, PolicyMode.NORMAL),
    ([.075, .075, 0., 0.], [.4, 0, .5], [True]*4, PolicyMode.NORMAL),
    ([.075, .075, 0., 0.], [.4, 0, 0], [False, False, True, True], PolicyMode.NORMAL),
    ([.30, .30, 0., 0.], [.4, 0, 0], [False, False, True, True], PolicyMode.NORMAL),
    ([.075, .075, np.nan, 0.], [.4, 0, 0], [True]*4, PolicyMode.NORMAL),
])
def test_normal_enters_climb_from_treads_without_history_or_forward_target(treads, command, contacts, expected):
    from dataclasses import replace
    from types import SimpleNamespace
    contract = dict(height_split_m=.16, forward_min_x=.1, max_abs_y=.1, max_abs_yaw=.1,
                    detector_confirm_s=.04, attempt_timeout_s=12., successor_search_s=.2,
                    recovery_hold_s=.2, recovery_timeout_s=5., recovery_base_height_m=.35)
    router = S10TeacherSkillRouter(contract, .02)
    detection = replace(_detect(''), has_target=False)
    state = SimpleNamespace(base_rotation_w=np.eye(3), base_pos_w=np.array([0, 0, .4]))
    for _ in range(2):
        assert router.select(command, detection, state, np.asarray(treads)+.11, contacts, treads) == PolicyMode.NORMAL
    mode = router.select(command, detection, state, np.asarray(treads)+.11, contacts, treads)
    assert mode == expected
    if expected in (PolicyMode.LOW_STEP_SEQUENCE, PolicyMode.HIGH_CLIMB):
        assert router.last_entry_reason == (
            "normal_high_treads" if expected == PolicyMode.HIGH_CLIMB else "normal_low_treads"
        )
        assert router.locked_upper_z_w == max(treads[:2])
        assert router.locked_direction_w is not None
        router.select(command, detection, state, np.asarray(treads)+.11, contacts, treads)


def test_runtime_shadows_normal_only_during_recovery_and_resets_other_dormant_grus():
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
    np.testing.assert_allclose(
        runtime.actors["NORMAL"].calls[-1], np.asarray([.3, 0, 0])
    )
    # NORMAL keeps and advances its hidden state during RECOVERY so its first
    # controlling frame is not a zero-hidden cold start.
    assert runtime.actors["NORMAL"].reset_count == 1
    assert runtime.actors["HIGH_CLIMB"].reset_count == 1

    runtime.router.mode = PolicyMode.NORMAL
    runtime.step(
        np.asarray([.3, 0, 0]), np.zeros(57), np.zeros((1, 41, 33)),
        unused, unused, np.zeros(4), np.zeros(4, bool),
    )
    assert len(runtime.actors["NORMAL"].calls) == 2
    assert runtime.actors["NORMAL"].reset_count == 1

    runtime.router.mode = PolicyMode.LOW_STEP_SEQUENCE
    runtime.step(
        np.asarray([1.0, .05, -.1]), np.zeros(57), np.zeros((1, 41, 33)),
        unused, unused, np.zeros(4), np.zeros(4, bool),
    )
    np.testing.assert_allclose(
        runtime.actors["LOW_STEP_SEQUENCE"].calls[-1], [.6, .05, -.1]
    )


@pytest.mark.parametrize('command', [[0, 0, 0], [-1, 0, 0], [0, .6, 0], [0, 0, -1]])
def test_high_operator_handoff_delivers_command_to_normal_immediately(command):
    from types import SimpleNamespace
    contract = dict(height_split_m=.16, forward_min_x=.1, max_abs_y=.1,
                    max_abs_yaw=.1, detector_confirm_s=.04, attempt_timeout_s=12.,
                    successor_search_s=.2, recovery_hold_s=.2,
                    recovery_timeout_s=5., recovery_base_height_m=.35)
    router = S10TeacherSkillRouter(contract, .02)
    router.mode = PolicyMode.HIGH_CLIMB
    router.locked_edge_center_w = np.zeros(3)
    router.locked_direction_w = np.array([1., 0.])
    class Actor:
        def __init__(self):
            self.commands = []
        def reset(self):
            pass
        def __call__(self, command, *_):
            self.commands.append(np.asarray(command).copy())
            return np.zeros(16)
    runtime = object.__new__(TeacherSkillRuntime)
    runtime.router = router
    runtime.actors = {mode.name: Actor() for mode in PolicyMode}
    runtime.low_height_corridor_half_width_m = 0.
    runtime.step(command, np.zeros(57), np.zeros((1, 41, 33)), None,
                 SimpleNamespace(), np.zeros(4), np.zeros(4, bool))
    assert router.mode == PolicyMode.NORMAL
    assert router.last_transition_reason == 'high_command_handoff'
    assert not router.just_entered_recovery
    np.testing.assert_array_equal(runtime.actors['NORMAL'].commands[0], np.asarray(command, np.float32))
    assert not runtime.actors['RECOVERY'].commands


def test_high_attempt_timeout_still_enters_recovery():
    from types import SimpleNamespace
    contract = dict(height_split_m=.16, forward_min_x=.1, max_abs_y=.1,
                    max_abs_yaw=.1, detector_confirm_s=.04, attempt_timeout_s=12.,
                    successor_search_s=.2, recovery_hold_s=.2,
                    recovery_timeout_s=5., recovery_base_height_m=.35)
    router = S10TeacherSkillRouter(contract, .02)
    router.mode = PolicyMode.HIGH_CLIMB
    router.mode_steps = router.attempt_steps - 1
    assert router.select([.4, 0, 0], SimpleNamespace(has_target=False),
                         SimpleNamespace(), np.zeros(4), np.zeros(4, bool)) == PolicyMode.RECOVERY
    assert router.last_transition_reason == 'attempt_timeout'
