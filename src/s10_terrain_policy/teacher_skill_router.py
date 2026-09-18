"""Hash-pinned recurrent teacher Actors and deterministic MuJoCo skill Router."""

from __future__ import annotations

import hashlib
import json
import math
from enum import IntEnum
from pathlib import Path

import numpy as np


class PolicyMode(IntEnum):
    NORMAL = 0
    HIGH_CLIMB = 1
    LOW_STEP_SEQUENCE = 2
    RECOVERY = 3


ROLE_NAMES = tuple(mode.name for mode in PolicyMode)


def clamp_height_corridor(height_map, half_width_m):
    """Clamp lateral sampling coordinates; retain all 41 forward rows.

    The frozen grid spans y=-1.6..1.6 at 0.1 m. Boundaries between
    columns use linear interpolation, so e.g. 0.25 means exactly 0.25 m.
    Zero disables the transform. Never mutate the raw scanner input.
    """
    if not np.isfinite(half_width_m) or not 0.0 <= half_width_m <= 1.6:
        raise ValueError("corridor half width must be finite and within [0, 1.6] m")
    if half_width_m == 0.0:
        return height_map
    height = np.asarray(height_map)
    if height.shape[-2:] != (41, 33):
        raise ValueError(f"expected height map ending in (41, 33), got {height.shape}")
    y = np.linspace(-1.6, 1.6, 33)
    query = np.clip(y, -half_width_m, half_width_m)
    return np.asarray([np.interp(query, y, row) for row in height.reshape(-1, 33)],
                      dtype=height.dtype).reshape(height.shape)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clamp_height_window(height_map, half_width_m=0., x_range=None):
    height = clamp_height_corridor(height_map, half_width_m)
    if x_range is None:
        return height
    if (len(x_range) != 2 or not np.isfinite(x_range).all()
            or not -.8 <= x_range[0] < x_range[1] <= 3.2):
        raise ValueError("height X range must satisfy -0.8 <= min < max <= 3.2")
    x = np.linspace(-.8, 3.2, 41)
    query = np.clip(x, *x_range)
    transposed = np.asarray(height).swapaxes(-1, -2)
    result = np.asarray([np.interp(query, x, row) for row in transposed.reshape(-1, 41)],
                        dtype=transposed.dtype).reshape(transposed.shape)
    return result.swapaxes(-1, -2).copy()


class S10LowCommandAdapter:
    """Reproduce the frozen model99 LowGoalCommand input contract.

    LOW follows the world crossing direction captured when the Router locks a
    tread.  The direction is transformed into the current body frame on every
    control step, then converted to ``vx, vy, wz`` with model99's original
    gain, limit and smoothing constants.  The operator command itself remains
    unchanged for routing and for every other Actor.
    """

    def __init__(self, dt, max_speed_mps=.6, yaw_gain=.5, yaw_limit=.5,
                 smoothing_tau_s=.2):
        self.dt = float(dt)
        self.max_speed_mps = float(max_speed_mps)
        self.yaw_gain = float(yaw_gain)
        self.yaw_limit = float(yaw_limit)
        self.smoothing_tau_s = float(smoothing_tau_s)
        if self.dt <= 0. or self.smoothing_tau_s <= 0.:
            raise ValueError("LOW command adapter dt and smoothing tau must be positive")
        if self.max_speed_mps <= 0. or self.yaw_limit <= 0.:
            raise ValueError("LOW command adapter speed and yaw limit must be positive")
        self.alpha = -math.expm1(-self.dt / self.smoothing_tau_s)
        self.reset()

    def reset(self):
        self.active = False
        self.locked_direction_w = np.zeros(2, np.float64)
        self.command_b = np.zeros(3, np.float64)

    def update(self, raw_command_b, base_rotation_w, active, latched_direction_w):
        raw = np.asarray(raw_command_b, np.float64).reshape(3)
        if not active:
            self.reset()
            return np.zeros(3, np.float32)

        candidate = np.asarray(latched_direction_w, np.float64).reshape(-1)[:2]
        norm = float(np.linalg.norm(candidate))
        if not self.active:
            if norm <= 1.e-6:
                raise RuntimeError("LOW entered without a valid latched world direction")
            self.locked_direction_w = candidate / norm

        rotation = np.asarray(base_rotation_w, np.float64).reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        direction_b = np.asarray([
            cosine * self.locked_direction_w[0] + sine * self.locked_direction_w[1],
            -sine * self.locked_direction_w[0] + cosine * self.locked_direction_w[1],
        ])
        heading_error = float(np.arctan2(direction_b[1], direction_b[0]))
        speed = min(float(np.linalg.norm(raw[:2])), self.max_speed_mps)
        desired = np.asarray([
            speed * math.cos(heading_error),
            speed * math.sin(heading_error),
            np.clip(self.yaw_gain * heading_error, -self.yaw_limit, self.yaw_limit),
        ])
        smoothed = self.command_b + self.alpha * (desired - self.command_b)
        smoothed_xy_norm = float(np.linalg.norm(smoothed[:2]))
        if smoothed_xy_norm > 1.e-9:
            smoothed[:2] *= speed / smoothed_xy_norm
        next_command = desired if not self.active else smoothed
        self.command_b = next_command
        self.active = True
        return next_command.astype(np.float32)


class RecurrentOnnxActor:
    def __init__(self, path: Path, *, intra_op_threads=None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for Router playback") from exc
        self.path = path
        options = ort.SessionOptions()
        if intra_op_threads is not None:
            if intra_op_threads < 1:
                raise ValueError("Actor thread count must be positive")
            options.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        input_shapes = {item.name: item.shape for item in self.session.get_inputs()}
        output_shapes = {item.name: item.shape for item in self.session.get_outputs()}
        expected_inputs = {"command", "proprio", "height_map", "h_prev"}
        if set(input_shapes) != expected_inputs:
            raise ValueError(f"{path} has incompatible inputs: {input_shapes}")
        if set(output_shapes) != {"actions", "h_next"}:
            raise ValueError(f"{path} has incompatible outputs: {output_shapes}")
        self.hidden = np.zeros((1, 128), np.float32)

    def reset(self) -> None:
        self.hidden.fill(0.0)

    def __call__(self, command, proprio, height_map) -> np.ndarray:
        action, hidden = self.session.run(
            ["actions", "h_next"],
            {
                "command": np.asarray(command, np.float32).reshape(1, 3),
                "proprio": np.asarray(proprio, np.float32).reshape(1, 57),
                "height_map": np.asarray(height_map, np.float32).reshape(1, 1, 41, 33),
                "h_prev": self.hidden,
            },
        )
        if action.shape != (1, 16) or hidden.shape != (1, 128):
            raise RuntimeError(
                f"invalid recurrent outputs: action={action.shape}, hidden={hidden.shape}"
            )
        if not np.isfinite(action).all() or not np.isfinite(hidden).all():
            raise RuntimeError("teacher Actor produced non-finite output")
        max_abs = float(np.max(np.abs(action)))
        if max_abs > 100.0 + 1.0e-3:
            raise RuntimeError(
                f"teacher Actor exceeds raw action safety bound: {max_abs:.6g}"
            )
        self.hidden = hidden.astype(np.float32, copy=True)
        return action[0].astype(np.float32, copy=False)


class S10TeacherSkillRouter:
    def __init__(self, contract: dict, dt: float) -> None:
        self.dt = float(dt)
        self.height_split = float(contract["height_split_m"])
        self.forward_min = float(contract["forward_min_x"])
        self.max_y = float(contract["max_abs_y"])
        self.max_yaw = float(contract["max_abs_yaw"])
        self.confirm_steps = max(1, round(float(contract["detector_confirm_s"]) / dt))
        self.attempt_steps = max(1, round(float(contract["attempt_timeout_s"]) / dt))
        self.successor_steps = max(1, round(float(contract["successor_search_s"]) / dt))
        self.recovery_hold_steps = max(1, round(float(contract["recovery_hold_s"]) / dt))
        self.recovery_timeout_steps = max(1, round(float(contract["recovery_timeout_s"]) / dt))
        self.recovery_base_height = float(contract["recovery_base_height_m"])
        self.reset()

    def reset(self) -> None:
        self.mode = PolicyMode.NORMAL
        self.arm_steps = 0
        self.mode_steps = 0
        self.successor_wait_steps = 0
        self.recovery_stable_steps = 0
        self.locked_edge_center_w = None
        self.locked_direction_w = None
        self.locked_upper_z_w = 0.0
        self.wheel_support_steps = np.zeros(4, dtype=np.int32)
        self.wheel_confirmed = np.zeros(4, dtype=bool)
        self.pending_successor = None
        self.high_successor_steps = 0
        self.high_front_support_steps = 0
        self.just_entered_recovery = False
        self.normal_handoff_settling = False
        self.straddle_steps = 0
        self.normal_straddle_steps = 0
        self.last_entry_reason = None
        self.last_transition_reason = None

    def _command_in_skill(self, command) -> bool:
        return bool(
            command[0] > self.forward_min
            and abs(command[1]) <= self.max_y
            and abs(command[2]) <= self.max_yaw
        )

    def high_successor_pending(self) -> bool:
        """Whether LOW has already seen the next edge that belongs to HIGH."""
        return bool(
            self.mode == PolicyMode.LOW_STEP_SEQUENCE
            and self.pending_successor is not None
            and self.pending_successor.height_m >= self.height_split
        )

    @staticmethod
    def _strict_zero(command) -> bool:
        return bool(np.all(np.abs(command) <= 1.0e-6))

    def _lock(self, detection, base_rotation_w, command_b) -> None:
        self._lock_target(detection.edge_segment_w.mean(axis=0),
                          float(detection.upper_z_w), base_rotation_w, command_b)
        self.last_entry_reason = "forward_detector"

    def _lock_target(self, center_w, upper_z_w, base_rotation_w, command_b) -> None:
        self.locked_edge_center_w = np.asarray(center_w, float).copy()
        rotation = np.asarray(base_rotation_w, np.float64).reshape(3, 3)
        command_xy = np.asarray(command_b, np.float64).reshape(3)[:2]
        # Match Isaac's command_world contract: the accepted body-frame XY
        # command, not merely the chassis forward axis, becomes immutable in
        # world coordinates for this target lifecycle.
        direction = rotation[:2, :2] @ command_xy
        direction /= max(1.0e-9, float(np.linalg.norm(direction)))
        self.locked_direction_w = direction
        self.locked_upper_z_w = float(upper_z_w)
        self.mode_steps = 0
        self.successor_wait_steps = 0
        self.wheel_support_steps.fill(0)
        self.wheel_confirmed.fill(False)
        self.pending_successor = None
        self.high_successor_steps = 0
        self.high_front_support_steps = 0

    def _enter_recovery(self, reason: str) -> None:
        self.mode = PolicyMode.RECOVERY
        self.mode_steps = 0
        self.recovery_stable_steps = 0
        self.just_entered_recovery = True
        self.last_transition_reason = reason

    def _complete_to_normal(self, reason: str) -> None:
        """Finish a physically confirmed climb without invoking RECOVERY."""
        self.mode = PolicyMode.NORMAL
        self.mode_steps = 0
        self.arm_steps = 0
        self.successor_wait_steps = 0
        self.locked_edge_center_w = None
        self.locked_direction_w = None
        self.pending_successor = None
        self.high_successor_steps = 0
        self.high_front_support_steps = 0
        self.normal_handoff_settling = False
        self.last_transition_reason = reason

    def _recovery_ready(self, state) -> bool:
        rotation = np.asarray(state.base_rotation_w, np.float64).reshape(3, 3)
        pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        return bool(
            state.base_pos_w[2] >= self.recovery_base_height
            and abs(roll) <= np.deg2rad(10.0)
            and abs(pitch) <= np.deg2rad(15.0)
            and abs(state.base_lin_vel_b[2]) <= 0.15
            and np.linalg.norm(state.base_ang_vel_b[:2]) <= 0.50
            # A four-wheel posture can still be sliding or yawing on the top
            # tread.  Handing that state directly to a cold NORMAL actor is
            # the visible post-climb turn seen in GUI playback.  Reuse the
            # Router's deployed pure-forward envelope as the handoff gate.
            and abs(state.base_lin_vel_b[1]) <= self.max_y
            and abs(state.base_ang_vel_b[2]) <= self.max_yaw
        )

    def select(
        self, command, detection, state, wheel_pos_z, wheel_contact, wheel_support_z=None
    ) -> PolicyMode:
        command = np.asarray(command, np.float64).reshape(3)
        self.just_entered_recovery = False
        self.normal_handoff_settling = False
        command_in_skill = self._command_in_skill(command)
        strict_zero = self._strict_zero(command)
        # 轮下几何用于判断跨层支撑；不能仅凭机身俯仰推断台阶。
        split_support = False
        if wheel_support_z is not None:
            tread = np.asarray(wheel_support_z, float).reshape(4)
            split_support = bool(np.isfinite(tread).all()
                                 and np.min(tread[:2]) - np.min(tread[2:]) >= .04)
        self.straddle_steps = self.straddle_steps + 1 if split_support else 0
        if self.mode == PolicyMode.NORMAL:
            # A tread already beneath the chassis is outside the forward fan.
            # Bootstrap LOW or HIGH from current physical support, without
            # pause history.  The edge may already be behind the forward fan
            # once both front wheels are on the upper tread.  Require grounded
            # front wheels so lifted wheels over a lower floor cannot enter.
            normal_straddle = bool(
                command_in_skill and split_support
                and np.asarray(wheel_contact, bool).reshape(4)[:2].all()
                and np.all(np.abs(np.asarray(wheel_pos_z)[:2] - tread[:2] - .11) <= .05)
            )
            self.normal_straddle_steps = self.normal_straddle_steps + 1 if normal_straddle else 0
            if self.normal_straddle_steps >= max(3, self.confirm_steps):
                straddle_height = np.max(tread[:2]) - np.min(tread[2:])
                self.mode = (
                    PolicyMode.HIGH_CLIMB
                    if straddle_height >= self.height_split
                    else PolicyMode.LOW_STEP_SEQUENCE
                )
                # This is an under-body progress anchor, not a detected edge.
                # It supplies successor-distance bookkeeping and the actual
                # front tread height for rear-wheel completion checks.
                self._lock_target(
                    state.base_pos_w,
                    np.max(tread[:2]),
                    state.base_rotation_w,
                    command,
                )
                self.last_entry_reason = (
                    "normal_high_treads"
                    if self.mode == PolicyMode.HIGH_CLIMB
                    else "normal_low_treads"
                )
                self.arm_steps = 0
                self.normal_straddle_steps = 0
                return self.mode
            if detection.has_target and command_in_skill:
                self.arm_steps += 1
                if self.arm_steps >= self.confirm_steps:
                    self.mode = (
                        PolicyMode.HIGH_CLIMB
                        if detection.height_m >= self.height_split
                        else PolicyMode.LOW_STEP_SEQUENCE
                    )
                    self._lock(detection, state.base_rotation_w, command)
                    self.arm_steps = 0
            else:
                self.arm_steps = 0
            return self.mode

        self.normal_straddle_steps = 0
        if self.mode in (PolicyMode.HIGH_CLIMB, PolicyMode.LOW_STEP_SEQUENCE):
            self.mode_steps += 1
            # Both climbing experts yield operator command changes to NORMAL.
            # Any command outside the pure-forward
            # envelope belongs to NORMAL, including zero, reverse, lateral and
            # yaw commands; changing operator intent is not a recovery failure.
            # 人工松键/倒车/横移/转向立即归还NORMAL，只有尝试超时进入RECOVERY。
            if strict_zero or not command_in_skill:
                reason = ("high_command_handoff" if self.mode == PolicyMode.HIGH_CLIMB
                          else "low_command_handoff")
                self._complete_to_normal(reason)
                return self.mode
            # HIGH连续检测确认后冻结目标；视野丢失不撤销已确认的交接。
            high_locked = (
                self.high_successor_pending()
                and self.high_successor_steps >= self.confirm_steps
            )
            if not high_locked:
                high_seen = False
                if detection.has_target:
                    next_center = detection.edge_segment_w.mean(axis=0)
                    advance = float(
                        (next_center[:2] - self.locked_edge_center_w[:2])
                        @ self.locked_direction_w
                    )
                    if advance >= 0.20:
                        previous = self.pending_successor
                        high_seen = bool(
                            self.mode == PolicyMode.LOW_STEP_SEQUENCE
                            and detection.height_m >= self.height_split
                        )
                        same_high = bool(
                            high_seen and previous is not None
                            and previous.height_m >= self.height_split
                            and np.linalg.norm(next_center[:2]
                                - previous.edge_segment_w.mean(axis=0)[:2]) <= .15
                            and abs(detection.upper_z_w - previous.upper_z_w) <= .025
                        )
                        self.pending_successor = detection
                        self.high_successor_steps = (
                            self.high_successor_steps + 1 if same_high
                            else 1 if high_seen else 0
                        )
                if not high_seen:
                    if self.high_successor_pending():
                        self.pending_successor = None
                    self.high_successor_steps = 0
                high_locked = (
                    self.high_successor_pending()
                    and self.high_successor_steps >= self.confirm_steps
                )

            # LOW负责接近；双前轮必须同时接触锁存的HIGH顶面并连续确认3拍。
            # 后轮不参与此门，避免紧邻台阶上等待四轮LOW完成而死锁。
            front_supported = False
            if high_locked and wheel_support_z is not None:
                upper_z = self.pending_successor.upper_z_w
                front_supported = bool(
                    np.asarray(wheel_contact, bool)[:2].all()
                    and np.isfinite(tread[:2]).all()
                    and np.all(np.abs(tread[:2] - upper_z) <= .025)
                    and np.all(np.abs(np.asarray(wheel_pos_z)[:2]
                                      - tread[:2] - .11) <= .05)
                )
            self.high_front_support_steps = (
                self.high_front_support_steps + 1 if front_supported else 0
            )
            if self.high_front_support_steps >= 3:
                successor = self.pending_successor
                self.mode = PolicyMode.HIGH_CLIMB
                self._lock(successor, state.base_rotation_w, command)
                self.last_entry_reason = "low_to_high_successor"
                self.last_transition_reason = "confirmed_high_front_support"
                return self.mode

            wheel_pos_z = np.asarray(wheel_pos_z, np.float64).reshape(4)
            wheel_contact = np.asarray(wheel_contact, bool).reshape(4)
            # Wheel radius is approximately 0.11 m. A contacted wheel centre
            # at least 6 cm above the locked tread has genuinely reached that
            # tread; three policy frames reject one-frame contact spikes.
            support_now = wheel_contact & (
                wheel_pos_z >= self.locked_upper_z_w + 0.06
            )
            if wheel_support_z is not None:
                support_now &= np.isfinite(tread) & (tread >= self.locked_upper_z_w - .025)
            self.wheel_support_steps = np.where(
                support_now, self.wheel_support_steps + 1, 0
            )
            self.wheel_confirmed |= self.wheel_support_steps >= 3
            edge_complete = bool(self.wheel_confirmed.all())
            attempt_timed_out = self.mode_steps >= self.attempt_steps
            if edge_complete:
                # A released/incompatible command or the attempt deadline must
                # not turn an already completed, level landing into a failure.
                # This is the normal top-of-stair stop path.
                if not split_support and attempt_timed_out:
                    self._complete_to_normal("confirmed_level_support")
                    return self.mode

                # On continuous stairs the front axle can already be on the
                # next tread while the rear axle finishes the locked one.  The
                # split-support posture is evidence for the successor, not a
                # reason to block target progress.
                if (
                    self.pending_successor is not None
                    and self.mode == PolicyMode.LOW_STEP_SEQUENCE
                    and self.pending_successor.height_m < self.height_split
                    and command_in_skill
                ):
                    self._lock(
                        self.pending_successor, state.base_rotation_w, command
                    )
                    self.last_transition_reason = "confirmed_low_successor"
                elif high_locked:
                    pass  # Await front support on the latched HIGH tread.
                elif self.pending_successor is not None and not split_support:
                    self._complete_to_normal("successor_requires_reroute")
                elif not split_support:
                    self.normal_handoff_settling = True
                    self.successor_wait_steps += 1
                    if self.successor_wait_steps >= self.successor_steps:
                        self._complete_to_normal("confirmed_final_tread")
            if (self.mode in (PolicyMode.HIGH_CLIMB, PolicyMode.LOW_STEP_SEQUENCE)
                    and attempt_timed_out):
                self._enter_recovery("attempt_timeout")
            return self.mode

        self.mode_steps += 1
        if self._recovery_ready(state):
            self.recovery_stable_steps += 1
        else:
            self.recovery_stable_steps = 0
        # Keep the operator command queued, but do not let a held-forward key
        # bypass the physical stability contract after a climb.  RECOVERY sees
        # zero command below and hands off after a continuous stable window.
        if (
            self.recovery_stable_steps >= self.recovery_hold_steps
            or self.mode_steps >= self.recovery_timeout_steps
        ):
            self.mode = PolicyMode.NORMAL
        if self.mode == PolicyMode.NORMAL:
            self.mode_steps = 0
            self.arm_steps = 0
            self.locked_edge_center_w = None
            self.locked_direction_w = None
        return self.mode


class TeacherSkillRuntime:
    def __init__(self, bundle_path: Path, dt: float, *, low_height_corridor_half_width_m=0.0,
                 low_height_x_range=None, actor_threads=None) -> None:
        clamp_height_window(np.zeros((1, 41, 33), np.float32), low_height_corridor_half_width_m, low_height_x_range)
        self.low_height_corridor_half_width_m = float(low_height_corridor_half_width_m)
        self.low_height_x_range = low_height_x_range
        bundle_path = bundle_path.expanduser().resolve()
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        if payload.get("kind") != "s10-goai-teacher-router-bundle":
            raise ValueError(f"invalid teacher Router bundle: {bundle_path}")
        if payload.get("actor_protocol") != "s10-asymmetric-cnn-gru-1413-v1":
            raise ValueError(f"unsupported actor protocol: {payload.get('actor_protocol')!r}")
        self.bundle_path = bundle_path
        self.detector_contract = payload["detector"]
        self.router = S10TeacherSkillRouter(payload["router"], dt)
        router_contract = payload["router"]
        self.high_forward_command_max_mps = float(
            router_contract.get("high_forward_command_max_mps", 0.4)
        )
        if not np.isfinite(self.high_forward_command_max_mps) or self.high_forward_command_max_mps <= 0.0:
            raise ValueError("HIGH forward command cap must be finite and positive")
        self.low_forward_command_max_mps = float(
            router_contract.get("low_forward_command_max_mps", 0.6)
        )
        if self.low_forward_command_max_mps <= 0.0:
            raise ValueError("Low forward command cap must be positive")
        adapter_contract = router_contract.get("low_command_adapter")
        if adapter_contract is None:
            self.low_command_adapter = None
        elif adapter_contract == "model99_latched_world_direction_v1":
            self.low_command_adapter = S10LowCommandAdapter(
                dt,
                max_speed_mps=self.low_forward_command_max_mps,
                yaw_gain=float(router_contract.get("low_command_yaw_gain", .5)),
                yaw_limit=float(router_contract.get("low_command_yaw_limit", .5)),
                smoothing_tau_s=float(
                    router_contract.get("low_command_smoothing_tau_s", .2)
                ),
            )
        else:
            raise ValueError(f"unsupported LOW command adapter: {adapter_contract!r}")
        self.actors = {}
        for role in ROLE_NAMES:
            entry = payload["skills"][role]
            path = (bundle_path.parent / entry["path"]).resolve()
            if _sha256(path) != entry["sha256"]:
                raise ValueError(f"{role} ONNX hash mismatch: {path}")
            self.actors[role] = RecurrentOnnxActor(path, intra_op_threads=actor_threads)

    def reset(self) -> None:
        self.router.reset()
        if self.low_command_adapter is not None:
            self.low_command_adapter.reset()
        for actor in self.actors.values():
            actor.reset()

    def step(
        self,
        command,
        proprio,
        height_map,
        detection,
        state,
        wheel_pos_z,
        wheel_contact,
        *, low_height_map=None, high_height_map=None, wheel_support_z=None,
    ) -> np.ndarray:
        router_args = (command, detection, state, wheel_pos_z, wheel_contact)
        mode = (self.router.select(*router_args) if wheel_support_z is None
                else self.router.select(*router_args, wheel_support_z))
        # 各Actor独立持有GRU状态。休眠者清零；NORMAL在完成等待/恢复时预热，
        # HIGH在待交接目标出现后预热，但只有当前模式的动作会发送给机器人。
        normal_handoff_settling = bool(
            getattr(self.router, "normal_handoff_settling", False)
        )
        high_successor_pending = getattr(
            self.router, "high_successor_pending", None
        )
        high_handoff_arming = bool(
            mode == PolicyMode.LOW_STEP_SEQUENCE
            and high_successor_pending is not None
            and high_successor_pending()
        )
        for role, actor in self.actors.items():
            normal_shadow = role == "NORMAL" and (
                mode == PolicyMode.RECOVERY
                or normal_handoff_settling
            )
            high_shadow = role == "HIGH_CLIMB" and high_handoff_arming
            if role != mode.name and not normal_shadow and not high_shadow:
                actor.reset()
        low_command_adapter = getattr(self, "low_command_adapter", None)
        if mode == PolicyMode.RECOVERY:
            active_command = np.zeros(3, np.float32)
        else:
            active_command = np.asarray(command, np.float32).copy()
            if mode == PolicyMode.HIGH_CLIMB:
                # Adapt only the expert input; routing and NORMAL shadow keep
                # the user's raw command. Zero/reverse/lateral/yaw are unchanged.
                active_command[0] = min(
                    float(active_command[0]),
                    getattr(self, "high_forward_command_max_mps", 0.4),
                )
            if mode == PolicyMode.LOW_STEP_SEQUENCE:
                if low_command_adapter is None:
                    active_command[0] = min(
                        float(active_command[0]), self.low_forward_command_max_mps
                    )
                else:
                    active_command = low_command_adapter.update(
                        command,
                        state.base_rotation_w,
                        True,
                        self.router.locked_direction_w,
                    )
        if mode != PolicyMode.LOW_STEP_SEQUENCE and low_command_adapter is not None:
            low_command_adapter.update(command, state.base_rotation_w, False, None)
        actor_height = height_map
        # NORMAL/LOW use the <=18 cm connected surface. HIGH receives a
        # separately selected <=detector-max surface so a 23--45 cm tread under
        # an overhead structure does not fall back to the first raw ray hit.
        if mode in (PolicyMode.NORMAL, PolicyMode.LOW_STEP_SEQUENCE):
            actor_height = clamp_height_window(
                low_height_map if low_height_map is not None else height_map,
                self.low_height_corridor_half_width_m,
                getattr(self, "low_height_x_range", None),
            )
        elif mode == PolicyMode.HIGH_CLIMB:
            actor_height = clamp_height_window(
                high_height_map if high_height_map is not None
                else low_height_map if low_height_map is not None
                else height_map,
                self.low_height_corridor_half_width_m,
                getattr(self, "low_height_x_range", None),
            )
        self.last_actor_height_map = actor_height
        if mode == PolicyMode.RECOVERY:
            normal_height = clamp_height_window(
                low_height_map if low_height_map is not None else height_map,
                self.low_height_corridor_half_width_m,
                getattr(self, "low_height_x_range", None),
            )
            self.actors["NORMAL"](
                np.asarray(command, np.float32), proprio, normal_height
            )
        elif normal_handoff_settling:
            self.actors["NORMAL"](
                np.asarray(command, np.float32), proprio, actor_height
            )
        if high_handoff_arming:
            high_command = np.asarray(command, np.float32).copy()
            high_command[0] = min(
                float(high_command[0]),
                getattr(self, "high_forward_command_max_mps", 0.4),
            )
            high_height = clamp_height_window(
                high_height_map if high_height_map is not None
                else low_height_map if low_height_map is not None
                else height_map,
                self.low_height_corridor_half_width_m,
                getattr(self, "low_height_x_range", None),
            )
            self.actors["HIGH_CLIMB"](high_command, proprio, high_height)
        return self.actors[mode.name](active_command, proprio, actor_height)
