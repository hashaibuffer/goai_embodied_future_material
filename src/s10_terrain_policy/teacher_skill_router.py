"""Hash-pinned recurrent teacher Actors and deterministic MuJoCo skill Router."""

from __future__ import annotations

import hashlib
import json
from enum import IntEnum
from pathlib import Path

import numpy as np


class PolicyMode(IntEnum):
    NORMAL = 0
    HIGH_CLIMB = 1
    LOW_STEP_SEQUENCE = 2
    RECOVERY = 3


ROLE_NAMES = tuple(mode.name for mode in PolicyMode)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RecurrentOnnxActor:
    def __init__(self, path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for Router playback") from exc
        self.path = path
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
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
        self.just_entered_recovery = False

    def _command_in_skill(self, command) -> bool:
        return bool(
            command[0] > self.forward_min
            and abs(command[1]) <= self.max_y
            and abs(command[2]) <= self.max_yaw
        )

    @staticmethod
    def _strict_zero(command) -> bool:
        return bool(np.all(np.abs(command) <= 1.0e-6))

    def _lock(self, detection, base_rotation_w) -> None:
        self.locked_edge_center_w = detection.edge_segment_w.mean(axis=0)
        rotation = np.asarray(base_rotation_w, np.float64).reshape(3, 3)
        direction = rotation[:2, 0].copy()
        direction /= max(1.0e-9, float(np.linalg.norm(direction)))
        self.locked_direction_w = direction
        self.locked_upper_z_w = float(detection.upper_z_w)
        self.mode_steps = 0
        self.successor_wait_steps = 0
        self.wheel_support_steps.fill(0)
        self.wheel_confirmed.fill(False)
        self.pending_successor = None

    def _enter_recovery(self) -> None:
        self.mode = PolicyMode.RECOVERY
        self.mode_steps = 0
        self.recovery_stable_steps = 0
        self.just_entered_recovery = True

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
        )

    def select(
        self, command, detection, state, wheel_pos_z, wheel_contact
    ) -> PolicyMode:
        command = np.asarray(command, np.float64).reshape(3)
        self.just_entered_recovery = False
        command_in_skill = self._command_in_skill(command)
        strict_zero = self._strict_zero(command)

        if self.mode == PolicyMode.NORMAL:
            if detection.has_target and command_in_skill:
                self.arm_steps += 1
                if self.arm_steps >= self.confirm_steps:
                    self.mode = (
                        PolicyMode.HIGH_CLIMB
                        if detection.height_m >= self.height_split
                        else PolicyMode.LOW_STEP_SEQUENCE
                    )
                    self._lock(detection, state.base_rotation_w)
                    self.arm_steps = 0
            else:
                self.arm_steps = 0
            return self.mode

        if self.mode in (PolicyMode.HIGH_CLIMB, PolicyMode.LOW_STEP_SEQUENCE):
            self.mode_steps += 1
            if strict_zero or not command_in_skill or self.mode_steps >= self.attempt_steps:
                self._enter_recovery()
                return self.mode

            if detection.has_target:
                next_center = detection.edge_segment_w.mean(axis=0)
                advance = float(
                    (next_center[:2] - self.locked_edge_center_w[:2])
                    @ self.locked_direction_w
                )
                if advance >= 0.20:
                    self.pending_successor = detection

            wheel_pos_z = np.asarray(wheel_pos_z, np.float64).reshape(4)
            wheel_contact = np.asarray(wheel_contact, bool).reshape(4)
            # Wheel radius is approximately 0.11 m. A contacted wheel centre
            # at least 6 cm above the locked tread has genuinely reached that
            # tread; three policy frames reject one-frame contact spikes.
            support_now = wheel_contact & (
                wheel_pos_z >= self.locked_upper_z_w + 0.06
            )
            self.wheel_support_steps = np.where(
                support_now, self.wheel_support_steps + 1, 0
            )
            self.wheel_confirmed |= self.wheel_support_steps >= 3
            if self.wheel_confirmed.all():
                if (
                    self.pending_successor is not None
                    and self.mode == PolicyMode.LOW_STEP_SEQUENCE
                    and self.pending_successor.height_m < self.height_split
                ):
                    self._lock(self.pending_successor, state.base_rotation_w)
                elif self.pending_successor is not None:
                    self._enter_recovery()
                else:
                    self.successor_wait_steps += 1
                    if self.successor_wait_steps >= self.successor_steps:
                        self._enter_recovery()
            return self.mode

        self.mode_steps += 1
        if not strict_zero:
            self.mode = PolicyMode.NORMAL
        else:
            if self._recovery_ready(state):
                self.recovery_stable_steps += 1
            else:
                self.recovery_stable_steps = 0
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
    def __init__(self, bundle_path: Path, dt: float) -> None:
        bundle_path = bundle_path.expanduser().resolve()
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        if payload.get("kind") != "s10-goai-teacher-router-bundle":
            raise ValueError(f"invalid teacher Router bundle: {bundle_path}")
        if payload.get("actor_protocol") != "s10-asymmetric-cnn-gru-1413-v1":
            raise ValueError(f"unsupported actor protocol: {payload.get('actor_protocol')!r}")
        self.bundle_path = bundle_path
        self.detector_contract = payload["detector"]
        self.router = S10TeacherSkillRouter(payload["router"], dt)
        self.actors = {}
        for role in ROLE_NAMES:
            entry = payload["skills"][role]
            path = (bundle_path.parent / entry["path"]).resolve()
            if _sha256(path) != entry["sha256"]:
                raise ValueError(f"{role} ONNX hash mismatch: {path}")
            self.actors[role] = RecurrentOnnxActor(path)

    def reset(self) -> None:
        self.router.reset()
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
    ) -> np.ndarray:
        mode = self.router.select(
            command, detection, state, wheel_pos_z, wheel_contact
        )
        # A recurrent skill owns history only while it owns the robot.  Do not
        # let dormant experts integrate commands/terrain from another skill:
        # keep them at zero state and execute only the selected Actor.
        for role, actor in self.actors.items():
            if role != mode.name:
                actor.reset()
        active_command = (
            np.zeros(3, np.float32)
            if mode == PolicyMode.RECOVERY
            else command
        )
        return self.actors[mode.name](active_command, proprio, height_map)
