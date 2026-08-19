"""S10 privileged-teacher protocol implemented directly on MuJoCo state."""
from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np

ROBOT_ORDER = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
]
POLICY_ORDER = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]
ROBOT_TO_POLICY = np.asarray([ROBOT_ORDER.index(name) for name in POLICY_ORDER])
POLICY_TO_ROBOT = np.asarray([POLICY_ORDER.index(name) for name in ROBOT_ORDER])
DEFAULT_POLICY = np.asarray(
    [0, -.3, .6, 0, -.3, .6, 0, .3, -.6, 0, .3, -.6, 0, 0, 0, 0], np.float32)
DEFAULT_ROBOT = np.asarray(
    [0, -.3, .6, 0, 0, -.3, .6, 0, 0, .3, -.6, 0, 0, .3, -.6, 0], np.float32)
ACTION_SCALE_ROBOT = np.asarray([.125, .25, .25, 5.0] * 4, np.float32)
JOINT_DIR = np.asarray([1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1], np.float32)
POS_OFFSET_RAD = np.deg2rad(np.asarray(
    [-35, -145, 156, 0, 35, -145, 156, 0, -35, 145, -156, 0, 35, 145, -156, 0],
    np.float32))
JOINT_INIT_RAW = np.asarray(
    [-.438, -1.16, 2.76, 0, .438, -1.16, 2.76, 0,
     -.438, 1.16, -2.76, 0, .438, 1.16, -2.76, 0], np.float64)


@dataclass(frozen=True)
class S10PolicyState:
    base_pos_w: np.ndarray
    base_rotation_w: np.ndarray
    base_lin_vel_b: np.ndarray
    base_ang_vel_b: np.ndarray
    joint_pos_robot: np.ndarray
    joint_vel_robot: np.ndarray


def state_from_mujoco(model, data, base_body_id) -> S10PolicyState:
    spatial = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, spatial, 1)
    rotation = data.xmat[base_body_id].reshape(3, 3).copy()
    raw_pos = data.qpos[7:23]
    raw_vel = data.qvel[6:22]
    return S10PolicyState(
        data.xpos[base_body_id].copy(), rotation,
        spatial[3:6].astype(np.float32), spatial[:3].astype(np.float32),
        ((raw_pos - POS_OFFSET_RAD) * JOINT_DIR).astype(np.float32),
        (raw_vel * JOINT_DIR).astype(np.float32))


def assemble_official_57(state: S10PolicyState, command_raw, last_action_norm):
    command = np.asarray(command_raw, np.float32).reshape(3)
    command = np.clip(command, [-1.0, -.6, -1.0], [1.0, .6, 1.0])
    joint_pos = state.joint_pos_robot[ROBOT_TO_POLICY].copy()
    joint_vel = state.joint_vel_robot[ROBOT_TO_POLICY].copy()
    joint_pos[12:16] = 0.0
    obs = np.concatenate([
        state.base_ang_vel_b * .25,
        state.base_rotation_w.T @ np.asarray([0, 0, -1], np.float32),
        command,
        joint_pos - DEFAULT_POLICY,
        joint_vel * .05,
        np.asarray(last_action_norm, np.float32).reshape(16),
    ]).astype(np.float32)
    if obs.shape != (57,) or not np.isfinite(obs).all():
        raise RuntimeError(f"invalid official proprio shape/value: {obs.shape}")
    return obs


def assemble_teacher_1413(state, proprio_57, privileged_height):
    obs = np.concatenate([
        state.base_lin_vel_b,
        np.asarray(proprio_57, np.float32).reshape(57),
        np.asarray(privileged_height, np.float32).reshape(1353),
    ]).astype(np.float32)
    if obs.shape != (1413,) or not np.isfinite(obs).all():
        raise RuntimeError(f"invalid teacher observation: {obs.shape}")
    return obs


def decode_action_norm(action_norm):
    action = np.asarray(action_norm, np.float32).reshape(16)
    physical = action[POLICY_TO_ROBOT] * ACTION_SCALE_ROBOT + DEFAULT_ROBOT
    goal_pos = np.zeros(16, np.float32)
    goal_vel = np.zeros(16, np.float32)
    for leg in range(4):
        start = 4 * leg
        goal_pos[start:start + 3] = physical[start:start + 3]
        goal_vel[start + 3] = physical[start + 3]
    return goal_pos, goal_vel


def published_targets_to_raw(goal_pos, goal_vel):
    return (
        np.asarray(goal_pos, np.float32) * JOINT_DIR + POS_OFFSET_RAD,
        np.asarray(goal_vel, np.float32) * JOINT_DIR,
    )


class PrivilegedHeightScanner:
    """Exact Isaac GridPattern equivalent: 41x33 vertical rays, yaw aligned."""
    X = np.linspace(-.8, 3.2, 41, dtype=np.float64)
    Y = np.linspace(-1.6, 1.6, 33, dtype=np.float64)

    def __init__(self, model, *, body_exclude=-1):
        self.model = model
        gx, gy = np.meshgrid(self.X, self.Y, indexing="xy")
        self.local_xy = np.column_stack([gx.reshape(-1), gy.reshape(-1)])
        if self.local_xy.shape != (1353, 2):
            raise RuntimeError("privileged grid must contain 1353 rays")
        # Static terrain only: group0 on, robot group1 and overlay group2 off.
        self.geomgroup = np.asarray([1, 0, 0, 1, 1, 1], dtype=np.uint8)
        self.body_exclude = int(body_exclude)
        self.direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)

    def scan(self, data, base_pos_w, base_rotation_w):
        pos = np.asarray(base_pos_w, np.float64).reshape(3)
        rotation = np.asarray(base_rotation_w, np.float64).reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        xy_w = self.local_xy @ np.asarray([[c, s], [-s, c]]) + pos[:2]
        starts = np.column_stack([xy_w, np.full(1353, pos[2] + 20.0)])
        distances = np.full(1353, -1.0, dtype=np.float64)
        geom_ids = np.full(1353, -1, dtype=np.int32)
        geom_out = np.empty(1, dtype=np.int32)
        for index, start in enumerate(starts):
            geom_out[0] = -1
            distances[index] = mujoco.mj_ray(
                self.model, data, start, self.direction, self.geomgroup,
                True, self.body_exclude, geom_out)
            geom_ids[index] = geom_out[0]
        hit = (geom_ids >= 0) & (distances >= 0.0)
        height = np.full(1353, -1.0, dtype=np.float32)
        # hit_z = base_z + 20 - distance; Isaac term = base_z - hit_z - 0.5.
        height[hit] = np.clip(distances[hit] - 20.5, -1.0, 1.0)
        return height, hit, geom_ids
