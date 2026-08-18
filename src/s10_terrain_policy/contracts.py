"""T00 observation / action contract.

Assembly matches S10PolicyRunner::getRobotAction. Scales, default pose, and
joint permutations live in configs/policy.yaml and must not be reinvented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _as_f32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _as_i64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def load_yaml(name: str, config_dir: Path | None = None) -> dict:
    path = (config_dir or CONFIG_DIR) / name
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def generate_permutation(source: list[str], dest: list[str]) -> list[int]:
    """Same as S10PolicyRunner::generate_permutation(from=source, to=dest)."""
    index = {name: i for i, name in enumerate(source)}
    missing = [name for name in dest if name not in index]
    if missing:
        raise KeyError(f"joint names missing from source order: {missing}")
    return [index[name] for name in dest]


def rpy_to_rm(rpy_rad: np.ndarray) -> np.ndarray:
    """Z-Y-X RPY, matching basic_function.hpp RpyToRm (yaw * pitch * roll)."""
    roll, pitch, yaw = np.asarray(rpy_rad, dtype=np.float64).reshape(3)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def deg2rad(degrees) -> np.ndarray:
    return np.asarray(degrees, dtype=np.float64) * (np.pi / 180.0)


@dataclass(frozen=True)
class RobotAction:
    goal_joint_pos: np.ndarray
    goal_joint_vel: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


class PolicyContract:
    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.track = load_yaml("track.yaml", self.config_dir)
        self.lidar = load_yaml("lidar.yaml", self.config_dir)
        self.heightmap = load_yaml("heightmap.yaml", self.config_dir)
        self.policy_cfg = load_yaml("policy.yaml", self.config_dir)
        self.policy = self.policy_cfg["policy"]

        self.omega_scale = float(self.policy["omega_scale"])
        self.dof_vel_scale = float(self.policy["dof_vel_scale"])
        self.gravity_world = _as_f32(self.policy["gravity_world"])
        self.robot_order = list(self.policy["robot_order"])
        self.policy_order = list(self.policy["policy_order"])
        self.robot2policy_idx = _as_i64(self.policy["robot2policy_idx"])
        self.policy2robot_idx = _as_i64(self.policy["policy2robot_idx"])
        computed_r2p = generate_permutation(self.robot_order, self.policy_order)
        computed_p2r = generate_permutation(self.policy_order, self.robot_order)
        if list(self.robot2policy_idx) != computed_r2p or list(self.policy2robot_idx) != computed_p2r:
            raise ValueError("policy.yaml permutation arrays do not match robot_order/policy_order")
        self.default_pose_policy = _as_f32(self.policy["default_pose_policy"])
        self.default_pose_robot = _as_f32(self.policy["default_pose_robot"])
        self.action_scale_robot = _as_f32(self.policy["action_scale_robot"])
        self.kp_robot = _as_f32(self.policy["kp_robot"])
        self.kd_robot = _as_f32(self.policy["kd_robot"])
        self.obs_slices = {
            name: (int(bounds[0]), int(bounds[1]))
            for name, bounds in self.policy["obs_slices"].items()
        }
        self.command_limits = {}
        for axis in ("vx", "vy", "wz"):
            lo, hi = self.policy_cfg["command"][axis]
            self.command_limits[axis] = (float(lo), float(hi))
        hm = self.heightmap["heightmap"]
        self.policy_nx, self.policy_ny = (int(hm["policy_grid"][0]), int(hm["policy_grid"][1]))

    @classmethod
    def load(cls, config_dir: Path | None = None) -> "PolicyContract":
        return cls(config_dir)

    def clip_command(self, command) -> np.ndarray:
        vx, vy, wz = _as_f32(command).reshape(3)
        vx_lo, vx_hi = self.command_limits["vx"]
        vy_lo, vy_hi = self.command_limits["vy"]
        wz_lo, wz_hi = self.command_limits["wz"]
        return np.array(
            [
                np.clip(vx, vx_lo, vx_hi),
                np.clip(vy, vy_lo, vy_hi),
                np.clip(wz, wz_lo, wz_hi),
            ],
            dtype=np.float32,
        )

    def rotation_from_imu(
        self,
        *,
        base_rot_mat: np.ndarray | None = None,
        base_rpy_rad: np.ndarray | None = None,
        imu_rpy_deg: np.ndarray | None = None,
    ) -> np.ndarray:
        provided = sum(value is not None for value in (base_rot_mat, base_rpy_rad, imu_rpy_deg))
        if provided != 1:
            raise ValueError("provide exactly one of base_rot_mat, base_rpy_rad, imu_rpy_deg")
        if base_rot_mat is not None:
            return np.asarray(base_rot_mat, dtype=np.float64).reshape(3, 3)
        if imu_rpy_deg is not None:
            base_rpy_rad = deg2rad(imu_rpy_deg)
        return rpy_to_rm(np.asarray(base_rpy_rad, dtype=np.float64))

    def assemble_proprio_57(
        self,
        *,
        base_omega_rad,
        command_raw,
        joint_pos_robot,
        joint_vel_robot,
        last_action_norm,
        base_rot_mat: np.ndarray | None = None,
        base_rpy_rad: np.ndarray | None = None,
        imu_rpy_deg: np.ndarray | None = None,
    ) -> np.ndarray:
        """Official 57-D proprio. command_raw must already be the unrewritten nav command."""
        rot = self.rotation_from_imu(
            base_rot_mat=base_rot_mat,
            base_rpy_rad=base_rpy_rad,
            imu_rpy_deg=imu_rpy_deg,
        )
        omega = _as_f32(base_omega_rad).reshape(3) * np.float32(self.omega_scale)
        projected_gravity = rot.T @ self.gravity_world.astype(np.float64)
        command = self.clip_command(command_raw)

        joint_pos = _as_f32(joint_pos_robot).reshape(16)[self.robot2policy_idx]
        joint_vel = _as_f32(joint_vel_robot).reshape(16)[self.robot2policy_idx]
        joint_pos[12:16] = 0.0
        joint_pos = joint_pos - self.default_pose_policy
        joint_vel = joint_vel * np.float32(self.dof_vel_scale)
        last_action = _as_f32(last_action_norm).reshape(16)

        obs = np.concatenate(
            [
                omega.astype(np.float32),
                projected_gravity.astype(np.float32),
                command,
                joint_pos.astype(np.float32),
                joint_vel.astype(np.float32),
                last_action,
            ]
        )
        if obs.shape != (57,):
            raise RuntimeError(f"proprio observation has shape {obs.shape}, expected (57,)")
        return obs

    def flatten_heightmap(self, height, validity) -> np.ndarray:
        height_arr = np.asarray(height, dtype=np.float32).reshape(self.policy_nx, self.policy_ny)
        validity_arr = np.asarray(validity, dtype=np.float32).reshape(self.policy_nx, self.policy_ny)
        return np.concatenate(
            [height_arr.reshape(-1, order="C"), validity_arr.reshape(-1, order="C")]
        )

    def unflatten_heightmap(self, flat) -> tuple[np.ndarray, np.ndarray]:
        values = _as_f32(flat)
        expected = 2 * self.policy_nx * self.policy_ny
        if values.size != expected:
            raise ValueError(f"heightmap flat size {values.size} != {expected}")
        mid = self.policy_nx * self.policy_ny
        height = values[:mid].reshape(self.policy_nx, self.policy_ny)
        validity = values[mid:].reshape(self.policy_nx, self.policy_ny)
        return height, validity

    def assemble_student_441(
        self,
        proprio_57,
        height,
        validity,
    ) -> np.ndarray:
        proprio = _as_f32(proprio_57).reshape(57)
        heightmap = self.flatten_heightmap(height, validity)
        obs = np.concatenate([proprio, heightmap])
        if obs.shape != (441,):
            raise RuntimeError(f"student observation has shape {obs.shape}, expected (441,)")
        return obs

    def slice_obs(self, obs, name: str) -> np.ndarray:
        start, end = self.obs_slices[name]
        return _as_f32(obs)[start:end]

    def reconstruct_robot_action(self, action_norm_policy) -> RobotAction:
        """Official decode: permute to robot_order, scale once, add default_pose_robot."""
        action_norm = _as_f32(action_norm_policy).reshape(16)
        scaled = action_norm[self.policy2robot_idx] * self.action_scale_robot
        physical = scaled + self.default_pose_robot
        goal_pos = np.zeros(16, dtype=np.float32)
        goal_vel = np.zeros(16, dtype=np.float32)
        for leg in range(4):
            base = 4 * leg
            goal_pos[base : base + 3] = physical[base : base + 3]
            goal_vel[base + 3] = physical[base + 3]
        return RobotAction(
            goal_joint_pos=goal_pos,
            goal_joint_vel=goal_vel,
            kp=self.kp_robot.copy(),
            kd=self.kd_robot.copy(),
        )


def command_from_mapping(values: Mapping[str, float]) -> np.ndarray:
    return np.array([values["vx"], values["vy"], values["wz"]], dtype=np.float32)
