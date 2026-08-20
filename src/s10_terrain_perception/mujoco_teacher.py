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
    """从 MuJoCo 状态提取策略输入。

    注意：MJCF 的关节零点定义与 Isaac Lab/URDF 一致，data.qpos/qvel 直接
    就是 policy(published) 空间的关节角/角速度，不需要 POS_OFFSET_RAD/
    JOINT_DIR 变换 —— 那套变换只属于真实机器人电机编码器相对 URDF 零点的
    标定偏移（见 s10_interface.hpp），与仿真 qpos 无关。
    历史 bug：此前误把 qpos 当作电机编码器 raw 值做了
    ((raw_pos - POS_OFFSET_RAD) * JOINT_DIR) 变换，导致机器人站姿下
    joint_pos 观测偏离 DEFAULT_POLICY 最大 ~2.7rad，策略严重 OOD 后
    输出饱和动作、约 11 步内倒地。修复为直接使用 qpos/qvel。
    """
    spatial = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, spatial, 1)
    rotation = data.xmat[base_body_id].reshape(3, 3).copy()
    raw_pos = data.qpos[7:23]
    raw_vel = data.qvel[6:22]
    return S10PolicyState(
        data.xpos[base_body_id].copy(), rotation,
        spatial[3:6].astype(np.float32), spatial[:3].astype(np.float32),
        np.asarray(raw_pos, np.float32).copy(),
        np.asarray(raw_vel, np.float32).copy())


def assemble_official_57(state: S10PolicyState, command_raw, last_action_raw):
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
        np.asarray(last_action_raw, np.float32).reshape(16),
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


def decode_action_raw(action_raw):
    """将策略原始动作解码为 MuJoCo 关节目标位置和速度。

    注意：action_raw 是策略网络的原始输出（Isaac 训练时仅 clip 到 ±100，
    不是归一化到 [-1,1] 的动作）。此函数直接套用 action_scale_robot +
    default_pose_robot，与 Isaac Lab JointPositionAction/JointVelocityAction、
    真机 TerrainPolicyMath::DecodeAction 的公式完全一致。历史误区：曾误认为
    此处的 action 必须 ∈[-1,1]（因此字段名叫 action_norm），实际上 Isaac
    训练时 clip_actions=100，真机 runner 也无 [-1,1] 裁剪，强行把策略原始
    输出削平到 ±1 会丢失左右腿动作幅度差异，导致学生无法学会上台阶等复杂行为。
    """
    action = np.asarray(action_raw, np.float32).reshape(16)
    physical = action[POLICY_TO_ROBOT] * ACTION_SCALE_ROBOT + DEFAULT_ROBOT
    goal_pos = np.zeros(16, np.float32)
    goal_vel = np.zeros(16, np.float32)
    for leg in range(4):
        start = 4 * leg
        goal_pos[start:start + 3] = physical[start:start + 3]
        goal_vel[start + 3] = physical[start + 3]
    return goal_pos, goal_vel


def published_targets_to_raw(goal_pos, goal_vel):
    """将策略解码出的目标关节角/角速度转换为 MuJoCo PD 控制目标（qpos/qvel 空间）。

    注意：MJCF 的关节零点与 policy(published) 空间一致，此处不需要
    POS_OFFSET_RAD/JOINT_DIR 变换（那是真实电机编码器标定，仅用于
    真机硬件接口，见 s10_interface.hpp）。历史 bug：此前对 goal_pos/
    goal_vel 施加了 *JOINT_DIR+POS_OFFSET_RAD 的二次错误变换，导致目标
    超出 MJCF 关节硬限位（如 hipy 跳到 ±2.8rad > ±2.53rad），是采集时
    机器人约 11 步倒地的根因之一。修复为直接透传。
    """
    return (
        np.asarray(goal_pos, np.float32).copy(),
        np.asarray(goal_vel, np.float32).copy(),
    )


# ---------------------------------------------------------------------------
# StandUp 起立：官方状态机 StandUpState 的 MuJoCo 等价实现。
# 比赛运行时 rl_deploy 会先走 StandUpState（纯运动学起立，无网络），
# 起立完成后再进 RLControlMode 跑策略。采集器必须复刻同一起立时序，
# 否则首拍把坐姿 JOINT_INIT_RAW 直接喂给站姿训练的教师 -> OOD -> 塌陷。
# ---------------------------------------------------------------------------
STAND_UP_DURATION_S = 3.0
# 官方起立刚度：腿 swing_leg_kp/kd = 120/2；轮是速度控制 kp=0、kd=0.6 阻尼。
STAND_UP_KP = np.asarray([120.0, 120.0, 120.0, 0.0] * 4, np.float32)
STAND_UP_KD = np.asarray([2.0, 2.0, 2.0, 0.6] * 4, np.float32)


def cubic_spline_pos(x0, v0, xf, vf, t, T):
    """复刻官方 GetCubicSplinePos：三次 Hermite 插值位置。"""
    if t >= T:
        return float(xf)
    a = (vf * T - 2.0 * xf + v0 * T + 2.0 * x0) / (T ** 3)
    b = (3.0 * xf - vf * T - 2.0 * v0 * T - 3.0 * x0) / (T ** 2)
    return a * t ** 3 + b * t ** 2 + v0 * t + x0


def cubic_spline_vel(x0, v0, xf, vf, t, T):
    """复刻官方 GetCubicSplineVel：三次 Hermite 插值速度。"""
    if t >= T:
        return 0.0
    a = (vf * T - 2.0 * xf + v0 * T + 2.0 * x0) / (T ** 3)
    b = (3.0 * xf - vf * T - 2.0 * v0 * T - 3.0 * x0) / (T ** 2)
    return 3.0 * a * t ** 2 + 2.0 * b * t + v0


def stand_up_target_raw():
    """起立终点 = 教师训练默认站姿，直接就是 raw/MJCF 关节角。

    注意：DEFAULT_ROBOT 与 JOINT_INIT_RAW 同属 raw 空间（两者都是
    IK 直接算出的物理关节角，可用 GetHipYPosByHeight/GetKneePosByHeight
    核实：h=0.48 -> hipy=-0.284, knee=0.568，与 DEFAULT_ROBOT 的
    [-0.3, 0.6] 吻合）。decode_action_raw/published_targets_to_raw 是
    "策略输出 action_raw -> 目标关节角" 的运行时解码管线，只能作用于
    策略动作，不能套在静态的 DEFAULT_ROBOT 常量上，否则会被
    JOINT_DIR/POS_OFFSET_RAD 二次错误变换，导致目标超出 MJCF 关节限位
    （例如 hipy 会跳到 ±2.8rad，远超 ±2.53rad 硬限位）。
    """
    return DEFAULT_ROBOT.astype(np.float64).copy()


def run_stand_up(model, data, base_id, duration_s=STAND_UP_DURATION_S, log=False):
    """从坐姿 JOINT_INIT_RAW 起立到 DEFAULT_ROBOT，返回起立后的 base_z。

    起立期只做位置+速度前馈 PD，不喂任何策略、不写任何数据。
    起立刚度沿用官方 StandUpState（腿 120/2，轮 0/0.6 阻尼）。
    """
    dt = float(model.opt.timestep)
    init_raw = JOINT_INIT_RAW.astype(np.float64)
    target_raw = stand_up_target_raw().astype(np.float64)
    kp = STAND_UP_KP
    kd = STAND_UP_KD
    steps = int(round(duration_s / dt))
    for i in range(steps):
        t = (i + 1) * dt
        planned_pos = np.asarray([
            cubic_spline_pos(init_raw[j], 0.0, target_raw[j], 0.0, t, duration_s)
            for j in range(16)], np.float64)
        planned_vel = np.asarray([
            cubic_spline_vel(init_raw[j], 0.0, target_raw[j], 0.0, t, duration_s)
            for j in range(16)], np.float64)
        q = data.qpos[7:23]
        dq = data.qvel[6:22]
        data.ctrl[:] = kp * (planned_pos - q) + kd * (planned_vel - dq)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    base_z = float(data.xpos[base_id][2])
    if log:
        print(f"[StandUp] complete: base_z={base_z:.3f} (standing posture DEFAULT_ROBOT)")
    return base_z


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
