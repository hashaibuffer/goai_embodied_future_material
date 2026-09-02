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


def assemble_asymmetric_teacher_inputs(state, proprio_57, privileged_height):
    """Return the deployed CNN-GRU Actor inputs in their trained order.

    The asymmetric teacher is not the historical flat 1413-D MLP protocol.
    It consumes command[3], proprio[57] and height_map[1,41,33] separately;
    its proprio begins with base linear velocity and excludes the command.
    """
    official = np.asarray(proprio_57, np.float32).reshape(57)
    command = official[6:9].copy()
    proprio = np.concatenate(
        [state.base_lin_vel_b, official[:6], official[9:]]
    ).astype(np.float32)
    # MuJoCo's ray grid is emitted by ``meshgrid(indexing="xy")``: x changes
    # fastest, so its flat storage is [y=33, x=41].  The trained Actor CNN,
    # however, consumes [channel, x=41, y=33].  A bare reshape silently
    # scrambles those spatial axes; transpose the semantic grid explicitly.
    height_yx = np.asarray(privileged_height, np.float32).reshape(33, 41)
    height_map = np.ascontiguousarray(height_yx.T)[None, ...]
    if command.shape != (3,) or proprio.shape != (57,):
        raise RuntimeError(
            f"invalid asymmetric input shapes: command={command.shape}, proprio={proprio.shape}"
        )
    if not (
        np.isfinite(command).all()
        and np.isfinite(proprio).all()
        and np.isfinite(height_map).all()
    ):
        raise RuntimeError("asymmetric teacher inputs contain non-finite values")
    return command, proprio, height_map


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


def sanitize_privileged_height_grid(
        height_grid, valid_grid, fallback_height, *,
        deep_drop_threshold=.52, deep_drop_fill=.17):
    """Apply the CommandFirst finite-cliff protocol to one 33x41 grid."""
    height = np.asarray(height_grid, np.float32).reshape(33, 41)
    valid = np.asarray(valid_grid, bool).reshape(33, 41)
    if not 0.0 < deep_drop_fill < deep_drop_threshold:
        raise ValueError("deep_drop_fill must be positive and below the threshold")
    result = height.copy()
    result[valid & (result > deep_drop_threshold)] = deep_drop_fill
    for row, column in np.argwhere(~valid):
        row_candidates = np.flatnonzero(valid[row])
        column_candidates = np.flatnonzero(valid[:, column])
        best_distance = np.inf
        best_value = float(fallback_height)
        if row_candidates.size:
            nearest_column = row_candidates[np.argmin(np.abs(row_candidates - column))]
            best_distance = abs(int(nearest_column) - int(column))
            best_value = float(result[row, nearest_column])
        if column_candidates.size:
            nearest_row = column_candidates[np.argmin(np.abs(column_candidates - row))]
            distance = abs(int(nearest_row) - int(row))
            if distance < best_distance:
                best_value = float(result[nearest_row, column])
        result[row, column] = best_value
    return result


@dataclass(frozen=True)
class PrivilegedHeightScan:
    height: np.ndarray
    hit: np.ndarray
    geom_ids: np.ndarray
    hit_z_w: np.ndarray
    base_ground_z_w: float


class PrivilegedHeightScanner:
    """Exact Isaac GridPattern equivalent: 41x33 vertical rays, yaw aligned.

    射线发射高度 ``RAY_ORIGIN_OFFSET_Z``：与 Isaac Lab 端 2026-08-22 的
    ``fix(s10): lower Isaac height scanner origin`` (commit 39cd082) 保持
    同步，从 base 上方 20m 降到 1.2m。原因：赛道上存在门框/屋顶等高处
    悬空结构，20m 高空垂直下射的射线会先命中这些悬空结构而不是真正的
    脚下地形，产生错误的"高障碍"读数，导致教师误判需要避障/攀爬。
    起点下移到贴近机身的 1.2m 后能排除大多数高处横梁，但较低的悬空楼板
    仍可能先于地面被命中。该扫描器只负责保持冻结Actor输入合同；Router
    使用独立的分层前向净空射线区分实体台阶和悬空障碍。

    高度值与 Isaac Lab 的 base-frame 语义一致：``height = base_z -
    ray_hit.z - offset``。MuJoCo 的 distance 从 ``base_z + 射线起点偏移``
    量起，因此需要减掉 ``RAY_ORIGIN_OFFSET_Z + offset``。抬高地形为负，
    下降为正。CommandFirst 协议保留 ``<=+0.52m`` 的有效下降；更深命中
    压缩成 ``+0.17m``。无命中优先沿扫描行/列复制最近有效高度，完全没有
    同轴支撑时才退回 3x3 足下扫描中值（不足 5/9 命中则为 ``+0.17m``）。
    """
    X = np.linspace(-.8, 3.2, 41, dtype=np.float64)
    Y = np.linspace(-1.6, 1.6, 33, dtype=np.float64)
    RAY_ORIGIN_OFFSET_Z = 1.2
    HEIGHT_SCAN_OFFSET = 0.5
    DEEP_DROP_THRESHOLD = 0.52
    DEEP_DROP_FILL = 0.17

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

    def scan_geometry(self, data, base_pos_w, base_rotation_w):
        pos = np.asarray(base_pos_w, np.float64).reshape(3)
        rotation = np.asarray(base_rotation_w, np.float64).reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        xy_w = self.local_xy @ np.asarray([[c, s], [-s, c]]) + pos[:2]
        starts = np.column_stack(
            [xy_w, np.full(1353, pos[2] + self.RAY_ORIGIN_OFFSET_Z)])
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
        height = np.zeros(1353, dtype=np.float32)
        # hit_z = base_z + RAY_ORIGIN_OFFSET_Z - distance;
        # Isaac term uses attached base z, not the elevated ray start:
        # base_z - hit_z - HEIGHT_SCAN_OFFSET
        # = distance - RAY_ORIGIN_OFFSET_Z - HEIGHT_SCAN_OFFSET.
        const = self.RAY_ORIGIN_OFFSET_Z + self.HEIGHT_SCAN_OFFSET
        height[hit] = np.maximum(distances[hit] - const, -1.0)

        # Match Isaac's 0.10m square base scanner (0.05m resolution): its
        # robust median is the final fallback only when at least 5/9 rays hit.
        base_axis = np.asarray([-.05, 0.0, .05], np.float64)
        bx, by = np.meshgrid(base_axis, base_axis, indexing="xy")
        base_xy = np.column_stack([bx.reshape(-1), by.reshape(-1)])
        base_xy_w = base_xy @ np.asarray([[c, s], [-s, c]]) + pos[:2]
        base_starts = np.column_stack([
            base_xy_w,
            np.full(base_xy_w.shape[0], pos[2] + self.RAY_ORIGIN_OFFSET_Z),
        ])
        base_height = []
        base_hit_z = []
        for start in base_starts:
            geom_out[0] = -1
            distance = mujoco.mj_ray(
                self.model, data, start, self.direction, self.geomgroup,
                True, self.body_exclude, geom_out)
            if geom_out[0] >= 0 and distance >= 0.0:
                base_height.append(max(distance - const, -1.0))
                base_hit_z.append(float(start[2] - distance))
        fallback = (
            float(np.median(base_height))
            if len(base_height) >= 5 else self.DEEP_DROP_FILL
        )
        sanitized = sanitize_privileged_height_grid(
            height.reshape(33, 41),
            hit.reshape(33, 41),
            fallback,
            deep_drop_threshold=self.DEEP_DROP_THRESHOLD,
            deep_drop_fill=self.DEEP_DROP_FILL,
        )
        hit_z_w = np.full(1353, np.nan, dtype=np.float64)
        hit_z_w[hit] = starts[hit, 2] - distances[hit]
        ground_z = (
            float(np.median(base_hit_z))
            if len(base_hit_z) >= 5 else float("nan")
        )
        return PrivilegedHeightScan(
            height=sanitized.reshape(-1),
            hit=hit,
            geom_ids=geom_ids,
            hit_z_w=hit_z_w,
            base_ground_z_w=ground_z,
        )

    def scan(self, data, base_pos_w, base_rotation_w):
        geometry = self.scan_geometry(data, base_pos_w, base_rotation_w)
        return geometry.height, geometry.hit, geometry.geom_ids
