#!/usr/bin/env python3
"""Headless MuJoCo rollout collector for a frozen S10 privileged teacher."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PERCEPTION = ROOT / "src" / "s10_terrain_perception"
sys.path.insert(0, str(PERCEPTION))
from d_priv_dataset import DPrivRecorder, validate_d_priv
from heightmap import build_heightmap, load_config as load_heightmap_config
from mujoco_lidar import MujocoLidarScanner, load_lidar_yaml
from mujoco_teacher import (
    JOINT_INIT_RAW, PrivilegedHeightScanner, assemble_official_57,
    assemble_teacher_1413, decode_action_raw, published_targets_to_raw,
    run_stand_up, state_from_mujoco)

DEFAULT_XML = ROOT / "models" / "mjcf" / "S10_track_lidar.xml"
DEFAULT_LIDAR = ROOT / "configs" / "lidar.yaml"
DEFAULT_HEIGHTMAP = ROOT / "configs" / "heightmap.yaml"
# 与 Isaac 训练侧 RSL-RL clip_actions=100 对齐（rsl_rl vecenv_wrapper.step()）；
# 不是 [-1,1] 的 a_norm 契约。见 doc/MUJOCO_ACTION_NORMALIZATION_FIX.md。
ACTION_RAW_LIMIT = 100.0 + 1e-3


class TeacherPolicy:
    def __init__(self, path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for real collection: pip install onnxruntime") from exc
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape[-1] != 1413:
            raise ValueError(f"teacher ONNX input must be obs[batch,1413], got {[(x.name, x.shape) for x in inputs]}")
        if len(outputs) != 1 or outputs[0].name != "actions" or outputs[0].shape[-1] != 16:
            raise ValueError(f"teacher ONNX output must be actions[batch,16], got {[(x.name, x.shape) for x in outputs]}")

    def __call__(self, obs):
        action = self.session.run(["actions"], {"obs": np.asarray(obs, np.float32)[None]})[0][0]
        if action.shape != (16,) or not np.isfinite(action).all():
            raise RuntimeError("teacher returned an invalid action")
        max_abs = float(np.max(np.abs(action)))
        if max_abs > ACTION_RAW_LIMIT:
            raise RuntimeError(
                f"teacher output exceeds the raw action safety limit: max_abs={max_abs:.6g} > "
                f"{ACTION_RAW_LIMIT:.6g}; re-export with --action-postprocess clip "
                "--action-limit 100 (matching Isaac clip_actions=100), or investigate why "
                "the actor produced an out-of-distribution value")
        return action.astype(np.float32)


class ZeroPolicy:
    def __call__(self, obs):
        if np.asarray(obs).shape != (1413,):
            raise ValueError("fake policy still requires a valid 1413-D observation")
        return np.zeros(16, np.float32)


class StudentRolloutPolicy:
    """441-D behavior policy used for DAgger-style state visitation."""

    def __init__(self, path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for student rollout: pip install onnxruntime") from exc
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape[-1] != 441:
            raise ValueError(
                "student ONNX input must be obs[batch,441], got "
                f"{[(x.name, x.shape) for x in inputs]}")
        if (len(outputs) != 1 or outputs[0].name != "actions"
                or outputs[0].shape[-1] != 16):
            raise ValueError(
                "student ONNX output must be actions[batch,16], got "
                f"{[(x.name, x.shape) for x in outputs]}")

    def __call__(self, obs):
        observation = np.asarray(obs, np.float32)
        if observation.shape != (441,) or not np.isfinite(observation).all():
            raise ValueError("student rollout requires one finite 441-D observation")
        action = self.session.run(
            ["actions"], {"obs": observation[None]})[0][0]
        if action.shape != (16,) or not np.isfinite(action).all():
            raise RuntimeError("student rollout policy returned an invalid action")
        max_abs = float(np.max(np.abs(action)))
        if max_abs > ACTION_RAW_LIMIT:
            raise RuntimeError(
                "student rollout output exceeds the raw action safety limit: "
                f"max_abs={max_abs:.6g} > {ACTION_RAW_LIMIT:.6g}")
        return action.astype(np.float32)


def file_sha256(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def waypoint_positions(model, data):
    rows = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("track_waypoint_"):
            token = name[len("track_waypoint_"):].split("_", 1)[0]
            if token.isdigit():
                rows.append((int(token), data.geom_xpos[geom_id].copy()))
    return sorted(rows)


def nearest_waypoint(base_pos, waypoints):
    if not waypoints:
        return -1
    return min(waypoints, key=lambda row: np.linalg.norm(row[1][:2] - base_pos[:2]))[0]


def wrap_angle(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def waypoint_bend_angle(waypoints, index):
    """Return the XY bend at one official waypoint in radians."""
    if index <= 0 or index >= len(waypoints) - 1:
        return 0.0
    before = waypoints[index - 1][1][:2]
    current = waypoints[index][1][:2]
    after = waypoints[index + 1][1][:2]
    incoming, outgoing = current - before, after - current
    denominator = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
    if denominator < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(
        np.dot(incoming, outgoing) / denominator, -1.0, 1.0)))


def initialize(model, data, start):
    model.opt.timestep = .001
    x, y, z, yaw = map(float, start)
    data.qpos[:3] = np.asarray([x, y, z])
    data.qpos[3:7] = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    data.qpos[7:23] = JOINT_INIT_RAW
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-onnx", type=Path)
    parser.add_argument("--fake-policy", action="store_true", help="Protocol smoke only; labels are unusable")
    parser.add_argument(
        "--rollout-policy", type=Path,
        help=("Optional 441->16 student ONNX that drives MuJoCo while the "
              "privileged teacher still supplies every saved action label"))
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--lidar-config", type=Path, default=DEFAULT_LIDAR)
    parser.add_argument("--heightmap-config", type=Path, default=DEFAULT_HEIGHTMAP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"), default=[.5, 0, 0])
    parser.add_argument("--command-schedule", type=Path, help="CSV rows: start_sample,vx,vy,wz")
    parser.add_argument("--autonav-target", nargs=2, type=float, metavar=("X", "Y"),
                        help="Closed-loop target; mutually exclusive with --command-schedule")
    parser.add_argument(
        "--waypoint-range", nargs=2, type=int, metavar=("FIRST", "LAST"),
        help=("Follow an inclusive range of official waypoints with the fixed "
              "0.20 m acceptance radius; mutually exclusive with fixed schedules"))
    parser.add_argument("--autonav-vx", type=float, default=.5)
    parser.add_argument("--start", nargs=4, type=float,
                        metavar=("X", "Y", "Z", "YAW"), default=[0, -2.5, .2, 0])
    parser.add_argument(
        "--start-waypoint", type=int,
        help="Spawn on this official waypoint, facing the following waypoint")
    parser.add_argument("--terrain-id", default="official_track")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--collection-config-sha256")
    parser.add_argument("--collection-scenario")
    parser.add_argument("--collection-phase")
    parser.add_argument("--stop-base-z", type=float, default=.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.fake_policy == (args.teacher_onnx is not None):
        parser.error("choose exactly one of --teacher-onnx or --fake-policy")
    if args.fake_policy and args.rollout_policy is not None:
        parser.error("--rollout-policy requires a real --teacher-onnx labeler")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    command_modes = sum(x is not None for x in (
        args.command_schedule, args.autonav_target, args.waypoint_range))
    if command_modes > 1:
        parser.error(
            "--command-schedule, --autonav-target and --waypoint-range "
            "are mutually exclusive")

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    if args.start_waypoint is not None:
        mujoco.mj_forward(model, data)
        spawn_waypoints = waypoint_positions(model, data)
        if not 0 <= args.start_waypoint < len(spawn_waypoints):
            parser.error(
                f"--start-waypoint must be in [0, {len(spawn_waypoints) - 1}]")
        current = spawn_waypoints[args.start_waypoint][1]
        following = spawn_waypoints[
            min(args.start_waypoint + 1, len(spawn_waypoints) - 1)][1]
        yaw = float(np.arctan2(
            following[1] - current[1], following[0] - current[0]))
        args.start = [float(current[0]), float(current[1]),
                      float(current[2]), yaw]
    initialize(model, data, args.start)
    if model.nu != 16:
        raise RuntimeError(f"S10 collector requires 16 actuators, MJCF has {model.nu}")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        raise RuntimeError("MJCF has no base_link")

    # 复刻官方 StandUpState：坐姿起立到站姿，起立期不喂教师、不写数据。
    # 否则首拍坐姿观测喂站姿教师 -> OOD 饱和 -> 关节目标突变 -> 塌陷。
    stand_up_z = run_stand_up(model, data, base_id, log=True)
    if stand_up_z < 0.30:
        print(f"warning: stand-up under height base_z={stand_up_z:.3f}, "
              f"teacher may start OOD; check MJCF/actuators", flush=True)

    rng = np.random.default_rng(args.seed)
    lidar_cfg = load_lidar_yaml(args.lidar_config)
    lidar = MujocoLidarScanner(model, lidar_cfg, rng=rng)
    heightmap_cfg = load_heightmap_config(args.heightmap_config)
    privileged = PrivilegedHeightScanner(model, body_exclude=base_id)
    teacher_policy = ZeroPolicy() if args.fake_policy else TeacherPolicy(args.teacher_onnx)
    rollout_policy = (
        StudentRolloutPolicy(args.rollout_policy)
        if args.rollout_policy is not None else None)
    teacher_model = None if args.fake_policy else args.teacher_onnx
    recorder = DPrivRecorder(
        teacher_model=teacher_model,
        teacher_source="fake_mujoco_smoke" if args.fake_policy else "privileged_mujoco",
        metadata={
            "collector_commit": git_head(), "xml": str(args.xml.resolve()),
            "xml_sha256": file_sha256(args.xml), "sim_dt_s": .001,
            "policy_dt_s": .02, "student_lidar_hz": lidar_cfg["rate_hz"],
            "teacher_grid": [41, 33], "teacher_height_formula": "base_z-hit_z-0.5",
            "command": list(map(float, args.command)),
            "command_schedule": str(args.command_schedule.resolve()) if args.command_schedule else None,
            "autonav_target": args.autonav_target,
            "waypoint_range": args.waypoint_range,
            "waypoint_reach_radius_m": .20 if args.waypoint_range else None,
            "autonav_controller": "official_route_v1" if args.waypoint_range else (
                "single_target_v1" if args.autonav_target else None),
            "autonav_vx": args.autonav_vx if (
                args.autonav_target or args.waypoint_range) else None,
            "start_xyzyaw": list(map(float, args.start)),
            "start_waypoint": args.start_waypoint,
            "terrain_id": args.terrain_id, "episode_id": args.episode_id,
            "collection_config_sha256": args.collection_config_sha256,
            "collection_scenario": args.collection_scenario,
            "collection_phase": args.collection_phase,
            "seed": args.seed,
            "labels_usable": not args.fake_policy,
            "behavior_policy": "student_onnx" if rollout_policy else (
                "fake_policy" if args.fake_policy else "privileged_teacher"),
            "behavior_model": (
                str(args.rollout_policy.resolve()) if args.rollout_policy else None),
            "behavior_sha256": (
                file_sha256(args.rollout_policy) if args.rollout_policy else None),
            "dagger_state_visitation": rollout_policy is not None,
        })
    waypoints = waypoint_positions(model, data)
    route_index = None
    route_last = None
    if args.waypoint_range:
        route_index, route_last = args.waypoint_range
        waypoint_ids = [row[0] for row in waypoints]
        if waypoint_ids != list(range(len(waypoints))):
            raise RuntimeError(
                f"official waypoint ids must be contiguous from zero, got {waypoint_ids}")
        if not 0 <= route_index <= route_last < len(waypoints):
            parser.error(
                f"--waypoint-range must satisfy 0 <= FIRST <= LAST < {len(waypoints)}")
    command = np.asarray(args.command, np.float32)
    schedule = None
    if args.command_schedule:
        schedule = np.loadtxt(args.command_schedule, delimiter=",", ndmin=2)
        if schedule.shape[1] != 4 or np.any(np.diff(schedule[:, 0]) < 0):
            raise ValueError("command schedule must be sorted CSV rows: start_sample,vx,vy,wz")
    last_action = np.zeros(16, np.float32)
    goal_pos, goal_vel = decode_action_raw(last_action)
    raw_pos, raw_vel = published_targets_to_raw(goal_pos, goal_vel)
    kp = np.asarray([80, 80, 80, 0] * 4, np.float32)
    kd = np.asarray([2, 2, 2, .6] * 4, np.float32)

    state = state_from_mujoco(model, data, base_id)
    lidar_scan = lidar.scan(data, state.base_pos_w, state.base_rotation_w, apply_noise=True)
    student_map = build_heightmap(
        lidar_scan.points_w, state.base_pos_w, state.base_rotation_w, heightmap_cfg)
    physics_step = 0
    autonav_wz = 0.0
    for sample in range(args.samples):
        if schedule is not None:
            active = schedule[schedule[:, 0] <= sample]
            if len(active):
                command = active[-1, 1:4].astype(np.float32)
        state = state_from_mujoco(model, data, base_id)
        nav_target = None
        nav_index = None
        if args.waypoint_range:
            while route_index <= route_last:
                distance = np.linalg.norm(
                    waypoints[route_index][1][:2] - state.base_pos_w[:2])
                if distance >= .20:
                    break
                print(f"reached official waypoint {route_index:02d}", flush=True)
                route_index += 1
                autonav_wz = 0.0
            if route_index <= route_last:
                nav_index = route_index
                nav_target = waypoints[route_index][1][:2]
        elif args.autonav_target:
            nav_target = np.asarray(args.autonav_target, np.float64)
        if nav_target is not None:
            target = np.asarray(nav_target, np.float64)
            delta = target - state.base_pos_w[:2]
            target_yaw = float(np.arctan2(delta[1], delta[0]))
            yaw = float(np.arctan2(
                state.base_rotation_w[1, 0], state.base_rotation_w[0, 0]))
            error = wrap_angle(target_yaw - yaw)
            desired_wz = float(np.clip(2.0 * error, -.6, .6))
            autonav_wz += float(np.clip(desired_wz - autonav_wz, -.01, .01))
            alignment = float(np.clip((.20 - abs(error)) / .15, 0.0, 1.0))
            base_vx = args.autonav_vx
            if (nav_index is not None and np.linalg.norm(delta) < 1.2
                    and waypoint_bend_angle(waypoints, nav_index) > .785):
                base_vx = min(base_vx, .25)
            if abs(error) < .20:
                vx = base_vx * alignment
            else:
                vx = min(base_vx, .12) if abs(error) < 1.4 else 0.0
            command = np.asarray([vx, 0.0, autonav_wz], np.float32)
        elif args.waypoint_range:
            command = np.zeros(3, np.float32)
        command = np.clip(command, [-1, -.6, -1], [1, .6, 1]).astype(np.float32)
        # fake-policy 模式只验证协议格式，零动作无法维持站姿属预期，跳过高度检查。
        if not args.fake_policy and sample > 0 and state.base_pos_w[2] < args.stop_base_z:
            print(f"stopping early: base_z={state.base_pos_w[2]:.3f} < {args.stop_base_z:.3f}")
            break
        proprio = assemble_official_57(state, command, last_action)
        privileged_height, hit, _ = privileged.scan(
            data, state.base_pos_w, state.base_rotation_w)
        teacher_obs = assemble_teacher_1413(state, proprio, privileged_height)
        student_obs = np.concatenate([proprio, student_map.reshape(-1)]).astype(np.float32)
        if student_obs.shape != (441,):
            raise RuntimeError(f"student observation is {student_obs.shape}, expected (441,)")
        # Label and behavior are deliberately separate.  The privileged
        # teacher labels states visited by the student; last_action and the
        # actuators must follow the behavior action to avoid future leakage.
        teacher_action = teacher_policy(teacher_obs)
        behavior_action = (
            rollout_policy(student_obs) if rollout_policy else teacher_action)
        pose = np.concatenate([state.base_pos_w, data.xquat[base_id]]).astype(np.float32)
        recorder.append(
            student_obs=student_obs, teacher_action_raw=teacher_action, command_raw=command,
            waypoint_id=nearest_waypoint(state.base_pos_w, waypoints), terrain_id=args.terrain_id,
            episode_id=args.episode_id, step_id=sample, base_pose_wxyz=pose,
            privileged_hit_fraction=float(hit.mean()))
        last_action = behavior_action
        goal_pos, goal_vel = decode_action_raw(behavior_action)
        raw_pos, raw_vel = published_targets_to_raw(goal_pos, goal_vel)
        for _ in range(20):
            q, dq = data.qpos[7:23], data.qvel[6:22]
            data.ctrl[:] = kp * (raw_pos - q) + kd * (raw_vel - dq)
            mujoco.mj_step(model, data)
            physics_step += 1
            if physics_step % 50 == 0:
                lidar_state = state_from_mujoco(model, data, base_id)
                lidar_scan = lidar.scan(
                    data, lidar_state.base_pos_w, lidar_state.base_rotation_w, apply_noise=True)
                if len(lidar_scan.points_w):
                    student_map = build_heightmap(
                        lidar_scan.points_w, lidar_state.base_pos_w,
                        lidar_state.base_rotation_w, heightmap_cfg)
        if args.log_every and (sample + 1) % args.log_every == 0:
            print(f"samples={sample + 1}/{args.samples} sim_time={data.time:.2f}s "
                  f"base_z={state.base_pos_w[2]:.3f} privileged_hits={hit.mean():.3f}", flush=True)

    output = recorder.write(args.output)
    count = validate_d_priv(output)
    summary = {
        "output": str(output.resolve()), "samples": count,
        "teacher": "fake_policy" if args.fake_policy else str(args.teacher_onnx.resolve()),
        "teacher_sha256": (
            "fake_policy" if args.fake_policy else file_sha256(args.teacher_onnx)),
        "teacher_scan_every": 1,
        "waypoint_reach_radius_m": .20 if args.waypoint_range else None,
        "autonav_controller": (
            "official_route_v1" if args.waypoint_range else None),
        "rollout_policy": (
            str(args.rollout_policy.resolve()) if args.rollout_policy else None),
        "labels_usable": not args.fake_policy,
        "waypoint_range": args.waypoint_range,
        "route_next_waypoint": route_index,
        "route_complete": bool(
            args.waypoint_range and route_index is not None
            and route_index > route_last),
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
