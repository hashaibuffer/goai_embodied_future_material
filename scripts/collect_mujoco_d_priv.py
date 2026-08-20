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
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--lidar-config", type=Path, default=DEFAULT_LIDAR)
    parser.add_argument("--heightmap-config", type=Path, default=DEFAULT_HEIGHTMAP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"), default=[.5, 0, 0])
    parser.add_argument("--command-schedule", type=Path, help="CSV rows: start_sample,vx,vy,wz")
    parser.add_argument("--start", nargs=4, type=float, metavar=("X", "Y", "Z", "YAW"), default=[0, -2.5, .2, 0])
    parser.add_argument("--terrain-id", default="official_track")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--stop-base-z", type=float, default=.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.fake_policy == (args.teacher_onnx is not None):
        parser.error("choose exactly one of --teacher-onnx or --fake-policy")
    if args.samples <= 0:
        parser.error("--samples must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
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
    policy = ZeroPolicy() if args.fake_policy else TeacherPolicy(args.teacher_onnx)
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
            "start_xyzyaw": list(map(float, args.start)),
            "terrain_id": args.terrain_id, "episode_id": args.episode_id,
            "seed": args.seed,
            "labels_usable": not args.fake_policy,
        })
    waypoints = waypoint_positions(model, data)
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
    for sample in range(args.samples):
        if schedule is not None:
            active = schedule[schedule[:, 0] <= sample]
            if len(active):
                command = active[-1, 1:4].astype(np.float32)
        command = np.clip(command, [-1, -.6, -1], [1, .6, 1]).astype(np.float32)
        state = state_from_mujoco(model, data, base_id)
        # fake-policy 模式只验证协议格式，零动作无法维持站姿属预期，跳过高度检查。
        if not args.fake_policy and sample > 0 and state.base_pos_w[2] < args.stop_base_z:
            print(f"stopping early: base_z={state.base_pos_w[2]:.3f} < {args.stop_base_z:.3f}")
            break
        proprio = assemble_official_57(state, command, last_action)
        privileged_height, hit, _ = privileged.scan(
            data, state.base_pos_w, state.base_rotation_w)
        teacher_obs = assemble_teacher_1413(state, proprio, privileged_height)
        action = policy(teacher_obs)
        student_obs = np.concatenate([proprio, student_map.reshape(-1)]).astype(np.float32)
        if student_obs.shape != (441,):
            raise RuntimeError(f"student observation is {student_obs.shape}, expected (441,)")
        pose = np.concatenate([state.base_pos_w, data.xquat[base_id]]).astype(np.float32)
        recorder.append(
            student_obs=student_obs, teacher_action_raw=action, command_raw=command,
            waypoint_id=nearest_waypoint(state.base_pos_w, waypoints), terrain_id=args.terrain_id,
            episode_id=args.episode_id, step_id=sample, base_pose_wxyz=pose,
            privileged_hit_fraction=float(hit.mean()))
        last_action = action
        goal_pos, goal_vel = decode_action_raw(action)
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
        "labels_usable": not args.fake_policy,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
