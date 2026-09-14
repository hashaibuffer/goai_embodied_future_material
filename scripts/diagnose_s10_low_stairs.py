#!/usr/bin/env python3
"""Headless model99 stair A/B. Synthetic observations are diagnostic only.

Compare official physics/native observations, open stairs/native observations,
and official physics/synthetic stairs observations. The last case is an oracle
ablation, NOT a deployable perception fix or an official-track success result.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/s10_terrain_perception"),
                str(ROOT / "src/s10_terrain_policy")]
from mujoco_teacher import (DEFAULT_ROBOT, PrivilegedHeightScanner,
                            assemble_asymmetric_teacher_inputs,
                            assemble_official_57, decode_action_raw, state_from_mujoco)
from teacher_skill_router import RecurrentOnnxActor, TeacherSkillRuntime
from teacher_skill_router import clamp_height_window
from locomotion_height import select_support_surface, SurfaceSelection, scan_xy_world, height_debug_record, wheel_tread_heights
from mujoco_climb_detector import S10ClimbDetector
from play_mujoco_teacher import wheel_contact_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("official", "open", "flat"), default="official")
    parser.add_argument("--observation", choices=("native", "synthetic-stairs"), default="native")
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-surface", action="store_true")
    parser.add_argument("--half-width", type=float, default=0.)
    parser.add_argument("--x-range", nargs=2, type=float)
    parser.add_argument("--start-x", type=float, default=33.6)
    parser.add_argument("--start-y", type=float, default=15.18)
    parser.add_argument("--yaw", type=float, default=np.pi/2)
    parser.add_argument("--router", action="store_true", help="use the full deployed Router, not direct Low")
    parser.add_argument("--pause-at-y", type=float, help="pause once while front/rear treads differ beyond this world Y")
    parser.add_argument("--pause-seconds", type=float, default=1.)
    parser.add_argument("--forget-router-on-resume", action="store_true",
                        help="diagnostic: reset Router to NORMAL once at resume, eliminating pause history")
    parser.add_argument("--command-vx", type=float, default=.4)
    args = parser.parse_args()
    clamp_height_window(np.zeros((1, 41, 33), np.float32), args.half_width, args.x_range)
    xml_path = ROOT / "models/mjcf/S10_track_lidar.xml"
    if args.scene in ("open", "flat"):
        xml = xml_path.read_text().replace("../../src/", str(ROOT / "src") + "/")
        stairs = '<worldbody><geom type="plane" size="50 50 .1" pos="0 0 1.666"/>'
        for i in range(14 if args.scene == "open" else 0):
            half_height = (i + 1) * .075 / 2
            stairs += (f'<geom type="box" size="4 .21 {half_height}" '
                       f'pos="33.6 {15.5 + i * .42 + .21} {1.666 + half_height}"/>')
        if args.scene == "open":
            stairs += '<geom type="box" size="4 4 .525" pos="33.6 25.38 2.191"/>'
        stairs += '</worldbody>'
        xml = re.sub(r'<include file="[^"]*scene.xml"/>', stairs, xml)
        model = mujoco.MjModel.from_xml_string(xml)
    else:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = .001
    data = mujoco.MjData(model)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    bundle = ROOT / "artifacts/s10_teacher_router/teacher_router_low_command_events_model99.json"
    payload = json.loads(bundle.read_text())
    entry = payload["skills"]["LOW_STEP_SEQUENCE"]
    actor_path = (bundle.parent / entry["path"]).resolve()
    import hashlib
    if hashlib.sha256(actor_path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Low ONNX hash mismatch")
    runtime = (TeacherSkillRuntime(bundle, .02, low_height_corridor_half_width_m=args.half_width,
                                  low_height_x_range=args.x_range, actor_threads=1) if args.router else None)
    actor = None if runtime else RecurrentOnnxActor(actor_path, intra_op_threads=1)
    scanner = PrivilegedHeightScanner(model, body_exclude=base)
    detector = S10ClimbDetector(model, base, payload["detector"])
    wheels = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name + "_wheel")
                       for name in ("fl", "fr", "hl", "hr")])
    data.qpos[:3] = [args.start_x, args.start_y, 2.09]
    data.qpos[3:7] = [np.cos(args.yaw/2), 0, 0, np.sin(args.yaw/2)]
    data.qpos[7:23] = DEFAULT_ROBOT
    mujoco.mj_forward(model, data)
    command = np.array([args.command_vx, 0, 0], np.float32)
    last = np.zeros(16, np.float32)
    kp, kd = np.array([80, 80, 80, 0] * 4), np.array([2, 2, 2, .6] * 4)
    rows = []
    top_reached = False
    pause_start = None
    started = time.perf_counter()
    for step in range(args.steps):
        state = state_from_mujoco(model, data, base)
        tread = wheel_tread_heights(scanner, data, data.xpos[wheels])
        if (args.pause_at_y is not None and pause_start is None and runtime is not None
                and runtime.router.mode.name == "LOW_STEP_SEQUENCE"
                and state.base_pos_w[1] >= args.pause_at_y
                and np.isfinite(tread).all() and min(tread[:2]) - max(tread[2:]) >= .04):
            pause_start = step
        command[0] = 0. if pause_start is not None and step < pause_start + round(args.pause_seconds / .02) else args.command_vx
        if (args.forget_router_on_resume and runtime is not None and pause_start is not None
                and step == pause_start + round(args.pause_seconds / .02)):
            runtime.reset()
        geometry = scanner.scan_geometry(data, state.base_pos_w, state.base_rotation_w)
        selection = (select_support_surface(scanner, data, state.base_pos_w, state.base_rotation_w, geometry,
                                           query_x_range=args.x_range, query_half_width=args.half_width)
                     if args.support_surface else SurfaceSelection(geometry, np.zeros(1353, bool),
                     np.zeros(1353, bool), scan_xy_world(scanner, state.base_pos_w, state.base_rotation_w), 0.))
        inputs = assemble_asymmetric_teacher_inputs(
            state, assemble_official_57(state, command, last), selection.scan.height)
        if args.observation == "synthetic-stairs":
            yaw = np.arctan2(state.base_rotation_w[1, 0], state.base_rotation_w[0, 0])
            x, y = np.meshgrid(scanner.X, scanner.Y, indexing="ij")
            world_y = state.base_pos_w[1] + np.sin(yaw) * x + np.cos(yaw) * y
            ground = 1.666 + np.clip(np.floor((world_y - 15.32) / .42) + 1, 0, 14) * .075
            inputs[2][:] = state.base_pos_w[2] - ground - .5
        if runtime:
            native = assemble_asymmetric_teacher_inputs(state, assemble_official_57(state, command, last), geometry.height)
            detection = detector.detect(selection.scan, data, state.base_pos_w, state.base_rotation_w, command)
            last = runtime.step(*native, detection, state, data.xpos[wheels, 2],
                                wheel_contact_mask(model, data, wheels), low_height_map=inputs[2],
                                wheel_support_z=tread)
            inputs = (inputs[0], inputs[1], runtime.last_actor_height_map)
            mode = runtime.router.mode.name
        else:
            inputs = (inputs[0], inputs[1], clamp_height_window(inputs[2], args.half_width, args.x_range))
            last = actor(*inputs)
            mode = "LOW_ONLY"
        if step % 25 == 0:
            row = {"time_s": step * .02, "position": state.base_pos_w.tolist(),
                   "velocity_body": state.base_lin_vel_b.tolist(), "command": command.tolist(),
                   "mode": mode, "height_map": inputs[2].tolist(),
                   "last_entry_reason": runtime.router.last_entry_reason if runtime else None,
                   "wheel_z": data.xpos[wheels, 2].tolist(),
                   "wheel_tread_z": [float(z) if np.isfinite(z) else None for z in tread],
                   "wheel_contact": wheel_contact_mask(model, data, wheels).tolist()}
            rows.append(row)
            row["height_debug"] = height_debug_record(geometry, selection, inputs[2], state.base_pos_w)
            print(f'{row["time_s"]:.2f}s {mode} xyz={np.round(state.base_pos_w, 3)}', flush=True)
        if state.base_pos_w[1] >= 21.375 and state.base_pos_w[2] >= 3.0:
            top_reached = True
            break
        position, velocity = decode_action_raw(last)
        for _ in range(20):
            data.ctrl[:] = kp * (position - data.qpos[7:23]) + kd * (velocity - data.qvel[6:22])
            mujoco.mj_step(model, data)
    result = {"scene": args.scene, "observation": args.observation,
              "support_surface": args.support_surface, "half_width": args.half_width,
              "x_range": args.x_range, "start": [args.start_x, args.start_y, args.yaw],
              "router": args.router,
              "forget_router_on_resume": args.forget_router_on_resume,
              "pause_start_s": None if pause_start is None else pause_start * .02,
              "pause_seconds": args.pause_seconds if pause_start is not None else 0.,
              "actor_sha256": entry["sha256"], "top_reached": top_reached,
              "top_reached_is_not_stop_or_router_acceptance": True,
              "final_wheel_z": data.xpos[wheels, 2].tolist(),
              "final_wheel_contact": wheel_contact_mask(model, data, wheels).tolist(),
              "wall_seconds": time.perf_counter() - started,
              "time_s": step * .02, "final_position": state.base_pos_w.tolist(), "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f'top_reached={top_reached} output={args.output}', flush=True)


if __name__ == "__main__":
    main()
