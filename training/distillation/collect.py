#!/usr/bin/env python3
"""ROS 2 TD collector: correlate teacher samples with TA status and write NPZ."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from collections import deque
from pathlib import Path

import numpy as np

from schema import ChunkedDatasetWriter

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--pre-failure-seconds", type=float, default=5.0)
    args = parser.parse_args()

    import rclpy
    from drdds.msg import AutoNavStatus, TeacherSample
    from rclpy.node import Node

    metadata = {
        "git_commit": git_commit(),
        "seed": args.seed,
        "teacher_source_codes": {"official": 0, "privileged": 1},
        "failure_codes": {"none": 0, "stall": 1, "tumble": 2, "out_of_bounds": 3},
        "policy_config_sha256": sha256(ROOT / "configs/policy.yaml"),
        "heightmap_config_sha256": sha256(ROOT / "configs/heightmap.yaml"),
        "teacher_collect_config_sha256": sha256(ROOT / "configs/teacher_collect.yaml"),
        "official_policy_sha256": sha256(ROOT / "src/S10_sdk_deploy/policy/policy.onnx"),
        "initial_pose_jitter": {
            "x_m": float(os.environ.get("S10_START_JITTER_X", "0")),
            "y_m": float(os.environ.get("S10_START_JITTER_Y", "0")),
            "yaw_rad": float(os.environ.get("S10_START_JITTER_YAW", "0")),
        },
    }
    writer = ChunkedDatasetWriter(args.output, metadata, args.chunk_size)

    class Collector(Node):
        def __init__(self):
            super().__init__("td_teacher_collect")
            self.status = None
            self.pending = deque()
            self.last_next_wp = -1
            self.create_subscription(TeacherSample, "/S10_TD_SAMPLE", self.sample_cb, 50)
            self.create_subscription(AutoNavStatus, "/S10_AUTONAV_STATUS", self.status_cb, 20)

        def status_cb(self, msg):
            if msg.failure_code:
                for record in self.pending:
                    record["pre_failure"] = True
                    record["failure_code"] = int(msg.failure_code)
            if self.last_next_wp >= 0 and msg.next_waypoint_id > self.last_next_wp:
                for record in self.pending:
                    if record["next_wp_id"] == self.last_next_wp:
                        record["success"] = True
            self.last_next_wp = int(msg.next_waypoint_id)
            self.status = msg

        def sample_cb(self, msg):
            status = self.status
            pose = np.zeros(7, dtype=np.float32)
            episode = wp_id = next_wp = 0
            if status is not None:
                p = status.pose
                pose[:] = [p.position.x, p.position.y, p.position.z,
                           p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
                episode = status.episode_id
                wp_id = status.waypoint_id
                next_wp = status.next_waypoint_id
            record = {
                "obs_student": msg.obs_student,
                "obs_teacher": msg.obs_teacher,
                "heightmap": np.asarray(msg.heightmap, dtype=np.float32).reshape(2, 16, 12),
                "action_teacher": msg.action_teacher,
                "cmd_raw": msg.cmd_raw,
                "cmd_terrain": msg.cmd_terrain,
                "risk_features": msg.risk_features,
                "pose": pose,
                "timestamp_ns": msg.timestamp_ns,
                "sequence": msg.sequence,
                "episode_id": episode,
                "wp_id": wp_id,
                "next_wp_id": next_wp,
                "teacher_source": 0,
                "failure_code": 0,
                "success": False,
                "pre_failure": False,
                "heightmap_valid": msg.heightmap_valid,
                "heightmap_age_ms": msg.heightmap_age_ms,
            }
            self.pending.append(record)
            cutoff = msg.timestamp_ns - int(args.pre_failure_seconds * 1e9)
            while self.pending and self.pending[0]["timestamp_ns"] < cutoff:
                writer.append(self.pending.popleft())

        def close(self):
            while self.pending:
                writer.append(self.pending.popleft())
            return writer.close()

    rclpy.init()
    node = Collector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        paths = node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for path in paths:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
