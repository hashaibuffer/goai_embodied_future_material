#!/usr/bin/env python3
"""Isolated runtime smoke: drive rl_deploy to RL mode and observe /JOINTS_CMD."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import threading
import time

import rclpy
from drdds.msg import ImuData, JointsData, JointsDataCmd
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


# Official sim/hardware encoder map: published = (q_raw - offset_rad) * dir.
# Handler reconstructs q_raw = published * dir + offset_rad.
JOINT_DIR = (
    1.0, 1.0, -1.0, 1.0,
    1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, -1.0, 1.0,
    -1.0, -1.0, 1.0, -1.0,
)
POS_OFFSET_DEG = (
    -35.0, -145.0, 156.0, 0.0,
    35.0, -145.0, 156.0, 0.0,
    -35.0, 145.0, -156.0, 0.0,
    35.0, 145.0, -156.0, 0.0,
)
POS_OFFSET_RAD = tuple(deg / 180.0 * math.pi for deg in POS_OFFSET_DEG)
DEFAULT_POSE_ROBOT = (
    0.0, -0.3, 0.6, 0.0,
    0.0, -0.3, 0.6, 0.0,
    0.0, 0.3, -0.6, 0.0,
    0.0, 0.3, -0.6, 0.0,
)


class SmokeNode(Node):
    def __init__(self) -> None:
        super().__init__("tc_joint_cmd_smoke")
        self.joints_pub = self.create_publisher(JointsData, "/JOINTS_DATA", 10)
        self.imu_pub = self.create_publisher(ImuData, "/IMU_DATA", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/S10_BASE_POSE", 10)
        self.create_subscription(JointsDataCmd, "/JOINTS_CMD", self._on_command, 10)
        self.runner_command_seen = False
        self.tick = 0

    def _on_command(self, message: JointsDataCmd) -> None:
        joints = message.data.joints_data
        leg_indices = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
        wheel_indices = (3, 7, 11, 15)
        if all(abs(joints[i].kp - 80.0) < 1e-4 for i in leg_indices) and all(
            abs(joints[i].kp) < 1e-4 and abs(joints[i].kd - 0.6) < 1e-4
            for i in wheel_indices
        ):
            self.runner_command_seen = True

    def publish_inputs(self) -> None:
        self.tick += 1
        stamp = self.get_clock().now().to_msg()
        excitation = 0.001 if self.tick % 2 else -0.001

        joints = JointsData()
        joints.header.stamp = stamp
        for item, q_raw, offset, direction in zip(
            joints.data.joints_data,
            DEFAULT_POSE_ROBOT,
            POS_OFFSET_RAD,
            JOINT_DIR,
        ):
            item.position = (q_raw + excitation - offset) * direction
            item.status_word = 1
        self.joints_pub.publish(joints)

        imu = ImuData()
        imu.header.stamp = stamp
        imu.data.acc_z = 9.81
        self.imu_pub.publish(imu)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.pose.position.x = 0.0
        pose.pose.position.y = -1.725
        pose.pose.position.z = 0.2
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)


def _pump_output(stream, lines: list[str]) -> None:
    for line in stream:
        lines.append(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rl_deploy")
    parser.add_argument("model")
    parser.add_argument("track_overlay")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--use-simulator",
        action="store_true",
        help="consume joint/IMU/pose topics from an already running MuJoCo simulator",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["S10_TRACK_OVERLAY"] = args.track_overlay
    rclpy.init()
    node = SmokeNode()
    process = subprocess.Popen(
        [args.rl_deploy, "--controller", "proprio_clone", "--model-path", args.model],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    logs: list[str] = []
    pump = threading.Thread(
        target=_pump_output, args=(process.stdout, logs), daemon=True
    )
    pump.start()
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline and process.poll() is None:
            if not args.use_simulator:
                node.publish_inputs()
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.runner_command_seen:
                print("Observed TerrainPolicyRunner command on /JOINTS_CMD")
                return 0
        if logs:
            print("".join(logs), end="", file=sys.stderr)
        if process.poll() is not None:
            print(f"rl_deploy exited {process.returncode}", file=sys.stderr)
        else:
            print("timed out waiting for TerrainPolicyRunner /JOINTS_CMD", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
