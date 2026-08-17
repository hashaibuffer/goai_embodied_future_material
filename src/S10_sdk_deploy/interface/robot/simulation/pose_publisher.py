"""
 * @file pose_publisher.py
 * @brief Thin wrapper publishing ground-truth base_link pose as PoseStamped.
 *
 * Reads MuJoCo world-frame truth for the `base_link` body and republishes it as
 * `geometry_msgs/msg/PoseStamped` on `/S10_BASE_POSE` for the AutoNav command
 * interface to consume.
 *
 * @copyright Copyright (c) 2025  DeepRobotics
"""

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped

TOPIC_BASE_POSE = "/S10_BASE_POSE"
# The pose below is expressed in the world (sim) frame. The plan pins
# frame_id to "base_link" (the body whose pose we report); keep it as a single
# constant so TD/TF consumers can repoint it without touching call sites.
FRAME_ID = "base_link"


def make_pose_stamped(xpos, xquat, timestamp: float) -> PoseStamped:
    """Assemble a PoseStamped from MuJoCo world-frame base_link truth.

    MuJoCo stores quaternions as (w, x, y, z) but geometry_msgs/Quaternion uses
    (x, y, z, w), so the components must be reordered.
    """
    msg = PoseStamped()
    msg.header.frame_id = FRAME_ID

    stamp = Time()
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1e9)
    stamp.sec = sec
    stamp.nanosec = nanosec
    msg.header.stamp = stamp

    msg.pose.position.x = float(xpos[0])
    msg.pose.position.y = float(xpos[1])
    msg.pose.position.z = float(xpos[2])

    # wxyz -> xyzw
    msg.pose.orientation.x = float(xquat[1])
    msg.pose.orientation.y = float(xquat[2])
    msg.pose.orientation.z = float(xquat[3])
    msg.pose.orientation.w = float(xquat[0])
    return msg


class PosePublisher:
    """Owns a `/S10_BASE_POSE` publisher bound to the given ROS node."""

    def __init__(self, node, qos_depth: int = 10):
        self._pub = node.create_publisher(PoseStamped, TOPIC_BASE_POSE, qos_depth)

    def publish(self, xpos, xquat, timestamp: float) -> None:
        self._pub.publish(make_pose_stamped(xpos, xquat, timestamp))
