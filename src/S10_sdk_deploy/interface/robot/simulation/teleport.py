"""
 * @file teleport.py
 * @brief Thin wrapper resetting the robot pose on a `/S10_TELEPORT` request.
 *
 * Reuses the same reset logic as `_set_initial_pose` (joint reset to JOINT_INIT,
 * base pose set from a target, then `mj_forward`) so TD can reuse this module
 * without reimplementing navigation/reset.
 *
 * @copyright Copyright (c) 2025  DeepRobotics
"""

from geometry_msgs.msg import PoseStamped

TOPIC_TELEPORT = "/S10_TELEPORT"


class TeleportHandler:
    """Subscribes `/S10_TELEPORT` and applies the requested base pose on the sim.

    The target pose is a full base pose: position (x, y, z) and orientation
    quaternion in geometry_msgs order (x, y, z, w). MuJoCo qpos stores base pos
    in ``qpos[0:3]`` and base quat in ``qpos[3:7]`` as (w, x, y, z), so the
    orientation must be reordered back from xyzw to wxyz on apply.
    """

    def __init__(self, node, apply_callback, qos_depth: int = 10, enabled: bool = True):
        """
        Args:
            node: The ROS node (rclpy Node).
            apply_callback: Callable(x, y, z, qw, qx, qy, qz) that actually moves
                the robot in the simulation. This keeps TeleportHandler decoupled
                from MuJoCo internals and lets the sim node own the reset logic.
            enabled: When False, subscribe but ignore teleports (eval scoring).
        """
        self._apply = apply_callback
        self._enabled = enabled
        self._node = node
        self._sub = node.create_subscription(
            PoseStamped, TOPIC_TELEPORT, self._on_teleport, qos_depth
        )

    def _on_teleport(self, msg: PoseStamped) -> None:
        if not self._enabled:
            self._node.get_logger().warn("[TELEPORT] ignored (eval / S10_ALLOW_TELEPORT=0)")
            return
        p = msg.pose.position
        q = msg.pose.orientation
        # geometry_msgs xyzw -> MuJoCo wxyz
        self._apply(
            float(p.x),
            float(p.y),
            float(p.z),
            float(q.w),
            float(q.x),
            float(q.y),
            float(q.z),
        )
