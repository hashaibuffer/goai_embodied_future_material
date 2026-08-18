#!/usr/bin/env python3
"""
base_pose.py — 占位位姿节点（TB / T2）

发布 /S10_BASE_POSE（geometry_msgs/PoseStamped, frame=world）：
  - 位置：固定为起点 [0, -2.5, 0.2]（= 官方 TRACK_START_BASE_POS）
  - 姿态：订阅 /IMU_DATA 的 rpy（度），按官方 quaternion_to_euler 约定重建四元数

占位性质：TA（s10_waypoint_navigation）实现真值位姿后停用本节点，
lidar 只依赖 /S10_BASE_POSE 话题契约，不受影响。

运行：python3 src/s10_terrain_perception/base_pose.py
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from drdds.msg import ImuData
from scipy.spatial.transform import Rotation

INIT_BASE_POS = (0.0, -2.5, 0.2)   # 官方 TRACK_START_BASE_POS
POSE_TOPIC = "/S10_BASE_POSE"
IMU_TOPIC = "/IMU_DATA"


class BasePoseNode(Node):
    def __init__(self):
        super().__init__("base_pose")
        self.pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, 10)
        self.create_subscription(ImuData, IMU_TOPIC, self._imu_cb, 10)
        self.get_logger().info(
            f"base_pose 占位节点就绪：位置固定 {INIT_BASE_POS}，姿态来自 /IMU_DATA rpy(度)"
        )

    def _imu_cb(self, msg):
        p = PoseStamped()
        # 注意：drdds 的 MetaType.frame_id 是 uint64，std_msgs Header.frame_id 是 string，
        # 不能直接 p.header = msg.header（序列化时会断言崩溃），只拷贝 stamp。
        p.header.stamp = msg.header.stamp
        p.header.frame_id = "world"
        p.pose.position.x = INIT_BASE_POS[0]
        p.pose.position.y = INIT_BASE_POS[1]
        p.pose.position.z = INIT_BASE_POS[2]
        # 官方 quaternion_to_euler 对应 R = Rz(yaw)@Ry(pitch)@Rx(roll)
        # scipy 内旋 'ZYX' 传 [yaw, pitch, roll]
        roll, pitch, yaw = msg.data.roll, msg.data.pitch, msg.data.yaw
        quat = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_quat()
        p.pose.orientation.x = quat[0]
        p.pose.orientation.y = quat[1]
        p.pose.orientation.z = quat[2]
        p.pose.orientation.w = quat[3]
        self.pose_pub.publish(p)


def main(args=None):
    rclpy.init(args=args)
    node = BasePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
