#!/usr/bin/env python3
"""heightmap_vis_node.py — S10 高度图/雷达 RViz 可视化转发节点（独立新节点）

RViz 没有 Float32MultiArray 显示插件，本节点把已有的消息转发成 RViz 能渲染的类型，
并发布 TF，使 RViz 固定 world 时高度图和雷达随小车移动。

订阅：
  - /S10_HEIGHTMAP  (std_msgs/Float32MultiArray, CHW 展平 (2, policy_nx, policy_ny))
  - /S10_SIM_LIDAR  (sensor_msgs/PointCloud2, frame=world)
  - /S10_BASE_POSE  (geometry_msgs/PoseStamped, frame=world, 官方仿真真值)

发布（仅供 RViz 显示）：
  - /tf             TF robot_horizontal → world（yaw-only，从位姿提取，去掉 roll/pitch）
  - /S10_HEIGHTMAP_VIS   (MarkerArray) 每有效格一个彩色方块：高度=归一化高度×0.4m，
                    颜色 蓝(0)→绿(0.5)→红(1)，frame=robot_horizontal
  - /S10_SIM_LIDAR_BODY  (PointCloud2) world 点云经 yaw-only 转到车系，frame=robot_horizontal

lidar_node.py / heightmap.py（T4 核心）零改动，本节点只做可视化转发。

运行：python3 src/s10_terrain_perception/heightmap_vis_node.py
"""
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, ColorRGBA, Header
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import heightmap as hm  # noqa: E402

WORLD = "world"
HEIGHTMAP_VIS_TOPIC = "/S10_HEIGHTMAP_VIS"
LIDAR_BODY_TOPIC = "/S10_SIM_LIDAR_BODY"
POSE_TOPIC = "/S10_BASE_POSE"
VIS_HEIGHT_SCALE = 0.4          # 归一化高度 1.0 → 0.4m 方块高（可视化尺度）


class HeightmapVisNode(Node):
    def __init__(self):
        super().__init__("heightmap_vis")
        self.cfg = hm.load_config()
        self.frame = self.cfg["frame"]                     # robot_horizontal
        self.nx, self.ny = int(self.cfg["policy_nx"]), int(self.cfg["policy_ny"])
        self.x_min, self.y_min = float(self.cfg["x_min"]), float(self.cfg["y_min"])
        self.x_max, self.y_max = float(self.cfg["x_max"]), float(self.cfg["y_max"])
        self.res_x = (self.x_max - self.x_min) / self.nx
        self.res_y = (self.y_max - self.y_min) / self.ny

        self._tf_pub = self.create_publisher(TFMessage, "/tf", 10)
        self._hm_vis_pub = self.create_publisher(MarkerArray, HEIGHTMAP_VIS_TOPIC, 10)
        self._lidar_body_pub = self.create_publisher(PointCloud2, LIDAR_BODY_TOPIC, 10)
        self.create_subscription(Float32MultiArray, self.cfg["topic"], self._hm_cb, 10)
        self.create_subscription(PointCloud2, "/S10_SIM_LIDAR", self._lidar_cb, 10)
        self.create_subscription(PoseStamped, POSE_TOPIC, self._pose_cb, 10)

        self._grid = None            # (2, nx, ny) float32
        self._points = None          # (N,3) world
        self._pos = np.zeros(3)
        self._yaw = 0.0

        self.create_timer(0.1, self._tick)                  # 10Hz 刷新 TF + 方块 + 车系点云
        self.get_logger().info(
            f"heightmap_vis 就绪: {self.cfg['topic']} + /S10_SIM_LIDAR → "
            f"{HEIGHTMAP_VIS_TOPIC} / {LIDAR_BODY_TOPIC} + TF {self.frame}→{WORLD}")

    # ---------- 订阅回调 ----------
    def _hm_cb(self, msg):
        if len(msg.layout.dim) != 3 or msg.layout.dim[0].size != 2:
            self.get_logger().warn(f"高度图 layout 异常: {len(msg.layout.dim)} dims")
            return
        data = np.asarray(msg.data, dtype=np.float32)
        try:
            grid = data.reshape(2, self.nx, self.ny)
        except ValueError:
            self.get_logger().warn(f"高度图长度 {len(data)} ≠ 2×{self.nx}×{self.ny}")
            return
        self._grid = grid

    def _lidar_cb(self, msg):
        n = int(msg.width) * int(msg.height)
        if n == 0 or msg.point_step < 12:
            return
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
        pts = np.empty((n, 3), dtype=np.float32)
        for k, name in enumerate(("x", "y", "z")):
            f = next(f for f in msg.fields if f.name == name)
            # 取该字段的 4 字节列 (n,4) → view float32 → (n,)
            pts[:, k] = raw[:, f.offset:f.offset + 4].copy().view(np.float32).ravel()
        self._points = pts

    def _pose_cb(self, msg):
        p = msg.pose
        self._pos = np.array([p.position.x, p.position.y, p.position.z])
        q = p.orientation
        # yaw = atan2(R[1,0], R[0,0])；四元数 R[1,0]=2(wz+xy), R[0,0]=1-2(y²+z²)
        x, y, z, w = q.x, q.y, q.z, q.w
        self._yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    # ---------- 发布 ----------
    def _tick(self):
        # 统一时间戳：TF 与所有显示消息用同一时刻，避免 RViz 查变换时
        # "extrapolation into the future"（点云 stamp 比 TF 新 0.24ms 就会触发）
        now = self.get_clock().now().to_msg()
        self._publish_tf(now)
        if self._grid is not None:
            self._hm_vis_pub.publish(self._make_heightmap_markers(now))
        if self._points is not None and len(self._points):
            self._lidar_body_pub.publish(self._make_body_pointcloud(now))

    def _publish_tf(self, stamp):
        """TF robot_horizontal → world（yaw-only，车动图跟着动）。"""
        ts = TransformStamped()
        ts.header.stamp = stamp
        ts.header.frame_id = WORLD
        ts.child_frame_id = self.frame
        ts.transform.translation.x = float(self._pos[0])
        ts.transform.translation.y = float(self._pos[1])
        ts.transform.translation.z = float(self._pos[2])
        ts.transform.rotation.x = 0.0
        ts.transform.rotation.y = 0.0
        ts.transform.rotation.z = float(np.sin(self._yaw / 2.0))
        ts.transform.rotation.w = float(np.cos(self._yaw / 2.0))
        self._tf_pub.publish(TFMessage(transforms=[ts]))

    def _make_heightmap_markers(self, stamp):
        """高度图 (2,nx,ny) → MarkerArray：每有效格一个彩色方块（frame=robot_horizontal）。"""
        h, mask = self._grid[0], self._grid[1]
        ma = MarkerArray()
        clean = Marker()
        clean.header.frame_id = self.frame
        clean.ns = "heightmap"
        clean.action = Marker.DELETEALL
        clean.id = 0
        ma.markers.append(clean)

        for i in range(self.nx):
            for j in range(self.ny):
                if mask[i, j] <= 0.5:
                    continue
                v = float(min(max(float(h[i, j]), 0.0), 1.0))
                zh = max(v * VIS_HEIGHT_SCALE, 0.02)         # 最矮留 2cm 薄片
                m = Marker()
                m.header.frame_id = self.frame
                m.header.stamp = stamp
                m.ns = "heightmap"
                m.id = i * self.ny + j + 1
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = self.x_min + (i + 0.5) * self.res_x
                m.pose.position.y = self.y_min + (j + 0.5) * self.res_y
                m.pose.position.z = zh / 2.0
                m.pose.orientation.w = 1.0
                m.scale.x = self.res_x * 0.95
                m.scale.y = self.res_y * 0.95
                m.scale.z = zh
                m.color = self._height_color(v)
                m.color.a = 0.85
                ma.markers.append(m)
        return ma

    def _make_body_pointcloud(self, stamp):
        """world 点云经 yaw-only 转到车系（frame=robot_horizontal），雷达随车移动。"""
        yaw = self._yaw
        Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                       [np.sin(yaw), np.cos(yaw), 0.0],
                       [0.0, 0.0, 1.0]])
        ph = (Ry.T @ (self._points - self._pos).T).T

        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame
        msg.height = 1
        msg.width = len(ph)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = 12 * len(ph)
        msg.is_bigendian = False
        msg.is_dense = True
        msg.data = ph.astype(np.float32).tobytes()
        return msg

    @staticmethod
    def _height_color(v):
        """蓝(0) → 绿(0.5) → 红(1)。"""
        c = ColorRGBA()
        if v < 0.5:
            t = v * 2.0
            c.r, c.g, c.b = 0.0, t, 1.0 - t
        else:
            t = (v - 0.5) * 2.0
            c.r, c.g, c.b = t, 1.0 - t, 0.0
        c.a = 1.0
        return c


def main(args=None):
    rclpy.init(args=args)
    node = HeightmapVisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
