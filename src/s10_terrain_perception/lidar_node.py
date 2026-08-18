#!/usr/bin/env python3
"""
lidar_node.py — S10 模拟雷达扫描节点（TB / T2 + T3 异步化）

- 加载带雷达的 MJCF：环境变量 S10_MUJOCO_XML 或默认根目录 models/mjcf/S10_track_lidar.xml
- 订阅 /S10_BASE_POSE（真值位姿，占位节点 base_pose 或 TA 提供）
- 官方参数 181×24=4344 条射线（前向 180° × 0°~-55° 向下偏置，量程 6m，盲区 0.1m）
- 过滤 overlay(group=2) 和机器人自几何：geomgroup + flg_static + bodyexclude
- T3：mj_multiRay 扫描移到独立 daemon 线程，最新帧缓存（ScanFrame）解耦；
  rclpy 定时器只做发布（20Hz 稳定输出最近一帧），扫描不阻塞 rclpy 主循环
- 命中点转世界坐标，发布 sensor_msgs/PointCloud2（frame=world）

运行：python3 src/s10_terrain_perception/lidar_node.py
"""
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

BASE_DIR = Path(__file__).resolve().parent
WS_ROOT = BASE_DIR.parent.parent        # 本文件在 workspace_root/src/s10_terrain_perception/，上两级 = 根


@dataclass
class ScanFrame:
    """一帧扫描结果（最新帧缓存）。points 放置后不再变异，锁内整体交换引用。"""
    seq: int                 # 单调递增帧号（可观察重复/丢帧）
    stamp: object            # 当时用的 pose.header.stamp（沿用现状）
    points: np.ndarray       # (N,3) 世界系，已加噪声/丢点
    n_hit: int               # 本帧原始命中数
    scan_ms: float           # 本帧扫描耗时 (ms)
    scan_done_t: float       # time.monotonic() 扫描完成时刻（算帧龄）
    pos: np.ndarray          # (3,) 位姿快照，供调试 / T4 heightmap
    R: np.ndarray            # (3,3) 姿态快照
DEFAULT_XML = WS_ROOT / "models" / "mjcf" / "S10_track_lidar.xml"
DEFAULT_CONFIG = WS_ROOT / "configs" / "lidar.yaml"

# 默认值 = 官方 T00 lidar.yaml（180°×24 线×6.0m×20Hz，0°~-55° 向下偏置，盲区 0.1m）
DEFAULTS = {
    "azimuth_deg": [-90.0, 90.0],
    "azimuth_beams": 181,
    "elevation_deg": [0.0, -55.0],
    "elevation_beams": 24,
    "range_min": 0.10,
    "cutoff": 6.0,
    "rate_hz": 20.0,
    "pointcloud_topic": "/S10_SIM_LIDAR",
    "pointcloud_frame": "world",
    "geomgroup": [1, 1, 0, 1, 1, 1],   # 全开，减 exclude_geom_groups=[2]
    "flg_static": True,
    "site_name": "lidar_front_site",
    "body_exclude": "base_link",
    "range_noise_std_m": 0.01,
    "dropout_probability": 0.03,
    "log_interval_s": 5.0,
}


def load_config(path=None):
    """读取雷达配置，兼容官方 T00 格式（/home/hashvr/下载/lidar.yaml）与自有格式。

    官方字段映射：
      topics.lidar → pointcloud_topic
      horizontal_fov_deg / horizontal_rays → azimuth_deg / azimuth_beams
      vertical_fov_deg / vertical_rays → elevation_deg(0~-fov 向下偏置) / elevation_beams
      range_min / range_max → range_min / cutoff
      update_rate_hz → rate_hz
      exclude_geom_groups → geomgroup（全开，关掉列出的组）
      body_frame_origin_m → site_origin（仅参考，site 坐标以模型为准）
      range_noise_std_m / dropout_probability → 同名字段
    """
    cfg = dict(DEFAULTS)
    p = Path(path) if path else DEFAULT_CONFIG
    if p.is_file():
        try:
            import yaml
            data = yaml.safe_load(p.read_text())
            l = (data.get("lidar") or {}) if isinstance(data, dict) else {}
            topics = (data.get("topics") or {}) if isinstance(data, dict) else {}

            if isinstance(topics, dict) and "lidar" in topics:
                cfg["pointcloud_topic"] = topics["lidar"]

            if "horizontal_rays" in l:
                cfg["azimuth_beams"] = int(l["horizontal_rays"])
            elif "azimuth_beams" in l:
                cfg["azimuth_beams"] = int(l["azimuth_beams"])
            if "horizontal_fov_deg" in l:
                hf = float(l["horizontal_fov_deg"])
                cfg["azimuth_deg"] = [-hf / 2.0, hf / 2.0]

            if "vertical_rays" in l:
                cfg["elevation_beams"] = int(l["vertical_rays"])
            elif "elevation_beams" in l:
                cfg["elevation_beams"] = int(l["elevation_beams"])
            if "vertical_fov_deg" in l:
                # 0°~-fov 向下偏置（模仿真机 0~90° 向下方向，消除脚下近场盲区）
                cfg["elevation_deg"] = [0.0, -float(l["vertical_fov_deg"])]
            elif "elevation_deg" in l:
                cfg["elevation_deg"] = [float(x) for x in l["elevation_deg"]]

            if "range_min" in l:
                cfg["range_min"] = float(l["range_min"])
            if "range_max" in l:
                cfg["cutoff"] = float(l["range_max"])
            elif "cutoff" in l:
                cfg["cutoff"] = float(l["cutoff"])

            if "update_rate_hz" in l:
                cfg["rate_hz"] = float(l["update_rate_hz"])
            elif "rate_hz" in l:
                cfg["rate_hz"] = float(l["rate_hz"])

            if "pointcloud_topic" in l:
                cfg["pointcloud_topic"] = l["pointcloud_topic"]
            if "site_name" in l:
                cfg["site_name"] = l["site_name"]
            if "body_frame_origin_m" in l:
                cfg["site_origin"] = [float(x) for x in l["body_frame_origin_m"]]

            if "exclude_geom_groups" in l:
                gg = [1] * 6
                for grp in l["exclude_geom_groups"]:
                    if 0 <= grp < 6:
                        gg[grp] = 0
                cfg["geomgroup"] = gg
            elif "geomgroup" in l:
                cfg["geomgroup"] = [int(x) for x in l["geomgroup"]]
            # exclude_robot_geoms=true：mj_multiRay 的 bodyexclude 只排除 base_link 自身，
            # 不递归子 body（fr_wheel 等）。机器人碰撞几何全在 group1，直接关闭该组。
            if l.get("exclude_robot_geoms", False) and len(cfg["geomgroup"]) > 1:
                cfg["geomgroup"][1] = 0

            if "range_noise_std_m" in l:
                cfg["range_noise_std_m"] = float(l["range_noise_std_m"])
            if "dropout_probability" in l:
                cfg["dropout_probability"] = float(l["dropout_probability"])
        except Exception as e:  # noqa: BLE001
            print(f"[lidar_node] 读取配置 {p} 失败，使用默认值: {e}")
    return cfg


class LidarNode(Node):
    def __init__(self, cfg=None):
        super().__init__("s10_lidar")
        self.cfg = cfg or load_config()

        xml_path = os.environ.get("S10_MUJOCO_XML") or str(DEFAULT_XML)
        xml_path = str(Path(xml_path).expanduser().resolve())
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find MJCF: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.cfg["site_name"])
        self.base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg["body_exclude"])
        if self.site_id < 0:
            raise RuntimeError(f"site '{self.cfg['site_name']}' 不在模型中")
        self.site_offset = self.model.site_pos[self.site_id].copy()   # 局部坐标 (m)

        # 预生成 fan 局部方向（雷达局部系：+x 为前方）
        az = np.linspace(self.cfg["azimuth_deg"][0], self.cfg["azimuth_deg"][1],
                         int(self.cfg["azimuth_beams"])) * np.pi / 180.0
        el = np.linspace(self.cfg["elevation_deg"][0], self.cfg["elevation_deg"][1],
                         int(self.cfg["elevation_beams"])) * np.pi / 180.0
        fan = [[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]
               for e in el for a in az]
        self.fan = np.asarray(fan, dtype=np.float64)          # (N,3)
        self.nray = len(self.fan)
        self.geomgroup = np.asarray(self.cfg["geomgroup"], dtype=np.uint8)
        self.range_min = float(self.cfg.get("range_min", 0.0))       # 盲区 (m)
        self.noise_std = float(self.cfg.get("range_noise_std_m", 0.0))
        self.dropout = float(self.cfg.get("dropout_probability", 0.0))

        self.pc_pub = self.create_publisher(PointCloud2, self.cfg["pointcloud_topic"], 10)
        self.create_subscription(PoseStamped, "/S10_BASE_POSE", self._pose_cb, 10)

        # ---- T3 异步化：扫描独立线程 + 最新帧缓存 ----
        self._lock = threading.Lock()          # 保护 _latest_pose / _latest_frame / _st
        self._latest_pose = None
        self._latest_frame = None
        self._last_pub = None
        self._frame_seq = 0
        self._stop = threading.Event()
        self._st = {                            # 跨线程性能统计（锁保护）
            "scan_ms_sum": 0.0, "scan_count": 0, "scan_ms_max": 0.0,
            "dup_pub": 0, "pub_count": 0, "pub_interval_sum": 0.0,
            "pub_interval_max": 0.0, "last_pub_t": None,
        }
        self._scan_count = 0                    # 5s 窗口扫描次数（仅 scan 线程用）
        self._scan_log_last = time.monotonic()
        self._pub_log_last = time.monotonic()

        period = 1.0 / float(self.cfg["rate_hz"])
        self.create_timer(period, self._publish_latest)      # rclpy 定时器只做发布
        self._scan_thread = threading.Thread(
            target=self._scan_loop, daemon=True, name="lidar_scan")
        self._scan_thread.start()

        self.get_logger().info(
            f"lidar 就绪(异步): model={Path(xml_path).name} nray={self.nray} "
            f"site='{self.cfg['site_name']}' offset={self.site_offset.round(3).tolist()} "
            f"cutoff={self.cfg['cutoff']}m rate={self.cfg['rate_hz']}Hz "
            f"topic={self.cfg['pointcloud_topic']} 扫描线程已启动")

    def _pose_cb(self, msg):
        with self._lock:
            self._latest_pose = msg

    @staticmethod
    def _pose_to_rotmat(msg):
        q = msg.pose.orientation
        return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def _scan_loop(self):
        """scan 线程：按 rate_hz 节拍扫描，把最新一帧写入缓存（可中断等待）。"""
        period = 1.0 / float(self.cfg["rate_hz"])
        next_due = time.monotonic()
        while not self._stop.is_set() and rclpy.ok():
            next_due += period
            self._scan_once()
            delay = next_due - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)          # 节拍 + 可中断
            else:
                next_due = time.monotonic()     # 落后则跳过补拍（不追赶）

    def _scan_once(self):
        """单帧扫描（原 _scan 核心）：mj_multiRay + 过滤 + 噪声/丢点，锁外执行重活。"""
        with self._lock:                        # 只取位姿引用，锁内不做重活
            pose = self._latest_pose
        if pose is None:
            return
        pos = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        R = self._pose_to_rotmat(pose)

        pnt = pos + R @ self.site_offset
        vec_world = (R @ self.fan.T).T                        # (N,3)
        dist = np.full(self.nray, -1.0, dtype=np.float64)
        geomid = np.full(self.nray, -1, dtype=np.int32)

        t0 = time.perf_counter()
        mujoco.mj_multiRay(
            self.model, self.data, pnt, vec_world.reshape(-1),
            self.geomgroup, bool(self.cfg["flg_static"]), self.base_body,
            geomid, dist, None, self.nray, float(self.cfg["cutoff"]),
        )

        hit = geomid >= 0
        # 注意：mj_multiRay 的 cutoff 参数对地形 mesh（static mesh）不裁剪（已实测），
        # 必须显式按量程过滤，否则 0°~-5° 近水平行会拖出几十米远点
        hit &= dist <= float(self.cfg["cutoff"])
        if self.range_min > 0:
            hit &= dist >= self.range_min            # 盲区过滤（官方 range_min 0.10m）
        n_hit = int(hit.sum())
        scan_ms = (time.perf_counter() - t0) * 1000.0

        if n_hit:
            points = pnt[None, :] + vec_world[hit] * dist[hit, None]   # 世界坐标
            # 测距噪声（沿射线方向抖动）+ 随机丢点（模拟真机）
            if self.noise_std > 0:
                points = points + vec_world[hit] * np.random.normal(
                    0, self.noise_std, size=(n_hit, 1))
            if self.dropout > 0:
                points = points[np.random.random(n_hit) >= self.dropout]
            if len(points):
                done_t = time.monotonic()
                with self._lock:                 # 锁内只做引用交换 + 统计（RMW 需锁）
                    self._frame_seq += 1
                    self._latest_frame = ScanFrame(
                        self._frame_seq, pose.header.stamp, points, n_hit,
                        scan_ms, done_t, pos.copy(), R.copy())
                    self._st["scan_count"] += 1
                    self._st["scan_ms_sum"] += scan_ms
                    if scan_ms > self._st["scan_ms_max"]:
                        self._st["scan_ms_max"] = scan_ms

        self._log_scan_stats_if_due(n_hit, geomid[hit])

    def _log_scan_stats_if_due(self, n_hit, hit_geomids):
        """防御日志（scan 线程，锁外）：命中数 + group 分布，每 log_interval_s 一行。"""
        self._scan_count += 1
        if time.monotonic() - self._scan_log_last < self.cfg["log_interval_s"]:
            return
        self._scan_log_last = time.monotonic()
        groups = {}
        for g in hit_geomids:
            grp = self.model.geom_group[int(g)]
            groups[grp] = groups.get(grp, 0) + 1
        bad = sorted(g for g in groups if g != 0)
        line = f"扫描 {self._scan_count} 次 | 命中 {n_hit}/{self.nray} | group 分布 {groups}"
        if bad:
            self.get_logger().warn(line + f"  ⚠️ 命中非 group0: {bad}")
        else:
            self.get_logger().info(line)
        self._scan_count = 0

    def _publish_latest(self):
        """rclpy 定时器回调（主线程）：从最新帧缓存发布，锁内只做引用交换。"""
        now_t = time.monotonic()
        with self._lock:
            frame = self._latest_frame
            if frame is not None and frame is self._last_pub:
                self._st["dup_pub"] += 1        # 扫描未产出新帧，复用上一帧
            self._last_pub = frame
            st = self._st
            last = st["last_pub_t"]
            if last is not None:
                dt = now_t - last
                st["pub_interval_sum"] += dt
                st["pub_count"] += 1
                if dt > st["pub_interval_max"]:
                    st["pub_interval_max"] = dt
            st["last_pub_t"] = now_t
        if frame is None:
            return
        self.pc_pub.publish(self._make_pointcloud(frame.points, frame.stamp))
        self._log_stats_if_due(frame)

    def _log_stats_if_due(self, frame):
        """每 log_interval_s 打一行 [perf] 性能日志（rclpy 主线程，锁外读统计快照）。"""
        if time.monotonic() - self._pub_log_last < self.cfg["log_interval_s"]:
            return
        self._pub_log_last = time.monotonic()
        with self._lock:
            st = self._st
            scan_avg = (st["scan_ms_sum"] / st["scan_count"]) if st["scan_count"] else 0.0
            pub_avg = (st["pub_interval_sum"] / st["pub_count"]) if st["pub_count"] else 0.0
            scan_max, pub_max, dup = st["scan_ms_max"], st["pub_interval_max"], st["dup_pub"]
            # 复位 5s 窗口统计
            st.update(scan_ms_sum=0.0, scan_count=0, scan_ms_max=0.0,
                      dup_pub=0, pub_count=0, pub_interval_sum=0.0,
                      pub_interval_max=0.0, last_pub_t=None)
        age_ms = (time.monotonic() - frame.scan_done_t) * 1000.0
        self.get_logger().info(
            f"[perf] 帧 {self._frame_seq} | 扫描 avg {scan_avg:.1f}ms max {scan_max:.1f}ms | "
            f"发布间隔 avg {pub_avg:.1f}ms max {pub_max:.1f}ms | "
            f"重复帧 {dup} | 最近帧龄 {age_ms:.0f}ms | 命中 {frame.n_hit}/{self.nray}")

    def stop(self):
        """停止 scan 线程（rclpy 关停前调用）。"""
        self._stop.set()
        if getattr(self, "_scan_thread", None) is not None:
            self._scan_thread.join(timeout=2.0)

    def _make_pointcloud(self, points, stamp):
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = self.cfg["pointcloud_frame"]
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = 12 * len(points)
        msg.is_bigendian = False
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()                 # 先停 scan 线程（_stop.set + join），再销毁
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
