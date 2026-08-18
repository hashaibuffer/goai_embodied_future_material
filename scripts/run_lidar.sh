#!/usr/bin/env bash
# run_lidar.sh — 一键启动 S10 雷达演示（后台 4 进程 + 日志，无 tmux 依赖）
#
#   进程：仿真 → logs/sim.log；位姿 → logs/pose.log；雷达 → logs/lidar.log；RViz2 弹窗 → logs/rviz.log
#
# 用法：
#   ./scripts/run_lidar.sh          # 启动
#   ./scripts/run_lidar.sh stop     # 全部停止
#   ROS_DOMAIN_ID=2 ./scripts/run_lidar.sh   # 自定义 ROS_DOMAIN_ID
#
# 日志位置：results/logs/
set -eo pipefail   # 注意不能加 -u：/opt/ros/jazzy/setup.bash 会引用未绑定变量

ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # 脚本在根目录 scripts/ 下，上 1 级 = workspace 根
LOG_DIR="$WS_ROOT/results/logs"
PID_DIR="${XDG_RUNTIME_DIR:-/tmp}/s10_demo_pids"
SIM_XML="$WS_ROOT/models/mjcf/S10_track_lidar.xml"
SIM_PY="$WS_ROOT/src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py"
BASE_POSE_PY="$WS_ROOT/src/s10_terrain_perception/base_pose.py"
LIDAR_PY="$WS_ROOT/src/s10_terrain_perception/lidar_node.py"

stop() {
    echo "停止 s10_demo 进程..."
    for f in "$PID_DIR"/*.pid; do
        [ -f "$f" ] || continue
        if kill "$(cat "$f")" 2>/dev/null; then
            echo "  ✓ 已停止 $(basename "$f" .pid)"
        fi
        rm -f "$f"
    done
}

# ---- stop 参数 ----
if [ "${1:-}" = "stop" ]; then
    stop
    exit 0
fi

# ---- 前置检查 ----
[ -f /opt/ros/jazzy/setup.bash ] || { echo "✗ 找不到 /opt/ros/jazzy/setup.bash"; exit 1; }
[ -f "$WS_ROOT/install/setup.bash" ] || { echo "✗ 找不到 $WS_ROOT/install/setup.bash，请先 colcon build"; exit 1; }
[ -f "$SIM_XML" ] || { echo "✗ 找不到雷达模型 $SIM_XML"; exit 1; }
[ -f "$SIM_PY" ] || { echo "✗ 找不到官方仿真 $SIM_PY"; exit 1; }

# ---- 已有实例先停 ----
stop

# ---- 统一环境（source 到本进程，子进程直接继承，PID 即实际 python 进程）----
source /opt/ros/jazzy/setup.bash
source "$WS_ROOT/install/setup.bash"
export ROS_DOMAIN_ID
command -v rviz2 >/dev/null || { echo "✗ 找不到 rviz2（安装 ros-jazzy-rviz2 后重试）"; exit 1; }
mkdir -p "$LOG_DIR" "$PID_DIR"

echo "▶ 仿真     → $LOG_DIR/sim.log"
S10_MUJOCO_XML="$SIM_XML" python3 "$SIM_PY" >"$LOG_DIR/sim.log" 2>&1 &
echo $! > "$PID_DIR/sim.pid"

echo "▶ 位姿     → $LOG_DIR/pose.log"
python3 "$BASE_POSE_PY" >"$LOG_DIR/pose.log" 2>&1 &
echo $! > "$PID_DIR/pose.pid"

echo "▶ 雷达     → $LOG_DIR/lidar.log"
python3 "$LIDAR_PY" >"$LOG_DIR/lidar.log" 2>&1 &
echo $! > "$PID_DIR/lidar.pid"

echo "▶ RViz2    → $LOG_DIR/rviz.log"
rviz2 >"$LOG_DIR/rviz.log" 2>&1 &
echo $! > "$PID_DIR/rviz.pid"

sleep 2
echo
echo "✅ 已启动（PID 存于 $PID_DIR）。"
echo "   看日志：  tail -f $LOG_DIR/sim.log $LOG_DIR/pose.log $LOG_DIR/lidar.log"
echo "   全部停止：$0 stop"
