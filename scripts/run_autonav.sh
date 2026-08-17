#!/usr/bin/env bash
# One-shot launcher for TA autonomous navigation (eval or collect mode).
#
# Usage:
#   scripts/run_autonav.sh [eval|collect] [headless|viewer]
# collect defaults to headless (S10_MUJOCO_VIEWER=0).
#
# Starts the MuJoCo simulator (ground-truth pose publisher) and the rl_deploy
# AutoNav controller in the background, then tails their logs. Ctrl-C stops both.

set -euo pipefail

MODE="${1:-eval}"
EXTRA="${2:-}"
export S10_AUTONAV_MODE="$MODE"

# Collect defaults to headless so long stall/teleport runs do not need a display.
# Override with S10_MUJOCO_VIEWER=1 or pass "viewer" as the second argument.
if [[ "$EXTRA" == "headless" || "$EXTRA" == "--headless" ]]; then
  export S10_MUJOCO_VIEWER=0
elif [[ "$EXTRA" == "viewer" || "$EXTRA" == "--viewer" ]]; then
  export S10_MUJOCO_VIEWER=1
elif [[ "$MODE" == "collect" && -z "${S10_MUJOCO_VIEWER:-}" ]]; then
  export S10_MUJOCO_VIEWER=0
fi

# Resolve repo root (this script lives in <repo>/scripts).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/install/setup.bash"

LOG_DIR="${S10_RESULTS_DIR:-$REPO_ROOT/results}"
mkdir -p "$LOG_DIR"
export S10_RESULTS_DIR="$LOG_DIR"

SIM_SCRIPT="$REPO_ROOT/src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py"

echo "[run_autonav] mode=$MODE viewer=${S10_MUJOCO_VIEWER:-1} results=$LOG_DIR"
echo "[run_autonav] starting simulator ..."
python3 "$SIM_SCRIPT" > "$LOG_DIR/sim.log" 2>&1 &
SIM_PID=$!

sleep 3

echo "[run_autonav] starting rl_deploy (AutoNav) ..."
ros2 run s10_sdk_deploy rl_deploy > "$LOG_DIR/rl_deploy.log" 2>&1 &
DEPLOY_PID=$!

cleanup() {
  echo "[run_autonav] shutting down ..."
  kill "$DEPLOY_PID" 2>/dev/null || true
  kill "$SIM_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM

echo "[run_autonav] running. logs: $LOG_DIR/sim.log, $LOG_DIR/rl_deploy.log"
tail -f "$LOG_DIR/rl_deploy.log" &
TAIL_PID=$!

wait "$DEPLOY_PID"
kill "$TAIL_PID" 2>/dev/null || true
cleanup
