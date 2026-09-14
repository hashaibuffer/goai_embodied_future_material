#!/usr/bin/env python3
"""Real-time keyboard-controlled MuJoCo playback for the frozen S10 privileged teacher.

Runs the same protocol as scripts/collect_mujoco_d_priv.py (16-actuator MJCF,
StandUp state machine, official 57-dim proprio, 1413-dim privileged teacher
observation, raw action decode) but drives a mujoco.viewer.launch_passive
window instead of collecting a fixed rollout, and reads vx/vy/wz commands
from the keyboard in real time instead of a fixed/scheduled command.

Height scanner sync note (2026-08-22): Isaac Lab lowered the S10 height
scanner ray origin from base+20m to base+1.2m (commit 39cd082, "fix(s10):
lower Isaac height scanner origin") because a 20m-high vertical ray can hit
an overhead structure (roof beam / door frame / gate) on the official track
before it reaches the real ground, producing a false "obstacle ahead"
reading.  PrivilegedHeightScanner in mujoco_teacher.py now mirrors that same
1.2m ray origin (see RAY_ORIGIN_OFFSET_Z there). This reduces high-roof hits
but cannot distinguish a genuinely low suspended slab from a tread. Router
playback therefore adds layered forward clearance rays; they remain outside
the Actor observation and can be displayed with --detector-debug-vis.

Keyboard controls (terminal focus, NOT the viewer window). vx/vy/wz are
independent axes that combine additively -- holding w+a at the same time
drives forward *and* turns simultaneously, exactly like a real joystick's
combined vector, not a single-axis-at-a-time input:
  w / s               hold: vx = +/- --vx-limit   (release -> back to 0)
  a / d               hold: wz = +/- --wz-limit   (release -> back to 0)
  q / e               hold: vy = +/- --vy-limit   (release -> back to 0)
  h                   stand back up IN PLACE (keep x/y/yaw, do not teleport)
  n                   teleport to the next route waypoint in a standing pose
  r                   reset episode: re-run StandUp from --start and clear command
  p                   pause / resume physics stepping
  Ctrl+C / close window to quit

Note: this reads raw keyboard state directly, the same source the official
src/S10_sdk_deploy/interface/user_command/keyboard_interface_sim.hpp uses
for hardware/ROS2 teleop -- NOT mujoco.viewer's key_callback. The MuJoCo
passive viewer's key_callback only fires on GLFW_PRESS (no release event is
ever delivered: python/mujoco/simulate.cc only forwards IsKeyDownEvent), so
there is no way to detect "key released" through it and every ASCII letter
is already bound to a built-in viewer shortcut anyway (W=Wireframe,
S=Shadow, Q=Camera, ... mjVISSTRING/mjRNDSTRING, see
https://github.com/google-deepmind/mujoco/issues/2953).

Two input backends, auto-selected by KeyboardCommand.start():
  1. evdev (preferred): reads /dev/input/eventN raw press(1)/repeat(2)/
     release(0) codes directly, exactly like keyboard_interface_sim.hpp's
     libevdev loop -- a per-key pressed_keys set gives exact simultaneous
     multi-key state, so holding w+a+q all together sums to a correct 3D
     vx/vy/wz vector with no cross-axis interference. Requires read access
     to /dev/input/eventN (Linux: `sudo usermod -aG input $USER`, then log
     out/in -- see keyboard_interface_sim.hpp's own header comment).
  2. termios stdin fallback (used automatically if evdev is unavailable/
     unauthorized, e.g. over SSH or without the input group): a background
     thread reads raw stdin bytes and records a per-key last-seen
     timestamp; a key is "held" until key_timeout_s elapses without a
     repeat. This is degraded relative to evdev -- terminal/keyboard
     N-key-rollover limits mean that on some keyboards only the
     most-recently-pressed key reliably auto-repeats while older
     simultaneously-held keys stall until released, so holding 3 keys at
     once may undercount. Click into the TERMINAL (not the 3D viewer
     window) to drive when using this backend.

Usage:
  python scripts/play_mujoco_teacher.py \
    --xml models/mjcf/S10_track_lidar.xml \
    --router-bundle artifacts/s10_teacher_router/teacher_router_bundle.json \
    --detector-debug-vis
  python scripts/play_mujoco_teacher.py \
    --xml models/mjcf/S10_track_lidar.xml \
    --teacher-onnx artifacts/teacher_model_1700_1413.onnx
  python scripts/play_mujoco_teacher.py \
    --xml models/mjcf/S10_track_lidar.xml \
    --fake-policy   # protocol smoke test, no viewer control loop change

Optional recording (writes a normal D_priv shard while you drive around):
  python scripts/play_mujoco_teacher.py --teacher-onnx ... --record out.npz
"""
from __future__ import annotations
import argparse
import json
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PERCEPTION = ROOT / "src" / "s10_terrain_perception"
TERRAIN_POLICY = ROOT / "src" / "s10_terrain_policy"
sys.path.insert(0, str(PERCEPTION))
sys.path.insert(0, str(TERRAIN_POLICY))
from d_priv_dataset import DPrivRecorder, validate_d_priv
from heightmap import build_heightmap, load_config as load_heightmap_config
from mujoco_climb_detector import S10ClimbDetector
from mujoco_lidar import MujocoLidarScanner, load_lidar_yaml
from mujoco_teacher import (
    DEFAULT_ROBOT, JOINT_INIT_RAW, PrivilegedHeightScanner, assemble_official_57,
    assemble_asymmetric_teacher_inputs, assemble_teacher_1413,
    decode_action_raw, published_targets_to_raw, run_stand_up, state_from_mujoco)
from teacher_skill_router import TeacherSkillRuntime, clamp_height_window
from locomotion_height import (SurfaceSelection, select_support_surface, wheel_tread_heights,
                               scan_xy_world, height_debug_record)

DEFAULT_LIDAR = ROOT / "configs" / "lidar.yaml"
DEFAULT_HEIGHTMAP = ROOT / "configs" / "heightmap.yaml"
# Matches Isaac clip_actions=100 (rsl_rl vecenv_wrapper.step()), not the
# retired [-1,1] a_norm contract.  See doc/MUJOCO_ACTION_NORMALIZATION_FIX.md.
ACTION_RAW_LIMIT = 100.0 + 1e-3
POLICY_DT_S = 0.02
SUBSTEPS_PER_POLICY_STEP = 20


class TeacherPolicy:
    def __init__(self, path, actor_threads=1):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for real playback: pip install onnxruntime") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = actor_threads
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape[-1] != 1413:
            raise ValueError(
                f"teacher ONNX input must be obs[batch,1413], got {[(x.name, x.shape) for x in inputs]}")
        if len(outputs) != 1 or outputs[0].name != "actions" or outputs[0].shape[-1] != 16:
            raise ValueError(
                f"teacher ONNX output must be actions[batch,16], got {[(x.name, x.shape) for x in outputs]}")

    def __call__(self, obs):
        action = self.session.run(["actions"], {"obs": np.asarray(obs, np.float32)[None]})[0][0]
        if action.shape != (16,) or not np.isfinite(action).all():
            raise RuntimeError("teacher returned an invalid action")
        max_abs = float(np.max(np.abs(action)))
        if max_abs > ACTION_RAW_LIMIT:
            raise RuntimeError(
                f"teacher output exceeds the raw action safety limit: max_abs={max_abs:.6g} > "
                f"{ACTION_RAW_LIMIT:.6g}; re-export with --action-postprocess clip "
                "--action-limit 100 (matching Isaac clip_actions=100), or investigate why "
                "the actor produced an out-of-distribution value")
        return action.astype(np.float32)


class ZeroPolicy:
    def __call__(self, obs):
        if np.asarray(obs).shape != (1413,):
            raise ValueError("fake policy still requires a valid 1413-D observation")
        return np.zeros(16, np.float32)


class _EvdevBackend:
    """Preferred backend: mirrors keyboard_interface_sim.hpp exactly -- reads
    raw press(1)/repeat(2)/release(0) EV_KEY events from every /dev/input/
    eventN that looks like a keyboard, and keeps a live set of currently
    pressed keycodes. Because release is a real, distinct event (not
    inferred from a timeout), holding any combination of w/a/s/d/q/e keys
    at once gives an exact simultaneous multi-key state -- this is what
    makes the 3D vx/vy/wz vector combine correctly under simultaneous
    input. Requires read access to /dev/input/eventN (Linux `input` group).
    """

    _CODE_TO_CHAR = None  # filled lazily from evdev.ecodes

    def __init__(self):
        import evdev
        from evdev import ecodes
        self._evdev = evdev
        if _EvdevBackend._CODE_TO_CHAR is None:
            _EvdevBackend._CODE_TO_CHAR = {
                ecodes.KEY_W: "w", ecodes.KEY_S: "s", ecodes.KEY_A: "a",
                ecodes.KEY_D: "d", ecodes.KEY_Q: "q", ecodes.KEY_E: "e",
                ecodes.KEY_R: "r", ecodes.KEY_P: "p", ecodes.KEY_H: "h",
                ecodes.KEY_N: "n",
            }
        self._devices = []
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            caps = dev.capabilities().get(ecodes.EV_KEY, [])
            name_lower = dev.name.lower()
            if ecodes.KEY_A in caps and ecodes.KEY_W in caps and "mouse" not in name_lower:
                self._devices.append(dev)
        if not self._devices:
            for dev in list(self._devices):
                dev.close()
            raise RuntimeError("no readable keyboard-capable /dev/input device found")

    def device_names(self):
        return [dev.name for dev in self._devices]

    def read_events(self, pressed, fresh_presses, timeout_s=0.05):
        """Block up to timeout_s for events on any device; update `pressed`
        (a set of chars) in place and append freshly-pressed chars (value==1
        edges) to `fresh_presses`. Returns True if anything was read."""
        r, _, _ = select.select(self._devices, [], [], timeout_s)
        got = False
        for dev in r:
            for event in dev.read():
                if event.type != self._evdev.ecodes.EV_KEY:
                    continue
                got = True
                char = self._CODE_TO_CHAR.get(event.code)
                if char is None:
                    continue
                if event.value in (1, 2):  # press or repeat
                    if event.value == 1:
                        fresh_presses.append(char)
                    pressed.add(char)
                elif event.value == 0:  # release
                    pressed.discard(char)
        return got

    def close(self):
        for dev in self._devices:
            try:
                dev.close()
            except OSError:
                pass


class KeyboardCommand:
    """Real-time vx/vy/wz command state, combining held movement keys into
    one 3D vector every policy step (w/s -> vx, a/d -> wz, q/e -> vy; all
    three axes are independent and add up when keys are held together, e.g.
    w+a simultaneously gives vx=+limit AND wz=+limit, matching a real
    joystick's combined input rather than one-axis-at-a-time).

    Two backends (see module docstring for the full rationale):
      * _EvdevBackend (preferred): exact simultaneous multi-key state via
        raw press/release events from /dev/input/eventN.
      * termios stdin fallback: a background thread reads raw stdin bytes
        and records a per-key last-seen timestamp; update() treats any key
        not seen within key_timeout_s as released. Degraded for 3+
        simultaneous keys on some keyboards (N-key-rollover / terminal
        auto-repeat limits), but works everywhere including over SSH.

    Both backends update the same vx/vy/wz/reset_requested/pause_toggled
    state, so main() does not need to know which one is active.
    """

    def __init__(self, vx_limit, vy_limit, wz_limit, key_timeout_s=0.3):
        self.vx_limit, self.vy_limit, self.wz_limit = vx_limit, vy_limit, wz_limit
        self.key_timeout_s = key_timeout_s
        self.vx = self.vy = self.wz = 0.0
        self.reset_requested = False
        self.recover_requested = False
        self.next_waypoint_requested = False
        self.pause_toggled = False

        self._lock = threading.Lock()
        self._pressed = set()      # evdev backend: exact currently-held chars
        self._last_seen = {}       # termios backend: char -> monotonic timestamp
        self._discrete_last_seen = {}  # suppress terminal key auto-repeat for n/r/h/p
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

        self._backend = None       # "evdev" | "termios" | None
        self._evdev = None
        self._fd = sys.stdin.fileno()
        self._old_term_settings = None

    def start(self):
        try:
            self._evdev = _EvdevBackend()
        except Exception as exc:  # ImportError, PermissionError, RuntimeError, ...
            print(f"note: evdev keyboard backend unavailable ({exc}); "
                  "falling back to terminal stdin input "
                  "(hold-3-keys-at-once may undercount on some keyboards; "
                  "for exact multi-key input run: sudo usermod -aG input $USER "
                  "then log out/in)", flush=True)
        else:
            self._backend = "evdev"
            print(f"keyboard backend: evdev ({', '.join(self._evdev.device_names())})", flush=True)
            self._thread.start()
            return self

        try:
            self._old_term_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except (termios.error, ValueError):
            self._old_term_settings = None
            print("warning: stdin is not a TTY (redirected?); keyboard control disabled", flush=True)
            return self
        self._backend = "termios"
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._evdev is not None:
            self._evdev.close()
        if self._old_term_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term_settings)

    def _read_loop(self):
        if self._backend == "evdev":
            self._read_loop_evdev()
        elif self._backend == "termios":
            self._read_loop_termios()

    def _read_loop_evdev(self):
        while not self._stop.is_set():
            fresh = []
            try:
                got = self._evdev.read_events(self._pressed, fresh, timeout_s=0.05)
            except OSError:
                break
            if not got:
                continue
            with self._lock:
                for char in fresh:
                    if char == "r":
                        self.reset_requested = True
                    elif char == "h":
                        self.recover_requested = True
                    elif char == "n":
                        self.next_waypoint_requested = True
                    elif char == "p":
                        self.pause_toggled = True

    def _read_loop_termios(self):
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.05)
            if not ready:
                continue
            try:
                chunk = sys.stdin.read(1)
            except (OSError, ValueError):
                break
            if not chunk:
                continue
            k = chunk.lower()
            now = time.monotonic()
            if k in ("w", "s", "a", "d", "q", "e"):
                with self._lock:
                    self._last_seen[k] = now
            elif k in ("r", "h", "n", "p"):
                with self._lock:
                    previous = self._discrete_last_seen.get(k)
                    self._discrete_last_seen[k] = now
                if previous is not None and now - previous <= self.key_timeout_s:
                    continue
                if k == "r":
                    self.reset_requested = True
                elif k == "h":
                    self.recover_requested = True
                elif k == "n":
                    self.next_waypoint_requested = True
                else:
                    self.pause_toggled = True

    def update(self):
        """Call once per policy step: recompute vx/vy/wz from currently-held keys."""
        if self._backend == "evdev":
            with self._lock:
                held = set(self._pressed)
        elif self._backend == "termios":
            now = time.monotonic()
            with self._lock:
                held = {k for k, t in self._last_seen.items() if now - t <= self.key_timeout_s}
        else:
            held = set()
        vx = vy = wz = 0.0
        if "w" in held:
            vx += self.vx_limit
        if "s" in held:
            vx -= self.vx_limit
        if "a" in held:
            wz += self.wz_limit
        if "d" in held:
            wz -= self.wz_limit
        if "q" in held:
            vy += self.vy_limit
        if "e" in held:
            vy -= self.vy_limit
        self.vx, self.vy, self.wz = vx, vy, wz

    def as_array(self):
        return np.asarray([self.vx, self.vy, self.wz], np.float32)


def initialize(model, data, start):
    x, y, z, yaw = map(float, start)
    data.qpos[:3] = np.asarray([x, y, z])
    data.qpos[3:7] = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    data.qpos[7:23] = JOINT_INIT_RAW
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def waypoint_positions(model, data):
    rows = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("track_waypoint_"):
            token = name[len("track_waypoint_"):].split("_", 1)[0]
            if token.isdigit():
                rows.append((int(token), data.geom_xpos[geom_id].copy()))
    return sorted(rows)


def nearest_waypoint(base_pos, waypoints):
    if not waypoints:
        return -1
    return min(waypoints, key=lambda row: np.linalg.norm(row[1][:2] - base_pos[:2]))[0]


def print_status(sample, state, command, hit_fraction, paused):
    flag = "PAUSED" if paused else "running"
    print(
        f"\r[{flag}] step={sample:6d}  vx={command[0]:+.2f} vy={command[1]:+.2f} "
        f"wz={command[2]:+.2f}  base_z={state.base_pos_w[2]:.3f}  "
        f"privileged_hit={hit_fraction:.2f}   ",
        end="", flush=True)


def _add_debug_line(scene, start, end, color, width=2.0):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(color, np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(start, np.float64),
        np.asarray(end, np.float64),
    )
    scene.ngeom += 1


def _add_debug_loop(scene, points, color, width=2.0):
    points = np.asarray(points, np.float64)
    if len(points) < 2:
        return
    for index in range(len(points)):
        _add_debug_line(
            scene, points[index], points[(index + 1) % len(points)], color, width
        )


def draw_detector_debug(viewer, detection):
    """Draw only in the viewer overlay scene; physics collision is untouched."""
    scene = viewer.user_scn
    scene.ngeom = 0
    _add_debug_loop(scene, detection.corridor_corners_w, (0.0, 0.7, 1.0, 0.8))
    for ray in detection.clearance_rays:
        color = (1.0, 0.1, 0.1, 1.0) if ray.hit else (0.1, 0.8, 0.2, 0.55)
        _add_debug_line(scene, ray.start_w, ray.end_w, color, 2.0)
    if detection.edge_candidate:
        edge_color = (
            (1.0, 0.0, 0.0, 1.0)
            if detection.overhead_rejected
            else (1.0, 0.8, 0.0, 1.0)
        )
        _add_debug_line(
            scene,
            detection.edge_segment_w[0],
            detection.edge_segment_w[1],
            edge_color,
            5.0,
        )
        _add_debug_loop(
            scene,
            detection.landing_corners_w,
            (0.1, 1.0, 0.2, 1.0) if detection.has_target else edge_color,
            3.0,
        )


def draw_actor_height_debug(viewer, scanner, state, height):
    """Deprecated compatibility hook: height markers are no longer rendered."""


def wheel_contact_mask(model, data, wheel_body_ids):
    """Return per-wheel physical contact without depending on geom names."""
    wheel_by_body = {int(body_id): index for index, body_id in enumerate(wheel_body_ids)}
    result = np.zeros(4, dtype=bool)
    for contact in data.contact[: data.ncon]:
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            body_id = int(model.geom_bodyid[geom_id])
            wheel_index = wheel_by_body.get(body_id)
            if wheel_index is not None:
                result[wheel_index] = True
    return result


def draw_height_layers(viewer, raw, selection, actor_height, position, layer):
    """Deprecated compatibility hook; three-layer data remains in JSONL only."""


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher-onnx", type=Path)
    parser.add_argument(
        "--router-bundle", type=Path,
        help="hash-pinned NORMAL/HIGH/LOW/RECOVERY recurrent teacher bundle",
    )
    parser.add_argument("--fake-policy", action="store_true", help="Protocol smoke only; ignores keyboard-driven realism")
    parser.add_argument("--low-height-corridor-half-width", type=float, default=0.0,
                        help="NORMAL/LOW/HIGH: lateral corridor half width in metres (0=off, max 1.6); extend each boundary height outward")
    parser.add_argument("--low-height-x-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="NORMAL/LOW/HIGH X coordinate clamp in metres; default preserves [-0.8, 3.2]")
    parser.add_argument("--low-support-surface", action="store_true",
                        help="select the connected support layer for NORMAL/LOW/HIGH and Detector; keep physical headroom checks")
    parser.add_argument("--height-debug-layer", choices=("raw", "selected", "actor", "all"),
                        help="deprecated, ignored: height markers removed; use --height-debug-log for data")
    parser.add_argument("--height-debug-log", type=Path,
                        help="append three-layer JSONL at --height-debug-every cadence; off unless explicitly requested")
    parser.add_argument("--height-debug-every", type=int, default=100,
                        help="height JSONL sampling interval in policy steps (default 100 = 2 simulation seconds)")
    parser.add_argument("--actor-height-debug-vis", action="store_true",
                        help="deprecated, ignored: height markers removed")
    parser.add_argument("--viewer-hz", type=float, default=30., help="maximum viewer sync/overlay update rate; policy remains 50 Hz")
    parser.add_argument("--actor-threads", type=int, default=1, help="ONNX intra-op threads per Actor (default 1 avoids CPU oversubscription)")
    parser.add_argument(
        "--detector-debug-vis", action="store_true",
        help="draw detector corridor, candidate edge, landing patch and clearance rays",
    )
    parser.add_argument(
        "--xml", type=Path, required=True,
        help="MuJoCo scene XML; required so the selected map is explicit in the command",
    )
    parser.add_argument("--lidar-config", type=Path, default=DEFAULT_LIDAR)
    parser.add_argument("--heightmap-config", type=Path, default=DEFAULT_HEIGHTMAP)
    parser.add_argument("--start", nargs=4, type=float, metavar=("X", "Y", "Z", "YAW"), default=[0, -2.5, .2, 0])
    parser.add_argument("--terrain-id", default="official_track")
    parser.add_argument("--stop-base-z", type=float, default=.08, help="auto reset if base falls below this height")
    parser.add_argument(
        "--waypoint-base-clearance", type=float, default=.42,
        help="base height above a waypoint's walkable-surface Z after pressing n (default: 0.42 m)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vx-limit", type=float, default=1.0)
    parser.add_argument("--vy-limit", type=float, default=0.6)
    parser.add_argument("--wz-limit", type=float, default=1.0)
    parser.add_argument("--key-timeout", type=float, default=0.3,
                         help="seconds without a repeat keystroke before a held key is treated as released")
    parser.add_argument("--real-time", action="store_true", default=True,
                         help="throttle stepping to wall-clock (default on)")
    parser.add_argument("--no-real-time", dest="real_time", action="store_false",
                         help="run stepping as fast as possible (viewer will look sped up)")
    parser.add_argument("--record", type=Path, help="optional: also write a D_priv shard while playing")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=0,
                         help="automation/smoke-test only: exit after N policy steps instead of "
                              "waiting for the viewer window to close (0 = run until closed)")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if not np.isfinite(args.low_height_corridor_half_width) or not 0 <= args.low_height_corridor_half_width <= 1.6:
        parser.error("--low-height-corridor-half-width must be within [0, 1.6]")
    if args.low_height_corridor_half_width and args.router_bundle is None:
        parser.error("--low-height-corridor-half-width requires --router-bundle")
    if (args.low_support_surface or args.low_height_x_range) and args.router_bundle is None:
        parser.error("Low height processing requires --router-bundle")
    try:
        clamp_height_window(np.zeros((1, 41, 33), np.float32),
                            args.low_height_corridor_half_width, args.low_height_x_range)
    except ValueError as error:
        parser.error(str(error))
    policy_choices = sum(
        (bool(args.fake_policy), args.teacher_onnx is not None, args.router_bundle is not None)
    )
    if policy_choices != 1:
        parser.error("choose exactly one of --teacher-onnx, --router-bundle or --fake-policy")

    if (not np.isfinite(args.viewer_hz) or args.viewer_hz <= 0
            or args.actor_threads < 1 or args.height_debug_every < 1):
        parser.error("--viewer-hz must be finite and positive; --actor-threads/--height-debug-every must be positive")
    if args.actor_height_debug_vis or args.height_debug_layer:
        print("Height point visualization removed; legacy display flags are ignored. JSONL logging remains available.")
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model.opt.timestep = .001
    data = mujoco.MjData(model)
    if model.nu != 16:
        raise RuntimeError(f"S10 collector requires 16 actuators, MJCF has {model.nu}")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        raise RuntimeError("MJCF has no base_link")
    wheel_body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")
        ],
        dtype=np.int32,
    )
    if np.any(wheel_body_ids < 0):
        raise RuntimeError(f"MJCF is missing S10 wheel bodies: {wheel_body_ids.tolist()}")

    privileged = PrivilegedHeightScanner(model, body_exclude=base_id)
    router_runtime = (
        TeacherSkillRuntime(args.router_bundle, POLICY_DT_S,
                            low_height_corridor_half_width_m=args.low_height_corridor_half_width,
                            low_height_x_range=args.low_height_x_range,
                            actor_threads=args.actor_threads)
        if args.router_bundle is not None else None
    )
    detector = (
        S10ClimbDetector(model, base_id, router_runtime.detector_contract)
        if router_runtime is not None else None
    )
    policy = (
        ZeroPolicy()
        if args.fake_policy
        else TeacherPolicy(args.teacher_onnx, args.actor_threads) if args.teacher_onnx is not None else None
    )

    recorder = None
    lidar = None
    heightmap_cfg = None
    if args.record:
        rng = np.random.default_rng(args.seed)
        lidar_cfg = load_lidar_yaml(args.lidar_config)
        lidar = MujocoLidarScanner(model, lidar_cfg, rng=rng)
        heightmap_cfg = load_heightmap_config(args.heightmap_config)
        teacher_model = (
            None if args.fake_policy
            else args.teacher_onnx if args.teacher_onnx is not None
            else args.router_bundle
        )
        recorder = DPrivRecorder(
            teacher_model=teacher_model,
            teacher_source="fake_mujoco_smoke" if args.fake_policy else "privileged_mujoco_play",
            metadata={
                "xml": str(args.xml.resolve()), "sim_dt_s": .001, "policy_dt_s": POLICY_DT_S,
                "student_lidar_hz": lidar_cfg["rate_hz"], "teacher_grid": [41, 33],
                "teacher_height_formula": "base_z-hit_z-0.5 (ray origin base_z+1.2m)",
                "start_xyzyaw": list(map(float, args.start)), "terrain_id": args.terrain_id,
                "seed": args.seed, "labels_usable": not args.fake_policy,
                "source": "play_mujoco_teacher.py (interactive keyboard control)",
            })
    waypoints_cache = {"rows": None, "cursor": None}

    command_state = KeyboardCommand(args.vx_limit, args.vy_limit, args.wz_limit, args.key_timeout)

    def do_episode_reset():
        initialize(model, data, args.start)
        stand_up_z = run_stand_up(model, data, base_id, log=True)
        if stand_up_z < 0.30:
            print(f"warning: stand-up under height base_z={stand_up_z:.3f}, "
                  f"teacher may start OOD; check MJCF/actuators", flush=True)
        command_state.vx = command_state.vy = command_state.wz = 0.0
        if router_runtime is not None:
            router_runtime.reset()
        waypoints_cache["rows"] = waypoint_positions(model, data)
        waypoints_cache["cursor"] = None
        return np.zeros(16, np.float32)

    def do_inplace_recover():
        """Stand back up exactly where the robot is: keep x/y and the current
        yaw heading, flatten the base to level (pitch=roll=0) so the robot
        does not stand up tilted, reset joints to the seated JOINT_INIT_RAW
        (the start pose run_stand_up's spline interpolates from), then re-run
        the same StandUp PD to DEFAULT_ROBOT.  Unlike do_episode_reset this
        never teleports the base back to --start."""
        data.qpos[:3] = np.asarray([data.qpos[0], data.qpos[1], max(data.qpos[2], 0.05)])
        w, x, y, z = data.qpos[3:7]
        yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        data.qpos[3:7] = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        data.qpos[7:23] = JOINT_INIT_RAW
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        stand_up_z = run_stand_up(model, data, base_id, log=True)
        if stand_up_z < 0.30:
            print(f"warning: in-place stand-up under height base_z={stand_up_z:.3f}, "
                  f"teacher may start OOD; check MJCF/actuators", flush=True)
        command_state.vx = command_state.vy = command_state.wz = 0.0
        if router_runtime is not None:
            router_runtime.reset()
        return np.zeros(16, np.float32)

    def do_next_waypoint_teleport():
        """Teleport to the next ordered route marker in a clean standing state.

        On the first press, infer the current route point from the waypoint
        nearest to the robot. Further presses advance strictly in waypoint-ID
        order. The waypoint Z is the walking-surface height, so the floating
        base is placed above it by --waypoint-base-clearance. At the final
        point, keep the incoming heading; the following press wraps to point 0.
        """
        rows = waypoints_cache["rows"] or waypoint_positions(model, data)
        if not rows:
            raise RuntimeError(
                "n=next-waypoint requested, but the selected --xml has no "
                "track_waypoint_* geoms"
            )
        waypoints_cache["rows"] = rows

        cursor = waypoints_cache["cursor"]
        if cursor is None:
            nearest_id = nearest_waypoint(data.xpos[base_id], rows)
            current_row = next(
                index for index, (waypoint_id, _) in enumerate(rows)
                if waypoint_id == nearest_id
            )
            target_row = (current_row + 1) % len(rows)
        else:
            target_row = (cursor + 1) % len(rows)

        waypoint_id, target = rows[target_row]
        if target_row + 1 < len(rows):
            heading_delta = rows[target_row + 1][1][:2] - target[:2]
        elif target_row > 0:
            heading_delta = target[:2] - rows[target_row - 1][1][:2]
        else:
            heading_delta = np.asarray([1.0, 0.0])
        yaw = float(np.arctan2(heading_delta[1], heading_delta[0]))

        data.qpos[:3] = np.asarray(
            [target[0], target[1], target[2] + args.waypoint_base_clearance]
        )
        data.qpos[3:7] = np.asarray(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        )
        data.qpos[7:23] = DEFAULT_ROBOT
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

        command_state.vx = command_state.vy = command_state.wz = 0.0
        if router_runtime is not None:
            router_runtime.reset()
        waypoints_cache["cursor"] = target_row
        print(
            f"\nteleported to waypoint {waypoint_id}: "
            f"xyz=({target[0]:.3f}, {target[1]:.3f}, "
            f"{target[2] + args.waypoint_base_clearance:.3f}) "
            f"yaw={np.rad2deg(yaw):.1f}deg"
        )
        return np.zeros(16, np.float32)

    last_action = do_episode_reset()
    kp = np.asarray([80, 80, 80, 0] * 4, np.float32)
    kd = np.asarray([2, 2, 2, .6] * 4, np.float32)
    goal_pos, goal_vel = decode_action_raw(last_action)
    raw_pos, raw_vel = published_targets_to_raw(goal_pos, goal_vel)

    student_map = None
    if recorder is not None:
        state = state_from_mujoco(model, data, base_id)
        lidar_scan = lidar.scan(
            data, state.base_pos_w, state.base_rotation_w, apply_noise=True
        )
        student_map = build_heightmap(
            lidar_scan.points_w,
            state.base_pos_w,
            state.base_rotation_w,
            heightmap_cfg,
        )

    print("Controls (type into the TERMINAL, not the 3D viewer window):")
    print("  hold w/s=vx  a/d=turn  q/e=vy(strafe)   h=recover(stand in place)")
    print("  n=next waypoint   r=reset(to start)   p=pause   Ctrl+C=quit")
    print(f"xml={args.xml.resolve()}")
    if router_runtime is not None:
        print(f"router_bundle={router_runtime.bundle_path}")
        print(f"low_height_corridor_half_width_m={args.low_height_corridor_half_width} (0=off)")
        print(f"low_support_surface={args.low_support_surface} low_height_x_range={args.low_height_x_range}")
    print(f"start={args.start}  vx_limit={args.vx_limit}  wz_limit={args.wz_limit}  "
          f"real_time={args.real_time}  record={args.record}")

    sample = 0
    paused = False
    command_state.start()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            next_viewer_sync = 0.
            while viewer.is_running():
                wall_start = time.perf_counter()
                command_state.update()

                if command_state.reset_requested:
                    last_action = do_episode_reset()
                    goal_pos, goal_vel = decode_action_raw(last_action)
                    raw_pos, raw_vel = published_targets_to_raw(goal_pos, goal_vel)
                    if recorder is not None:
                        state = state_from_mujoco(model, data, base_id)
                        lidar_scan = lidar.scan(
                            data,
                            state.base_pos_w,
                            state.base_rotation_w,
                            apply_noise=True,
                        )
                        student_map = build_heightmap(
                            lidar_scan.points_w,
                            state.base_pos_w,
                            state.base_rotation_w,
                            heightmap_cfg,
                        )
                    command_state.reset_requested = False
                    sample = 0
                    viewer.sync()
                    continue

                if command_state.recover_requested:
                    last_action = do_inplace_recover()
                    goal_pos, goal_vel = decode_action_raw(last_action)
                    raw_pos, raw_vel = published_targets_to_raw(goal_pos, goal_vel)
                    if recorder is not None:
                        state = state_from_mujoco(model, data, base_id)
                        lidar_scan = lidar.scan(
                            data,
                            state.base_pos_w,
                            state.base_rotation_w,
                            apply_noise=True,
                        )
                        student_map = build_heightmap(
                            lidar_scan.points_w,
                            state.base_pos_w,
                            state.base_rotation_w,
                            heightmap_cfg,
                        )
                    command_state.recover_requested = False
                    viewer.sync()
                    continue

                if command_state.next_waypoint_requested:
                    last_action = do_next_waypoint_teleport()
                    goal_pos, goal_vel = decode_action_raw(last_action)
                    raw_pos, raw_vel = published_targets_to_raw(goal_pos, goal_vel)
                    if recorder is not None:
                        state = state_from_mujoco(model, data, base_id)
                        lidar_scan = lidar.scan(
                            data,
                            state.base_pos_w,
                            state.base_rotation_w,
                            apply_noise=True,
                        )
                        student_map = build_heightmap(
                            lidar_scan.points_w,
                            state.base_pos_w,
                            state.base_rotation_w,
                            heightmap_cfg,
                        )
                    command_state.next_waypoint_requested = False
                    viewer.sync()
                    continue

                if command_state.pause_toggled:
                    paused = not paused
                    command_state.pause_toggled = False

                if paused:
                    viewer.sync()
                    time.sleep(0.02)
                    continue

                command = np.clip(command_state.as_array(), [-1, -.6, -1], [1, .6, 1]).astype(np.float32)
                state = state_from_mujoco(model, data, base_id)

                if not args.fake_policy and sample > 0 and state.base_pos_w[2] < args.stop_base_z:
                    print(f"\nfell over: base_z={state.base_pos_w[2]:.3f} < {args.stop_base_z:.3f}; auto-resetting")
                    last_action = do_episode_reset()
                    goal_pos, goal_vel = decode_action_raw(last_action)
                    raw_pos, raw_vel = published_targets_to_raw(goal_pos, goal_vel)
                    sample = 0
                    viewer.sync()
                    continue

                proprio = assemble_official_57(state, command, last_action)
                geometry = privileged.scan_geometry(
                    data, state.base_pos_w, state.base_rotation_w
                )
                privileged_height, hit = geometry.height, geometry.hit
                selection = (select_support_surface(privileged, data, state.base_pos_w,
                                                    state.base_rotation_w, geometry,
                                                    query_x_range=args.low_height_x_range,
                                                    query_half_width=args.low_height_corridor_half_width)
                             if args.low_support_surface else SurfaceSelection(
                                 geometry, np.zeros(1353, bool), np.zeros(1353, bool),
                                 scan_xy_world(privileged, state.base_pos_w, state.base_rotation_w), 0.))
                detection = None
                if router_runtime is not None:
                    recurrent_inputs = assemble_asymmetric_teacher_inputs(
                        state, proprio, privileged_height
                    )
                    detection = detector.detect(
                        selection.scan,
                        data,
                        state.base_pos_w,
                        state.base_rotation_w,
                        command,
                    )
                    wheel_treads = wheel_tread_heights(privileged, data, data.xpos[wheel_body_ids])
                    action = router_runtime.step(
                        *recurrent_inputs,
                        detection,
                        state,
                        data.xpos[wheel_body_ids, 2],
                        wheel_contact_mask(model, data, wheel_body_ids),
                        low_height_map=selection.scan.height.reshape(33, 41).T[None],
                        wheel_support_z=wheel_treads,
                    )
                else:
                    teacher_obs = assemble_teacher_1413(
                        state, proprio, privileged_height
                    )
                    action = policy(teacher_obs)

                actual_actor_height = (router_runtime.last_actor_height_map if router_runtime is not None
                                       else privileged_height.reshape(33, 41).T[None])
                if args.height_debug_log and sample % args.height_debug_every == 0:
                    debug = height_debug_record(geometry, selection, actual_actor_height, state.base_pos_w)
                    debug.update(step=sample, sim_time_s=float(data.time),
                                 base_pos_w=state.base_pos_w.tolist(), command=command.tolist(),
                                 mode=router_runtime.router.mode.name if router_runtime else "SINGLE",
                                 low_support_surface=args.low_support_surface,
                                 low_height_x_range=args.low_height_x_range,
                                 low_height_half_width=args.low_height_corridor_half_width)
                    if router_runtime is not None:
                        debug["wheel_tread_z_w"] = [float(z) if np.isfinite(z) else None for z in wheel_treads]
                        paused = router_runtime.router.paused_climb_mode
                        debug["paused_climb_mode"] = paused.name if paused is not None else None
                        debug["straddle_steps"] = router_runtime.router.straddle_steps
                        debug["normal_straddle_steps"] = router_runtime.router.normal_straddle_steps
                        debug["last_entry_reason"] = router_runtime.router.last_entry_reason
                    args.height_debug_log.parent.mkdir(parents=True, exist_ok=True)
                    with args.height_debug_log.open("a") as stream:
                        stream.write(json.dumps(debug, allow_nan=False) + "\n")

                if recorder is not None:
                    if student_map is None:
                        raise RuntimeError("recording requires an initialized student map")
                    student_obs = np.concatenate([proprio, student_map.reshape(-1)]).astype(np.float32)
                    pose = np.concatenate([state.base_pos_w, data.xquat[base_id]]).astype(np.float32)
                    recorder.append(
                        student_obs=student_obs, teacher_action_raw=action, command_raw=command,
                        waypoint_id=nearest_waypoint(state.base_pos_w, waypoints_cache["rows"]),
                        terrain_id=args.terrain_id, episode_id=0, step_id=sample, base_pose_wxyz=pose,
                        privileged_hit_fraction=float(hit.mean()))

                last_action = action
                goal_pos, goal_vel = decode_action_raw(action)
                raw_pos, raw_vel = published_targets_to_raw(goal_pos, goal_vel)
                for _ in range(SUBSTEPS_PER_POLICY_STEP):
                    q, dq = data.qpos[7:23], data.qvel[6:22]
                    data.ctrl[:] = kp * (raw_pos - q) + kd * (raw_vel - dq)
                    mujoco.mj_step(model, data)
                now = time.perf_counter()
                if now >= next_viewer_sync:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        if args.detector_debug_vis and detection is not None:
                            draw_detector_debug(viewer, detection)
                    viewer.sync()
                    next_viewer_sync = time.perf_counter() + 1. / args.viewer_hz

                if args.log_every and sample % args.log_every == 0:
                    if router_runtime is None:
                        print_status(sample, state, command, float(hit.mean()), paused)
                    else:
                        print(
                            f"\r[running] step={sample:6d} vx={command[0]:+.2f} "
                            f"vy={command[1]:+.2f} wz={command[2]:+.2f} "
                            f"mode={router_runtime.router.mode.name} "
                            f"entry={router_runtime.router.last_entry_reason or '-'} "
                            f"detector={int(detection.has_target)} "
                            f"overhead={int(detection.overhead_rejected)} "
                            f"headroom={int(detection.upper_clearance_blocked)} "
                            f"rise={detection.height_m:.4f}m "
                            f"base_z={state.base_pos_w[2]:.3f} ",
                            end="", flush=True,
                        )

                if recorder is not None:
                    lidar_state = state_from_mujoco(model, data, base_id)
                    lidar_scan = lidar.scan(
                        data,
                        lidar_state.base_pos_w,
                        lidar_state.base_rotation_w,
                        apply_noise=True,
                    )
                    if len(lidar_scan.points_w):
                        student_map = build_heightmap(
                            lidar_scan.points_w,
                            lidar_state.base_pos_w,
                            lidar_state.base_rotation_w,
                            heightmap_cfg,
                        )

                sample += 1
                if args.real_time:
                    elapsed = time.perf_counter() - wall_start
                    remaining = POLICY_DT_S - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
                if args.max_steps and sample >= args.max_steps:
                    print(f"\nreached --max-steps={args.max_steps}, exiting")
                    break
    finally:
        command_state.stop()

    print()
    if recorder is not None:
        output = recorder.write(args.record)
        count = validate_d_priv(output)
        summary = {"output": str(output.resolve()), "samples": count}
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
