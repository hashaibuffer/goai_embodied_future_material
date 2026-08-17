/**
 * @file autonav_interface.hpp
 * @brief Autonomous waypoint navigation replacing the keyboard interface.
 *
 * Subscribes `/S10_BASE_POSE` (geometry_msgs/msg/PoseStamped), parses
 * `track_overlay.xml` waypoints, performs state-gated stand-up + RL control
 * entry, and writes the official UserCommand velocity scales. Two modes:
 *   - eval:    never teleports, official timer is valid.
 *   - collect: on stall/tumble/out-of-bounds writes failure metadata and
 *              teleports to the current checkpoint to continue coverage.
 *
 * The control law, waypoint algorithm and stall detection mirror
 * s10_waypoint_navigation/navigation.py exactly.
 *
 * @author DeepRobotics
 * @date 2026-08-17
 * @copyright Copyright (c) 2025  DeepRobotics
 */

#pragma once

#include "user_command_interface.h"
#include "custom_types.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"

using namespace interface;
using namespace types;

class AutoNavCommandInterface : public UserCommandInterface
{
public:
    enum class Mode { EVAL, COLLECT };

    static Mode mode_from_string(const std::string& s)
    {
        if (s == "collect") return Mode::COLLECT;
        return Mode::EVAL;
    }

private:
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kStandHeight = 0.2;      // base_link nominal standing z
    static constexpr double kControlDtMs = 5.0;      // same cadence as keyboard
    static constexpr double kMinStandWaitS = 4.0;    // 2 * stand_duration_ + buffer

    // Command limits (frozen by T00).
    static constexpr float kMaxForward = 1.0f;
    static constexpr float kMaxSide = 0.6f;
    static constexpr float kMaxYaw = 1.0f;

    // Navigation parameters (mirror track.yaml non-frozen sections).
    static constexpr double kReachRadiusM = 0.20;
    static constexpr double kLookaheadStraightM = 1.60;
    static constexpr double kLookaheadTurnM = 0.80;
    static constexpr double kLookaheadMazeM = 0.45;
    static constexpr double kLookaheadWeight = 0.35;
    static constexpr double kYawGain = 2.0;
    static constexpr double kYawThresholdRad = 0.35;
    static constexpr double kTurnAngleThresholdRad = 0.785;  // ~45 deg
    static constexpr double kNominalVx = 0.70;
    static constexpr double kMazeVx = 0.35;

    // Stall detection parameters.
    static constexpr double kStallWindowS = 5.0;
    static constexpr double kStallMinDisplacementM = 0.30;
    static constexpr double kTumbleRollPitchRad = 1.047;  // ~60 deg
    static constexpr double kOutOfBoundsMarginM = 2.0;

    struct Waypoint
    {
        double x, y, z;
    };

    struct PoseSample
    {
        double t, x, y;
    };

    std::atomic<bool> running_{false};
    std::thread spin_thread_;
    std::thread nav_thread_;

    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr teleport_pub_;

    // Latest pose (thread-safe, updated by subscription callback).
    mutable std::mutex pose_mutex_;
    bool has_pose_ = false;
    double pose_x_ = 0.0, pose_y_ = 0.0, pose_z_ = 0.0;
    double pose_qx_ = 0.0, pose_qy_ = 0.0, pose_qz_ = 0.0, pose_qw_ = 1.0;
    double pose_stamp_s_ = 0.0;

    Mode mode_ = Mode::EVAL;
    std::string results_dir_ = "results";

    std::vector<Waypoint> waypoints_;
    int next_idx_ = 0;
    bool reached_any_ = false;
    int max_idx_ = 0;  // highest waypoint index reached (for baseline)

    // Stand-up state gating.
    bool stand_requested_ = false;
    bool rl_requested_ = false;
    bool standing_since_set_ = false;
    std::chrono::steady_clock::time_point standing_since_;

    // Stall history + logging throttling.
    std::deque<PoseSample> stall_history_;
    std::chrono::steady_clock::time_point last_no_pose_log_;
    std::chrono::steady_clock::time_point last_cmd_log_;
    bool eval_failure_logged_ = false;      // eval: log stall once per waypoint
    std::chrono::steady_clock::time_point teleport_cooldown_until_;  // collect: post-teleport grace

    double min_x_ = 0.0, max_x_ = 0.0, min_y_ = 0.0, max_y_ = 0.0;

    static double wrap_angle(double a)
    {
        a = std::fmod(a + kPi, 2.0 * kPi);
        if (a < 0) a += 2.0 * kPi;
        return a - kPi;
    }

    static float clip(float v, float lo, float hi)
    {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    void quat_to_rpy(double qx, double qy, double qz, double qw,
                     double& roll, double& pitch, double& yaw)
    {
        double sinr_cosp = 2.0 * (qw * qx + qy * qz);
        double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
        roll = std::atan2(sinr_cosp, cosr_cosp);

        double sinp = 2.0 * (qw * qy - qz * qx);
        if (std::fabs(sinp) >= 1.0) {
            pitch = std::copysign(kPi / 2.0, sinp);
        } else {
            pitch = std::asin(sinp);
        }

        double siny_cosp = 2.0 * (qw * qz + qx * qy);
        double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        yaw = std::atan2(siny_cosp, cosy_cosp);
    }

    void yaw_to_quat(double yaw, double& qx, double& qy, double& qz, double& qw)
    {
        qx = 0.0;
        qy = 0.0;
        qz = std::sin(yaw / 2.0);
        qw = std::cos(yaw / 2.0);
    }

    std::filesystem::path resolve_track_xml()
    {
        namespace fs = std::filesystem;
        const char* env_path = std::getenv("S10_TRACK_OVERLAY");
        if (env_path && env_path[0] != '\0') {
            return fs::path(env_path);
        }
        fs::path base = fs::path(__FILE__).parent_path();
        return (base / ".." / ".." / "S10_description" /
                "s10_mjcf" / "mjcf" / "track_overlay.xml").lexically_normal();
    }

    bool parse_waypoints(const std::filesystem::path& xml_path)
    {
        std::ifstream in(xml_path);
        if (!in.is_open()) {
            std::cerr << "[AutoNav] cannot open track XML: " << xml_path << std::endl;
            return false;
        }

        std::map<int, Waypoint> found;
        std::string line;
        while (std::getline(in, line)) {
            size_t p = line.find("track_waypoint_");
            if (p == std::string::npos) continue;
            p += std::strlen("track_waypoint_");
            size_t digit_end = line.find_first_not_of("0123456789", p);
            if (digit_end == std::string::npos || digit_end == p) continue;
            int id = std::stoi(line.substr(p, digit_end - p));

            size_t pos_at = line.find("pos=\"", digit_end);
            if (pos_at == std::string::npos) continue;
            size_t pos_start = pos_at + std::strlen("pos=\"");
            size_t pos_end = line.find('"', pos_start);
            if (pos_end == std::string::npos) continue;

            Waypoint w{};
            if (std::sscanf(line.substr(pos_start, pos_end - pos_start).c_str(),
                            "%lf %lf %lf", &w.x, &w.y, &w.z) != 3)
                continue;
            found[id] = w;
        }

        if (found.empty()) {
            std::cerr << "[AutoNav] no waypoints parsed from " << xml_path << std::endl;
            return false;
        }

        int max_id = found.rbegin()->first;
        for (int i = 0; i <= max_id; ++i) {
            auto it = found.find(i);
            if (it == found.end()) {
                std::cerr << "[AutoNav] missing waypoint id " << i << std::endl;
                return false;
            }
            waypoints_.push_back(it->second);
        }

        // Bounding box for out-of-bounds detection.
        min_x_ = max_x_ = waypoints_.front().x;
        min_y_ = max_y_ = waypoints_.front().y;
        for (const auto& w : waypoints_) {
            min_x_ = std::min(min_x_, w.x); max_x_ = std::max(max_x_, w.x);
            min_y_ = std::min(min_y_, w.y); max_y_ = std::max(max_y_, w.y);
        }
        return true;
    }

    void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        pose_x_ = msg->pose.position.x;
        pose_y_ = msg->pose.position.y;
        pose_z_ = msg->pose.position.z;
        pose_qx_ = msg->pose.orientation.x;
        pose_qy_ = msg->pose.orientation.y;
        pose_qz_ = msg->pose.orientation.z;
        pose_qw_ = msg->pose.orientation.w;
        pose_stamp_s_ = static_cast<double>(msg->header.stamp.sec) +
                        static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
        has_pose_ = true;
    }

    void snapshot_pose(double& x, double& y, double& z,
                       double& qx, double& qy, double& qz, double& qw,
                       double& t_s, bool& ok)
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        ok = has_pose_;
        x = pose_x_; y = pose_y_; z = pose_z_;
        qx = pose_qx_; qy = pose_qy_; qz = pose_qz_; qw = pose_qw_;
        t_s = pose_stamp_s_;
    }

    static double xy_dist(double ax, double ay, double bx, double by)
    {
        return std::hypot(ax - bx, ay - by);
    }

    double segment_bend_rad(int idx) const
    {
        if (idx <= 0 || idx >= static_cast<int>(waypoints_.size()) - 1) return 0.0;
        const Waypoint& a = waypoints_[idx - 1];
        const Waypoint& b = waypoints_[idx];
        const Waypoint& c = waypoints_[idx + 1];
        double v1x = b.x - a.x, v1y = b.y - a.y;
        double v2x = c.x - b.x, v2y = c.y - b.y;
        double n1 = std::hypot(v1x, v1y), n2 = std::hypot(v2x, v2y);
        if (n1 < 1e-9 || n2 < 1e-9) return 0.0;
        double cos_a = (v1x * v2x + v1y * v2y) / (n1 * n2);
        cos_a = std::max(-1.0, std::min(1.0, cos_a));
        return std::acos(cos_a);
    }

    bool is_maze_id(int id) const
    {
        return id >= 28 && id <= 32;
    }

    std::string segment_kind(int idx) const
    {
        if (is_maze_id(idx)) return "maze";
        return "straight";
    }

    double lookahead_switch(const std::string& kind) const
    {
        if (kind == "maze") return kLookaheadMazeM;
        if (kind == "turn") return kLookaheadTurnM;
        return kLookaheadStraightM;
    }

    double target_vx(const std::string& kind) const
    {
        if (kind == "maze") return kMazeVx;
        return kNominalVx;
    }

    // Mirrors navigation.py compute_command.
    void compute_command(double x, double y, double yaw,
                         float& fwd, float& side, float& yaw_cmd)
    {
        fwd = 0.0f; side = 0.0f; yaw_cmd = 0.0f;
        if (next_idx_ < 0 || next_idx_ >= static_cast<int>(waypoints_.size())) return;

        const Waypoint& wp = waypoints_[next_idx_];
        double tx = wp.x, ty = wp.y;

        std::string kind = segment_kind(next_idx_);
        if (next_idx_ != 0 && next_idx_ != static_cast<int>(waypoints_.size()) - 1) {
            double bend = segment_bend_rad(next_idx_);
            if (!is_maze_id(next_idx_) && bend > kTurnAngleThresholdRad)
                kind = "turn";
        }

        bool no_cross = (next_idx_ == 0 || next_idx_ == static_cast<int>(waypoints_.size()) - 1);
        if (!no_cross && next_idx_ + 1 < static_cast<int>(waypoints_.size())) {
            double sw = lookahead_switch(kind);
            if (xy_dist(x, y, tx, ty) < sw) {
                const Waypoint& nxt = waypoints_[next_idx_ + 1];
                tx = (1.0 - kLookaheadWeight) * tx + kLookaheadWeight * nxt.x;
                ty = (1.0 - kLookaheadWeight) * ty + kLookaheadWeight * nxt.y;
            }
        }

        double angle_to_target = std::atan2(ty - y, tx - x);
        double heading_error = wrap_angle(angle_to_target - yaw);

        if (std::fabs(heading_error) > kYawThresholdRad) {
            yaw_cmd = clip(static_cast<float>(kYawGain * heading_error), -kMaxYaw, kMaxYaw);
        } else {
            fwd = clip(static_cast<float>(target_vx(kind)), -kMaxForward, kMaxForward);
            yaw_cmd = clip(static_cast<float>(kYawGain * heading_error), -kMaxYaw, kMaxYaw);
        }
    }

    void advance_waypoint(double x, double y)
    {
        bool advanced = false;
        while (next_idx_ < static_cast<int>(waypoints_.size()) &&
               xy_dist(x, y, waypoints_[next_idx_].x, waypoints_[next_idx_].y) < kReachRadiusM) {
            if (!reached_any_) {
                reached_any_ = true;
                std::cout << "[AutoNav] first waypoint (timer start) reached, idx="
                          << next_idx_ << std::endl;
            }
            max_idx_ = std::max(max_idx_, next_idx_);
            ++next_idx_;
            advanced = true;
        }
        if (advanced) {
            // New segment: allow one fresh failure log for this waypoint.
            eval_failure_logged_ = false;
            stall_history_.clear();
        }
    }

    bool detect_stall(double& displacement)
    {
        displacement = 0.0;
        if (stall_history_.size() < 2) return false;
        double latest_t = stall_history_.back().t;
        double window_start = latest_t - kStallWindowS;

        // Drop samples older than the window.
        while (!stall_history_.empty() && stall_history_.front().t < window_start)
            stall_history_.pop_front();

        if (stall_history_.size() < 2) return false;
        if (stall_history_.back().t - stall_history_.front().t + 1e-9 < kStallWindowS) {
            return false;
        }
        displacement = xy_dist(stall_history_.front().x, stall_history_.front().y,
                               stall_history_.back().x, stall_history_.back().y);
        return displacement < kStallMinDisplacementM;
    }

    bool detect_tumble(double roll, double pitch) const
    {
        return std::fabs(roll) > kTumbleRollPitchRad || std::fabs(pitch) > kTumbleRollPitchRad;
    }

    bool detect_out_of_bounds(double x, double y) const
    {
        return x < min_x_ - kOutOfBoundsMarginM || x > max_x_ + kOutOfBoundsMarginM ||
               y < min_y_ - kOutOfBoundsMarginM || y > max_y_ + kOutOfBoundsMarginM;
    }

    void ensure_results_dir()
    {
        std::filesystem::create_directories(results_dir_);
    }

    void append_failure(const std::string& reason, double x, double y, double yaw,
                        bool teleported, double tele_x, double tele_y,
                        double stamp_s, float fwd, float side, float wz)
    {
        ensure_results_dir();
        nlohmann::json rec;
        rec["wp_id"] = (next_idx_ > 0 ? next_idx_ - 1 : 0);
        rec["progress_next_idx"] = next_idx_;
        rec["pose"] = {{"x", x}, {"y", y}, {"yaw", yaw}};
        rec["cmd"] = {{"vx", fwd}, {"vy", side}, {"wz", wz}};
        rec["t_fail"] = stamp_s;
        rec["mode"] = (mode_ == Mode::COLLECT ? "collect" : "eval");
        rec["reason"] = reason;
        rec["teleported"] = teleported;
        rec["teleport_target"] = {{"x", tele_x}, {"y", tele_y}};

        std::ofstream out(results_dir_ + "/failures.jsonl", std::ios::app);
        out << rec.dump() << "\n";
        out.close();

        std::ofstream seg(results_dir_ + "/fail_segments.md", std::ios::app);
        seg << "wp~=" << (next_idx_ > 0 ? next_idx_ - 1 : 0)
            << ", " << reason << "\n";
        seg.close();

        std::cout << "[AutoNav] failure recorded: " << reason
                  << " at wp~=" << (next_idx_ > 0 ? next_idx_ - 1 : 0) << std::endl;
    }

    void request_teleport(double tx, double ty, double tz, double yaw)
    {
        if (!teleport_pub_) return;
        geometry_msgs::msg::PoseStamped msg;
        msg.header.frame_id = "base_link";
        msg.pose.position.x = tx;
        msg.pose.position.y = ty;
        msg.pose.position.z = tz + kStandHeight;
        yaw_to_quat(yaw, msg.pose.orientation.x, msg.pose.orientation.y,
                    msg.pose.orientation.z, msg.pose.orientation.w);
        teleport_pub_->publish(msg);
        std::cout << "[AutoNav] teleport request -> (" << tx << ", " << ty << ")" << std::endl;
    }

    void nav_loop()
    {
        last_no_pose_log_ = std::chrono::steady_clock::now();
        last_cmd_log_ = std::chrono::steady_clock::now();

        while (running_) {
            auto now = std::chrono::steady_clock::now();

            // ---- State-gated stand up -> RL control (no sleep-only gating). ----
            uint8_t state = msfb_->GetCurrentState();
            if (!rl_requested_) {
                if (!stand_requested_ && (state == RobotMotionState::WaitingForStand ||
                                          state == RobotMotionState::LieDown)) {
                    usr_cmd_->target_mode = uint8_t(RobotMotionState::StandingUp);
                    stand_requested_ = true;
                    std::cout << "[AutoNav] requesting StandingUp" << std::endl;
                } else if (state == RobotMotionState::StandingUp) {
                    if (!standing_since_set_) {
                        standing_since_ = now;
                        standing_since_set_ = true;
                    }
                    double elapsed =
                        std::chrono::duration<double>(now - standing_since_).count();
                    if (elapsed >= kMinStandWaitS) {
                        usr_cmd_->target_mode = uint8_t(RobotMotionState::RLControlMode);
                        rl_requested_ = true;
                        std::cout << "[AutoNav] requesting RLControlMode" << std::endl;
                    }
                } else {
                    standing_since_set_ = false;
                }
            }

            // ---- Pose / navigation (only drive when in RL control). ----
            double x, y, z, qx, qy, qz, qw, t_s;
            bool ok;
            snapshot_pose(x, y, z, qx, qy, qz, qw, t_s, ok);

            if (state == RobotMotionState::RLControlMode) {
                if (!ok) {
                    usr_cmd_->forward_vel_scale = 0.0f;
                    usr_cmd_->side_vel_scale = 0.0f;
                    usr_cmd_->turnning_vel_scale = 0.0f;
                    if (std::chrono::duration<double>(now - last_no_pose_log_).count() > 1.0) {
                        std::cerr << "[AutoNav] no pose received; zeroing command" << std::endl;
                        last_no_pose_log_ = now;
                    }
                } else {
                    double roll, pitch, yaw;
                    quat_to_rpy(qx, qy, qz, qw, roll, pitch, yaw);

                    advance_waypoint(x, y);

                    float fwd, side, wz;
                    compute_command(x, y, yaw, fwd, side, wz);
                    usr_cmd_->forward_vel_scale = fwd;
                    usr_cmd_->side_vel_scale = side;
                    usr_cmd_->turnning_vel_scale = wz;

                    // ---- Failure detection (collect mode reacts). ----
                    stall_history_.push_back({t_s, x, y});

                    bool pure_yaw = (fwd == 0.0f && wz != 0.0f);
                    bool in_cooldown =
                        std::chrono::steady_clock::now() < teleport_cooldown_until_;

                    double disp = 0.0;
                    // Don't flag stall before the first waypoint is reached:
                    // startup re-orientation legitimately stays put.
                    bool stalled = reached_any_ && !pure_yaw && detect_stall(disp);
                    bool tumble = detect_tumble(roll, pitch);
                    bool oob = detect_out_of_bounds(x, y);

                    if ((stalled || tumble || oob) && !in_cooldown) {
                        std::string reason = tumble ? "tumble" : (oob ? "out_of_bounds" : "stall");
                        const bool finished =
                            next_idx_ < 0 || next_idx_ >= static_cast<int>(waypoints_.size());
                        if (mode_ == Mode::COLLECT) {
                            if (finished) {
                                append_failure(reason, x, y, yaw, false, 0.0, 0.0,
                                               t_s, fwd, side, wz);
                                usr_cmd_->forward_vel_scale = 0.0f;
                                usr_cmd_->side_vel_scale = 0.0f;
                                usr_cmd_->turnning_vel_scale = 0.0f;
                            } else {
                                const Waypoint& dest = waypoints_[next_idx_];
                                append_failure(reason, x, y, yaw, true, dest.x, dest.y,
                                               t_s, fwd, side, wz);
                                double target_yaw = yaw;
                                if (next_idx_ + 1 < static_cast<int>(waypoints_.size())) {
                                    const Waypoint& n = waypoints_[next_idx_ + 1];
                                    target_yaw = std::atan2(n.y - dest.y, n.x - dest.x);
                                }
                                request_teleport(dest.x, dest.y, dest.z, target_yaw);
                                stall_history_.clear();
                                teleport_cooldown_until_ =
                                    std::chrono::steady_clock::now() +
                                    std::chrono::seconds(static_cast<int>(kStallWindowS));
                            }
                        } else if (!eval_failure_logged_) {
                            append_failure(reason, x, y, yaw, false, 0.0, 0.0,
                                           t_s, fwd, side, wz);
                            eval_failure_logged_ = true;
                        }
                    }

                    if (std::chrono::duration<double>(now - last_cmd_log_).count() > 5.0) {
                        std::cout << "[AutoNav] idx=" << next_idx_ << "/" << waypoints_.size()
                                  << " cmd=(" << fwd << ", " << side << ", " << wz << ")"
                                  << std::endl;
                        last_cmd_log_ = now;
                    }
                }
            } else {
                // Not yet in RL control: keep zero velocity.
                usr_cmd_->forward_vel_scale = 0.0f;
                usr_cmd_->side_vel_scale = 0.0f;
                usr_cmd_->turnning_vel_scale = 0.0f;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(kControlDtMs)));
        }
    }

public:
    AutoNavCommandInterface(RobotName robot_name, Mode mode = Mode::EVAL)
        : UserCommandInterface(robot_name), mode_(mode)
    {
        const char* env_dir = std::getenv("S10_RESULTS_DIR");
        if (env_dir && env_dir[0] != '\0') results_dir_ = env_dir;

        std::memset(usr_cmd_, 0, sizeof(UserCommand));
        std::cout << "[AutoNavCommandInterface] initialized, mode="
                  << (mode_ == Mode::COLLECT ? "collect" : "eval") << std::endl;
    }

    ~AutoNavCommandInterface() { Stop(); }

    void Start() override
    {
        if (running_) return;

        std::filesystem::path xml = resolve_track_xml();
        if (!std::filesystem::exists(xml)) {
            std::cerr << "[AutoNav] track XML missing at " << xml
                      << "; disabling navigation" << std::endl;
            return;
        }
        if (!parse_waypoints(xml)) {
            std::cerr << "[AutoNav] failed to parse waypoints; disabling navigation" << std::endl;
            return;
        }
        std::cout << "[AutoNav] parsed " << waypoints_.size() << " waypoints" << std::endl;

        node_ = std::make_shared<rclcpp::Node>("autonav");
        pose_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/S10_BASE_POSE", 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { pose_callback(msg); });
        teleport_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/S10_TELEPORT", 10);

        running_ = true;

        spin_thread_ = std::thread([this]() {
            rclcpp::executors::SingleThreadedExecutor exec;
            exec.add_node(node_);
            exec.spin();
        });
        nav_thread_ = std::thread(&AutoNavCommandInterface::nav_loop, this);

        std::cout << "[AutoNav] started" << std::endl;
    }

    void Stop() override
    {
        running_ = false;
        if (node_) {
            rclcpp::shutdown();
        }
        if (spin_thread_.joinable()) spin_thread_.join();
        if (nav_thread_.joinable()) nav_thread_.join();

        usr_cmd_->forward_vel_scale = 0.0f;
        usr_cmd_->side_vel_scale = 0.0f;
        usr_cmd_->turnning_vel_scale = 0.0f;
        std::cout << "[AutoNav] stopped, max waypoint idx=" << max_idx_ << std::endl;
    }

    UserCommand* GetUserCommand() override { return usr_cmd_; }
};
