#pragma once

#include "policy_runner_base.hpp"
#include "terrain_command_math.hpp"
#include "terrain_policy_math.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

#include <drdds/msg/teacher_sample.hpp>
#include <onnxruntime_cxx_api.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

// TD training-only runner. The official 57->16 policy receives cmd_terrain;
// the recorded 441-D student observation receives cmd_raw + the same LiDAR
// heightmap. One ONNX result is both published as the label and decoded into
// the RobotAction that drives the simulator.
class TeacherCollectRunner final : public PolicyRunnerBase {
public:
    TeacherCollectRunner(
        const std::string& model_path,
        const rclcpp::Node::SharedPtr& node,
        const std::string& heightmap_topic = "/S10_HEIGHTMAP")
        : PolicyRunnerBase("teacher_collect"),
          model_path_(model_path),
          node_(node),
          env_(ORT_LOGGING_LEVEL_WARNING, "TeacherCollectRunner"),
          session_options_(),
          session_(nullptr),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        SetDecimation(TerrainPolicyMath::kDecimation);
        session_options_.SetIntraOpNumThreads(4);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
        ValidateModel();
        InitializeRobotAction();
        const char* rewrite_mode = std::getenv("S10_TD_REWRITE_MODE");
        rewrite_enabled_ = rewrite_mode == nullptr || std::string(rewrite_mode) != "off";
        heightmap_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
            heightmap_topic, rclcpp::QoS(1).best_effort(),
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                std::array<float, TerrainPolicyMath::kHeightmapDim> candidate{};
                if (msg->data.size() != candidate.size()) return;
                for (std::size_t i = 0; i < candidate.size(); ++i) {
                    if (!std::isfinite(msg->data[i])) return;
                    candidate[i] = msg->data[i];
                }
                std::lock_guard<std::mutex> lock(heightmap_mutex_);
                heightmap_ = candidate;
                heightmap_received_at_ = std::chrono::steady_clock::now();
                has_heightmap_ = true;
            });
        sample_pub_ = node_->create_publisher<drdds::msg::TeacherSample>(
            "/S10_TD_SAMPLE", rclcpp::QoS(20).reliable());
    }

    void DisplayPolicyInfo() override {
        std::cout << "TeacherCollectRunner official_model=" << model_path_
                  << " teacher=[1,57]->[1,16] student=[1,441]"
                  << " terrain_rewrite=" << (rewrite_enabled_ ? "on" : "off") << std::endl;
    }

    void OnEnter() override {
        run_cnt_ = 0;
        sequence_ = 0;
        last_action_.fill(0.0F);
        previous_terrain_command_.fill(0.0F);
    }

    RobotAction getRobotAction(const RobotBasicState& robot, const UserCommand& command) override {
        const std::array<float, 3> raw{
            command.forward_vel_scale,
            command.side_vel_scale,
            command.turnning_vel_scale};
        std::array<float, TerrainPolicyMath::kHeightmapDim> heightmap{};
        bool heightmap_valid = false;
        float heightmap_age_ms = -1.0F;
        {
            std::lock_guard<std::mutex> lock(heightmap_mutex_);
            if (has_heightmap_) {
                heightmap_age_ms = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - heightmap_received_at_).count();
                heightmap_valid = heightmap_age_ms <= 250.0F;
                if (heightmap_valid) heightmap = heightmap_;
            }
        }

        auto rewrite = TerrainCommandMath::Rewrite(raw, heightmap, previous_terrain_command_);
        if (!rewrite_enabled_) rewrite.command = raw;
        previous_terrain_command_ = rewrite.command;

        // Both observations are assembled before last_action_ is updated. This
        // prevents future-action leakage into the student sample.
        std::array<float, TerrainPolicyMath::kObservationDim> teacher_full{};
        std::array<float, TerrainPolicyMath::kObservationDim> student{};
        if (!Assemble(robot, rewrite.command, {}, teacher_full) ||
            !Assemble(robot, raw, heightmap, student)) {
            last_action_.fill(0.0F);
            Decode(last_action_);
            return robot_action_;
        }
        std::array<float, TerrainPolicyMath::kProprioDim> teacher{};
        std::copy_n(teacher_full.begin(), teacher.size(), teacher.begin());
        auto action = Infer(teacher);
        if (!std::all_of(action.begin(), action.end(), [](float v) { return std::isfinite(v); })) {
            action.fill(0.0F);
        }

        Publish(student, teacher, heightmap, raw, rewrite.command, action,
                rewrite.risk, heightmap_valid, heightmap_age_ms);
        last_action_ = action;
        Decode(action);
        ++run_cnt_;
        return robot_action_;
    }

private:
    void ValidateModel() {
        if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
            throw std::runtime_error("official teacher must have one input and output");
        const auto input = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const auto output = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input.size() != 2 || input.back() != 57 || output.size() != 2 || output.back() != 16)
            throw std::runtime_error("official teacher signature must be [1,57] -> [1,16]");
    }

    bool Assemble(
        const RobotBasicState& robot,
        const std::array<float, 3>& command,
        const std::array<float, TerrainPolicyMath::kHeightmapDim>& heightmap,
        std::array<float, TerrainPolicyMath::kObservationDim>& observation) {
        std::array<float, 3> omega{};
        std::array<float, 9> rotation{};
        std::array<float, TerrainPolicyMath::kActionDim> position{}, velocity{};
        for (std::size_t i = 0; i < 3; ++i) omega[i] = robot.base_omega(i);
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t col = 0; col < 3; ++col)
                rotation[row * 3 + col] = robot.base_rot_mat(row, col);
        for (std::size_t i = 0; i < TerrainPolicyMath::kActionDim; ++i) {
            position[i] = robot.joint_pos(i);
            velocity[i] = robot.joint_vel(i);
        }
        return TerrainPolicyMath::AssembleObservation(
            omega, rotation, command, position, velocity,
            last_action_, heightmap, observation);
    }

    std::array<float, TerrainPolicyMath::kActionDim> Infer(
        std::array<float, TerrainPolicyMath::kProprioDim>& observation) {
        constexpr std::array<int64_t, 2> shape{1, 57};
        auto tensor = Ort::Value::CreateTensor<float>(
            memory_info_, observation.data(), observation.size(), shape.data(), shape.size());
        const char* input_names[] = {"obs"};
        const char* output_names[] = {"actions"};
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
        std::array<float, TerrainPolicyMath::kActionDim> action{};
        std::copy_n(outputs.at(0).GetTensorData<float>(), action.size(), action.begin());
        return action;
    }

    void Publish(
        const std::array<float, 441>& student,
        const std::array<float, 57>& teacher,
        const std::array<float, 384>& heightmap,
        const std::array<float, 3>& raw,
        const std::array<float, 3>& terrain,
        const std::array<float, 16>& action,
        const std::array<float, 8>& risk,
        bool valid,
        float age_ms) {
        drdds::msg::TeacherSample msg;
        msg.timestamp_ns = node_->get_clock()->now().nanoseconds();
        msg.sequence = sequence_++;
        msg.obs_student = student;
        msg.obs_teacher = teacher;
        msg.heightmap = heightmap;
        msg.cmd_raw = raw;
        msg.cmd_terrain = terrain;
        msg.action_teacher = action;
        msg.risk_features = risk;
        msg.heightmap_valid = valid;
        msg.heightmap_age_ms = age_ms;
        sample_pub_->publish(msg);
    }

    void InitializeRobotAction() {
        robot_action_.goal_joint_pos = VecXf::Zero(16);
        robot_action_.goal_joint_vel = VecXf::Zero(16);
        robot_action_.kp = VecXf::Zero(16);
        robot_action_.kd = VecXf::Zero(16);
        robot_action_.tau_ff = VecXf::Zero(16);
        for (std::size_t i = 0; i < 16; ++i) {
            robot_action_.kp(i) = TerrainPolicyMath::kKpRobot[i];
            robot_action_.kd(i) = TerrainPolicyMath::kKdRobot[i];
        }
    }

    void Decode(const std::array<float, 16>& action) {
        robot_action_.goal_joint_pos.setZero();
        robot_action_.goal_joint_vel.setZero();
        const auto decoded = TerrainPolicyMath::DecodeAction(action);
        for (std::size_t i = 0; i < 16; ++i) {
            robot_action_.goal_joint_pos(i) = decoded.position[i];
            robot_action_.goal_joint_vel(i) = decoded.velocity[i];
        }
    }

    std::string model_path_;
    rclcpp::Node::SharedPtr node_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    std::array<float, 16> last_action_{};
    std::array<float, 3> previous_terrain_command_{};
    std::mutex heightmap_mutex_;
    std::array<float, 384> heightmap_{};
    bool has_heightmap_{false};
    bool rewrite_enabled_{true};
    std::chrono::steady_clock::time_point heightmap_received_at_{};
    std::uint64_t sequence_{0};
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr heightmap_sub_;
    rclcpp::Publisher<drdds::msg::TeacherSample>::SharedPtr sample_pub_;
    RobotAction robot_action_;
};
