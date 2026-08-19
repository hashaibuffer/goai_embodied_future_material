#pragma once

#include "policy_runner_base.hpp"
#include "terrain_policy_math.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

// C++ deployment implementation of the frozen configs/policy.yaml contract.
// RobotBasicState::base_rpy is already radians because DdsInterface::HandlerIMU
// converts the degree-valued ROS message before RLControlState sees it.
class TerrainPolicyRunner final : public PolicyRunnerBase {
public:
    enum class Controller { kProprioClone, kLearned };

    static Controller ParseController(const std::string& value) {
        if (value == "proprio_clone") return Controller::kProprioClone;
        if (value == "learned") return Controller::kLearned;
        throw std::invalid_argument(
            "--controller must be proprio_clone or learned, got: " + value);
    }

    TerrainPolicyRunner(
        const std::string& model_path,
        Controller controller,
        const rclcpp::Node::SharedPtr& node,
        const std::string& heightmap_topic = "/S10_HEIGHTMAP")
        : PolicyRunnerBase("terrain_policy"),
          model_path_(model_path),
          controller_(controller),
          env_(ORT_LOGGING_LEVEL_WARNING, "TerrainPolicyRunner"),
          session_options_(),
          session_(nullptr),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        SetDecimation(kDecimation);
        session_options_.SetIntraOpNumThreads(4);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
        ValidateModelSignature();

        observation_.fill(0.0F);
        last_action_.fill(0.0F);
        heightmap_.fill(0.0F);
        InitializeRobotAction();

        heightmap_sub_ = node->create_subscription<std_msgs::msg::Float32MultiArray>(
            heightmap_topic,
            rclcpp::QoS(1).best_effort(),
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                if (msg->data.size() != kHeightmapDim) {
                    ++invalid_heightmap_count_;
                    ClearHeightmap();
                    return;
                }
                std::array<float, kHeightmapDim> candidate{};
                for (std::size_t i = 0; i < candidate.size(); ++i) {
                    if (!std::isfinite(msg->data[i])) {
                        ++invalid_heightmap_count_;
                        ClearHeightmap();
                        return;
                    }
                    candidate[i] = msg->data[i];
                }
                std::lock_guard<std::mutex> lock(heightmap_mutex_);
                heightmap_ = candidate;
                has_heightmap_ = true;
                heightmap_received_at_ = std::chrono::steady_clock::now();
            });
    }

    void DisplayPolicyInfo() override {
        std::cout << "TerrainPolicyRunner model=" << model_path_
                  << " controller="
                  << (controller_ == Controller::kLearned ? "learned" : "proprio_clone")
                  << " input=[1,441] output=[1,16]" << std::endl;
    }

    void OnEnter() override {
        run_cnt_ = 0;
        cmd_vel_input_.setZero();
        last_action_.fill(0.0F);
    }

    RobotAction getRobotAction(const RobotBasicState& robot, const UserCommand& command) override {
        if (!AssembleObservation(robot, command)) {
            ++invalid_observation_count_;
            last_action_.fill(0.0F);
            DecodeAction(last_action_);
            return robot_action_;
        }
        auto action = Infer();
        if (!AllFinite(action)) {
            ++invalid_action_count_;
            action.fill(0.0F);
        }
        last_action_ = action;
        DecodeAction(action);
        ++run_cnt_;
        return robot_action_;
    }

    std::uint64_t invalid_heightmap_count() const { return invalid_heightmap_count_.load(); }
    std::uint64_t invalid_observation_count() const { return invalid_observation_count_.load(); }
    std::uint64_t invalid_action_count() const { return invalid_action_count_.load(); }

private:
    static constexpr std::size_t kProprioDim = 57;
    static constexpr std::size_t kHeightmapDim = 384;
    static constexpr std::size_t kObservationDim = 441;
    static constexpr std::size_t kActionDim = 16;
    static constexpr int kDecimation = 4;
    static constexpr float kOmegaScale = 0.25F;
    static constexpr float kDofVelScale = 0.05F;
    static constexpr auto kHeightmapTimeout = std::chrono::milliseconds(250);

    inline static constexpr std::array<float, kActionDim> kKpRobot{
        80.0F, 80.0F, 80.0F, 0.0F, 80.0F, 80.0F, 80.0F, 0.0F,
        80.0F, 80.0F, 80.0F, 0.0F, 80.0F, 80.0F, 80.0F, 0.0F};
    inline static constexpr std::array<float, kActionDim> kKdRobot{
        2.0F, 2.0F, 2.0F, 0.6F, 2.0F, 2.0F, 2.0F, 0.6F,
        2.0F, 2.0F, 2.0F, 0.6F, 2.0F, 2.0F, 2.0F, 0.6F};

    static bool AllFinite(const std::array<float, kActionDim>& values) {
        return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
    }

    static void RequireShape(
        const std::vector<int64_t>& actual,
        const std::array<int64_t, 2>& expected,
        const std::string& label) {
        // ONNX dynamic_axes export batch as -1 (sometimes 0). Only a positive
        // batch dim must match the runtime feed of 1.
        if (actual.size() != expected.size() ||
            actual.back() != expected.back() ||
            (actual[0] > 0 && actual[0] != expected[0])) {
            throw std::runtime_error(label + " tensor has wrong shape");
        }
    }

    void ValidateModelSignature() {
        if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
            throw std::runtime_error("model must have exactly one input and one output");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session_.GetInputNameAllocated(0, allocator);
        auto output_name = session_.GetOutputNameAllocated(0, allocator);
        if (std::string(input_name.get()) != "obs" || std::string(output_name.get()) != "actions") {
            throw std::runtime_error("model tensor names must be obs and actions");
        }
        const auto input_type = session_.GetInputTypeInfo(0);
        const auto output_type = session_.GetOutputTypeInfo(0);
        const auto input_info = input_type.GetTensorTypeAndShapeInfo();
        const auto output_info = output_type.GetTensorTypeAndShapeInfo();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("model input and output must be float32");
        }
        RequireShape(input_info.GetShape(), {1, static_cast<int64_t>(kObservationDim)}, "input");
        RequireShape(output_info.GetShape(), {1, static_cast<int64_t>(kActionDim)}, "output");
    }

    void InitializeRobotAction() {
        robot_action_.goal_joint_pos = VecXf::Zero(kActionDim);
        robot_action_.goal_joint_vel = VecXf::Zero(kActionDim);
        robot_action_.kp = VecXf::Zero(kActionDim);
        robot_action_.kd = VecXf::Zero(kActionDim);
        robot_action_.tau_ff = VecXf::Zero(kActionDim);
        for (std::size_t i = 0; i < kActionDim; ++i) {
            robot_action_.kp(i) = kKpRobot[i];
            robot_action_.kd(i) = kKdRobot[i];
        }
    }

    void ClearHeightmap() {
        std::lock_guard<std::mutex> lock(heightmap_mutex_);
        heightmap_.fill(0.0F);
        has_heightmap_ = false;
    }

    bool AssembleObservation(const RobotBasicState& robot, const UserCommand& user_command) {
        std::array<float, 3> omega{};
        std::array<float, 9> rotation{};
        std::array<float, 3> command{
            user_command.forward_vel_scale,
            user_command.side_vel_scale,
            user_command.turnning_vel_scale};
        std::array<float, kActionDim> joint_position{};
        std::array<float, kActionDim> joint_velocity{};
        for (std::size_t i = 0; i < 3; ++i) omega[i] = robot.base_omega(i);
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t col = 0; col < 3; ++col) {
                rotation[row * 3 + col] = robot.base_rot_mat(row, col);
            }
        }
        for (std::size_t i = 0; i < kActionDim; ++i) {
            joint_position[i] = robot.joint_pos(i);
            joint_velocity[i] = robot.joint_vel(i);
        }
        std::array<float, kHeightmapDim> heightmap{};
        if (controller_ == Controller::kLearned) {
            std::lock_guard<std::mutex> lock(heightmap_mutex_);
            const bool fresh = has_heightmap_ &&
                std::chrono::steady_clock::now() - heightmap_received_at_ <= kHeightmapTimeout;
            if (fresh) heightmap = heightmap_;
            else has_heightmap_ = false;
        }
        return TerrainPolicyMath::AssembleObservation(
            omega, rotation, command, joint_position, joint_velocity,
            last_action_, heightmap, observation_);
    }

    std::array<float, kActionDim> Infer() {
        constexpr std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(kObservationDim)};
        auto input = Ort::Value::CreateTensor<float>(
            memory_info_, observation_.data(), observation_.size(),
            input_shape.data(), input_shape.size());
        const char* input_names[] = {"obs"};
        const char* output_names[] = {"actions"};
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
        const float* data = outputs.at(0).GetTensorData<float>();
        std::array<float, kActionDim> action{};
        std::copy_n(data, kActionDim, action.begin());
        return action;
    }

    void DecodeAction(const std::array<float, kActionDim>& action) {
        robot_action_.goal_joint_pos.setZero();
        robot_action_.goal_joint_vel.setZero();
        const auto decoded = TerrainPolicyMath::DecodeAction(action);
        for (std::size_t robot_index = 0; robot_index < kActionDim; ++robot_index) {
            robot_action_.goal_joint_pos(robot_index) = decoded.position[robot_index];
            robot_action_.goal_joint_vel(robot_index) = decoded.velocity[robot_index];
        }
    }

    std::string model_path_;
    Controller controller_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    std::array<float, kObservationDim> observation_{};
    std::array<float, kActionDim> last_action_{};
    std::mutex heightmap_mutex_;
    std::array<float, kHeightmapDim> heightmap_{};
    bool has_heightmap_{false};
    std::chrono::steady_clock::time_point heightmap_received_at_{};
    std::atomic<std::uint64_t> invalid_heightmap_count_{0};
    std::atomic<std::uint64_t> invalid_observation_count_{0};
    std::atomic<std::uint64_t> invalid_action_count_{0};
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr heightmap_sub_;
    RobotAction robot_action_;
};
