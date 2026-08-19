#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

// Dependency-free implementation of the frozen policy.yaml math contract.
// The ROS runner and standalone C++ contract test execute these same functions.
struct TerrainPolicyMath {
    static constexpr std::size_t kProprioDim = 57;
    static constexpr std::size_t kHeightmapDim = 384;
    static constexpr std::size_t kObservationDim = 441;
    static constexpr std::size_t kActionDim = 16;
    static constexpr int kDecimation = 4;
    static constexpr float kOmegaScale = 0.25F;
    static constexpr float kDofVelScale = 0.05F;

    inline static constexpr std::array<int, kActionDim> kRobotToPolicy{
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15};
    inline static constexpr std::array<int, kActionDim> kPolicyToRobot{
        0, 1, 2, 12, 3, 4, 5, 13, 6, 7, 8, 14, 9, 10, 11, 15};
    inline static constexpr std::array<float, kActionDim> kDefaultPosePolicy{
        0.0F, -0.3F, 0.6F, 0.0F, -0.3F, 0.6F, 0.0F, 0.3F,
        -0.6F, 0.0F, 0.3F, -0.6F, 0.0F, 0.0F, 0.0F, 0.0F};
    inline static constexpr std::array<float, kActionDim> kDefaultPoseRobot{
        0.0F, -0.3F, 0.6F, 0.0F, 0.0F, -0.3F, 0.6F, 0.0F,
        0.0F, 0.3F, -0.6F, 0.0F, 0.0F, 0.3F, -0.6F, 0.0F};
    inline static constexpr std::array<float, kActionDim> kActionScaleRobot{
        0.125F, 0.25F, 0.25F, 5.0F, 0.125F, 0.25F, 0.25F, 5.0F,
        0.125F, 0.25F, 0.25F, 5.0F, 0.125F, 0.25F, 0.25F, 5.0F};
    inline static constexpr std::array<float, kActionDim> kKpRobot{
        80.0F, 80.0F, 80.0F, 0.0F, 80.0F, 80.0F, 80.0F, 0.0F,
        80.0F, 80.0F, 80.0F, 0.0F, 80.0F, 80.0F, 80.0F, 0.0F};
    inline static constexpr std::array<float, kActionDim> kKdRobot{
        2.0F, 2.0F, 2.0F, 0.6F, 2.0F, 2.0F, 2.0F, 0.6F,
        2.0F, 2.0F, 2.0F, 0.6F, 2.0F, 2.0F, 2.0F, 0.6F};

    struct DecodedAction {
        std::array<float, kActionDim> position{};
        std::array<float, kActionDim> velocity{};
    };

    static bool AssembleObservation(
        const std::array<float, 3>& base_omega,
        const std::array<float, 9>& rotation_world_from_body,
        const std::array<float, 3>& command,
        const std::array<float, kActionDim>& joint_position_robot,
        const std::array<float, kActionDim>& joint_velocity_robot,
        const std::array<float, kActionDim>& last_action,
        const std::array<float, kHeightmapDim>& heightmap,
        std::array<float, kObservationDim>& observation) {
        observation.fill(0.0F);
        std::size_t cursor = 0;
        for (float value : base_omega) observation[cursor++] = value * kOmegaScale;
        for (std::size_t i = 0; i < 3; ++i) {
            observation[cursor++] = -rotation_world_from_body[6 + i];
        }
        observation[cursor++] = std::clamp(command[0], -1.0F, 1.0F);
        observation[cursor++] = std::clamp(command[1], -0.6F, 0.6F);
        observation[cursor++] = std::clamp(command[2], -1.0F, 1.0F);
        for (std::size_t i = 0; i < kActionDim; ++i) {
            const float position = i >= 12 ? 0.0F : joint_position_robot[kRobotToPolicy[i]];
            observation[cursor++] = position - kDefaultPosePolicy[i];
        }
        for (std::size_t i = 0; i < kActionDim; ++i) {
            observation[cursor++] = joint_velocity_robot[kRobotToPolicy[i]] * kDofVelScale;
        }
        for (float value : last_action) observation[cursor++] = value;
        for (float value : heightmap) observation[cursor++] = value;
        return cursor == kObservationDim &&
            std::all_of(observation.begin(), observation.end(),
                        [](float value) { return std::isfinite(value); });
    }

    static DecodedAction DecodeAction(const std::array<float, kActionDim>& action) {
        DecodedAction decoded;
        for (std::size_t robot_index = 0; robot_index < kActionDim; ++robot_index) {
            const float physical =
                action[kPolicyToRobot[robot_index]] * kActionScaleRobot[robot_index] +
                kDefaultPoseRobot[robot_index];
            if (robot_index % 4 == 3) decoded.velocity[robot_index] = physical;
            else decoded.position[robot_index] = physical;
        }
        return decoded;
    }
};
