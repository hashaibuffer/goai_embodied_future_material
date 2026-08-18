#include "terrain_policy_math.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
bool Near(float actual, float expected) {
    return std::fabs(actual - expected) < 1.0e-6F;
}
}

int main() {
    std::array<float, 3> omega{1.0F, -0.4F, 0.8F};
    std::array<float, 9> rotation{1.0F, 0.0F, 0.0F,
                                  0.0F, 1.0F, 0.0F,
                                  0.0F, 0.0F, 1.0F};
    std::array<float, 3> command{2.0F, -1.0F, 4.0F};
    auto position = TerrainPolicyMath::kDefaultPoseRobot;
    std::array<float, 16> velocity{};
    velocity[3] = 10.0F;
    std::array<float, 16> last_action{};
    for (std::size_t i = 0; i < last_action.size(); ++i) {
        last_action[i] = static_cast<float>(i) / 16.0F;
    }
    std::array<float, 384> heightmap{};
    for (std::size_t i = 0; i < heightmap.size(); ++i) {
        heightmap[i] = static_cast<float>(i);
    }
    std::array<float, 441> observation{};
    if (!TerrainPolicyMath::AssembleObservation(
            omega, rotation, command, position, velocity,
            last_action, heightmap, observation)) return 1;
    if (!Near(observation[0], 0.25F) || !Near(observation[1], -0.1F) ||
        !Near(observation[2], 0.2F)) return 2;
    if (!Near(observation[3], 0.0F) || !Near(observation[4], 0.0F) ||
        !Near(observation[5], -1.0F)) return 3;
    if (!Near(observation[6], 1.0F) || !Near(observation[7], -0.6F) ||
        !Near(observation[8], 1.0F)) return 4;
    for (std::size_t i = 9; i < 25; ++i) {
        if (!Near(observation[i], 0.0F)) return 5;
    }
    if (!Near(observation[25 + 12], 0.5F)) return 6;
    for (std::size_t i = 0; i < 16; ++i) {
        if (!Near(observation[41 + i], last_action[i])) return 7;
    }
    for (std::size_t i = 0; i < heightmap.size(); ++i) {
        if (!Near(observation[57 + i], heightmap[i])) return 8;
    }

    std::array<float, 16> ones{};
    ones.fill(1.0F);
    const auto decoded_ones = TerrainPolicyMath::DecodeAction(ones);
    if (!Near(decoded_ones.position[0], 0.125F) ||
        !Near(decoded_ones.position[1], -0.05F) ||
        !Near(decoded_ones.position[2], 0.85F) ||
        !Near(decoded_ones.velocity[3], 5.0F)) return 9;

    // Non-uniform a_norm: identity permute would put action[12] on hip 12,
    // not wheel 3 (kPolicyToRobot[3] == 12).
    std::array<float, 16> action{};
    action[0] = 2.0F;
    action[12] = 1.0F;
    const auto decoded = TerrainPolicyMath::DecodeAction(action);
    if (!Near(decoded.position[0], 0.25F) ||
        !Near(decoded.position[1], -0.3F) ||
        !Near(decoded.velocity[3], 5.0F) ||
        !Near(decoded.position[12], 0.0F)) return 9;

    heightmap[0] = std::numeric_limits<float>::quiet_NaN();
    if (TerrainPolicyMath::AssembleObservation(
            omega, rotation, command, position, velocity,
            last_action, heightmap, observation)) return 10;

    std::cout << "C++ observation and action contract verified\n";
    return 0;
}
