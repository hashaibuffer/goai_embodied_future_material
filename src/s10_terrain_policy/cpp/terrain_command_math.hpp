#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

// Dependency-free TD terrain risk and command rewrite. Python tests mirror this
// contract so collection and offline inspection use identical semantics.
struct TerrainCommandMath {
    static constexpr std::size_t kHeightmapDim = 384;
    static constexpr std::size_t kRiskDim = 8;
    static constexpr std::size_t kNx = 16;
    static constexpr std::size_t kNy = 12;

    struct Result {
        std::array<float, 3> command{};
        std::array<float, kRiskDim> risk{};
    };

    static float Clamp(float value, float low, float high) {
        return std::max(low, std::min(high, value));
    }

    static float Slew(float previous, float target, float maximum_delta) {
        return previous + Clamp(target - previous, -maximum_delta, maximum_delta);
    }

    static Result Rewrite(
        const std::array<float, 3>& raw,
        const std::array<float, kHeightmapDim>& grid,
        const std::array<float, 3>& previous) {
        // Grid is CHW: 192 normalized heights followed by 192 validity values.
        float max_step = 0.0F;
        float max_drop = 0.0F;
        float near_sum = 0.0F, far_sum = 0.0F;
        float near_n = 0.0F, far_n = 0.0F;
        float left = 0.0F, right = 0.0F;
        float valid = 0.0F, total = 0.0F;
        for (std::size_t x = 4; x <= 11; ++x) {  // 0.325m .. 2.075m ahead
            for (std::size_t y = 0; y < kNy; ++y) {
                const std::size_t index = x * kNy + y;
                const float mask = grid[kNx * kNy + index] > 0.5F ? 1.0F : 0.0F;
                ++total;
                if (mask == 0.0F) continue;
                ++valid;
                const float height_m = grid[index] * 0.8F;
                max_step = std::max(max_step, height_m);
                max_drop = std::max(max_drop, -height_m);
                if (x <= 7) { near_sum += height_m; ++near_n; }
                else { far_sum += height_m; ++far_n; }
                const float obstacle = std::max(0.0F, height_m);
                if (y >= 6) left = std::max(left, obstacle);
                else right = std::max(right, obstacle);
            }
        }
        const float valid_fraction = total > 0.0F ? valid / total : 0.0F;
        const float unknown_fraction = 1.0F - valid_fraction;
        const float near_mean = near_n > 0.0F ? near_sum / near_n : 0.0F;
        const float far_mean = far_n > 0.0F ? far_sum / far_n : near_mean;
        const float slope = far_mean - near_mean;
        const float terrain_risk = std::max({
            max_step / 0.25F,
            max_drop / 0.20F,
            std::fabs(slope) / 0.20F,
            std::max(0.0F, (unknown_fraction - 0.35F) / 0.65F)});
        const float risk_score = Clamp(terrain_risk, 0.0F, 1.0F);

        std::array<float, 3> target{
            Clamp(raw[0], -1.0F, 1.0F),
            Clamp(raw[1], -0.6F, 0.6F),
            Clamp(raw[2], -1.0F, 1.0F)};
        const float speed_scale = 1.0F - 0.75F * risk_score;
        target[0] *= speed_scale;
        const float avoidance = Clamp((right - left) * 1.2F, -0.25F, 0.25F);
        target[1] = Clamp(target[1] + avoidance, -0.6F, 0.6F);
        target[2] *= 1.0F - 0.35F * risk_score;

        Result result;
        result.command = {
            Slew(previous[0], target[0], 0.04F),
            Slew(previous[1], target[1], 0.03F),
            Slew(previous[2], target[2], 0.06F)};
        result.risk = {
            max_step, max_drop, slope, left, right,
            unknown_fraction, valid_fraction, risk_score};
        return result;
    }
};
