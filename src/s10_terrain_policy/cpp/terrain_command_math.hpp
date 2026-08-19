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
        std::array<float, 8> profile{};
        std::array<bool, 8> profile_valid{};
        std::array<float, 24> left_samples{}, right_samples{};
        std::size_t left_n = 0, right_n = 0;
        float valid = 0.0F, total = 0.0F;
        for (std::size_t x = 4; x <= 11; ++x) {
            std::array<float, 6> corridor{};
            std::size_t corridor_n = 0;
            for (std::size_t y = 0; y < kNy; ++y) {
                const std::size_t index = x * kNy + y;
                const float mask = grid[kNx * kNy + index] > 0.5F ? 1.0F : 0.0F;
                ++total;
                if (mask == 0.0F) continue;
                ++valid;
                const float height_m = grid[index] * 0.8F;
                if (y >= 3 && y <= 8) corridor[corridor_n++] = height_m;
                if (y >= 9) left_samples[left_n++] = height_m;
                if (y <= 2) right_samples[right_n++] = height_m;
            }
            if (corridor_n >= 3) {
                std::sort(corridor.begin(), corridor.begin() + corridor_n);
                const std::size_t middle = corridor_n / 2;
                profile[x - 4] = corridor_n % 2 == 0
                    ? 0.5F * (corridor[middle - 1] + corridor[middle])
                    : corridor[middle];
                profile_valid[x - 4] = true;
            }
        }
        const auto percentile75 = [](auto& samples, std::size_t count) {
            if (count == 0) return 0.0F;
            std::sort(samples.begin(), samples.begin() + count);
            return std::max(0.0F, samples[static_cast<std::size_t>(0.75F * (count - 1))]);
        };
        const float left = percentile75(left_samples, left_n);
        const float right = percentile75(right_samples, right_n);
        float max_step = 0.0F, max_drop = 0.0F;
        for (std::size_t x = 1; x < profile.size(); ++x) {
            if (!profile_valid[x - 1] || !profile_valid[x]) continue;
            const float difference = profile[x] - profile[x - 1];
            max_step = std::max(max_step, difference);
            max_drop = std::max(max_drop, -difference);
        }
        float slope = 0.0F;
        std::size_t first = profile.size(), last = 0, profile_n = 0;
        for (std::size_t x = 0; x < profile.size(); ++x) {
            if (!profile_valid[x]) continue;
            first = std::min(first, x);
            last = x;
            ++profile_n;
        }
        if (profile_n >= 4) slope = profile[last] - profile[first];
        const float valid_fraction = total > 0.0F ? valid / total : 0.0F;
        const float unknown_fraction = 1.0F - valid_fraction;
        const float terrain_risk = std::max({
            max_step / 0.25F,
            max_drop / 0.20F,
            0.45F * std::fabs(slope) / 0.35F,
            0.70F * std::max(0.0F, (unknown_fraction - 0.45F) / 0.55F)});
        const float risk_score = Clamp(terrain_risk, 0.0F, 1.0F);

        std::array<float, 3> target{
            Clamp(raw[0], -1.0F, 1.0F),
            Clamp(raw[1], -0.6F, 0.6F),
            Clamp(raw[2], -1.0F, 1.0F)};
        const float active_risk = Clamp((risk_score - 0.25F) / 0.75F, 0.0F, 1.0F);
        const float speed_scale = 1.05F - 0.10F * active_risk;
        target[0] = Clamp(target[0] * speed_scale, -1.0F, 1.0F);
        const float difference = right - left;
        const float imbalance = std::copysign(
            std::max(std::fabs(difference) - 0.12F, 0.0F), difference);
        const float avoidance = Clamp(imbalance * 0.15F, -0.03F, 0.03F);
        target[1] = Clamp(target[1] + avoidance, -0.6F, 0.6F);
        target[2] *= 1.0F;

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
