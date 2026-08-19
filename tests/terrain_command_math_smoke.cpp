#include "terrain_command_math.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    std::array<float, 384> flat{};
    for (std::size_t i = 192; i < flat.size(); ++i) flat[i] = 1.0F;
    const std::array<float, 3> raw{1.0F, 0.0F, 0.0F};
    const auto clear = TerrainCommandMath::Rewrite(raw, flat, raw);
    assert(std::fabs(clear.command[0] - 1.0F) < 1e-6F);
    assert(clear.risk[7] == 0.0F);

    auto step = flat;
    for (std::size_t y = 3; y <= 8; ++y)
        step[8 * 12 + y] = 0.5F;  // broad normalized 0.5 -> 0.4m step
    for (std::size_t x = 4; x <= 11; ++x)
        for (std::size_t y = 9; y <= 11; ++y) step[x * 12 + y] = 0.5F;
    const auto blocked = TerrainCommandMath::Rewrite(raw, step, raw);
    assert(blocked.risk[0] > 0.39F);
    assert(blocked.risk[7] == 1.0F);
    assert(blocked.command[0] < raw[0]);
    assert(raw[0] - blocked.command[0] <= 0.04001F);
    assert(blocked.command[1] < 0.0F);  // left obstacle -> steer right
    std::cout << "TD terrain command contract verified\n";
}
