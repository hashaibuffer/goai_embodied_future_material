// local_planner_smoke.cpp — autonav_local_planner.hpp 行为冒烟测试（独立 g++，不碰 colcon）
//
// 三个合成 16×12 grid 断言，验证「单阈值 0.4m 判墙」的端到端行为：
//   1) 平地阵      ：全 h=0，航点正前      -> 跟航点直走（vx 高、|wz| 小）
//   2) 正前墙阵    ：正前 x≈1.2~1.7m 处 h=0.8 竖墙（clip 满）-> 直冲被改写（vx 降或 |wz| 大）
//   3) 可爬台阶阵  ：同位置 h=0.35 台阶（0~0.4m 内）          -> 不判墙，仍直走（与墙阵行为不同）
//
// 运行：
//   g++ -std=c++17 -Wall -Wextra -I src/S10_sdk_deploy/interface/user_command
//       tests/local_planner_smoke.cpp -o /tmp/lp_smoke && /tmp/lp_smoke
//
// 高度图契约：h_norm = clip(z-ground_ref,[-0.4,0.8])/0.8，idx = i*12 + j（i=x 轴 0..15，j=y 轴 0..11）。
// 规划器内回乘 kHeightDivisorM=0.8 得相对高度米。

#include "autonav_local_planner.hpp"

#include <cmath>
#include <cstdarg>
#include <cstdio>

using autonav_local_plan::Params;
using autonav_local_plan::Result;
using autonav_local_plan::plan_local;

namespace {
constexpr int NX = 16;
constexpr int NY = 12;
constexpr int N  = NX * NY;   // 192

// 设一格：h_m 为相对高度（米），v 为 validity(0/1)。内部换算成 h_norm = h_m / 0.8。
void set_cell(float* h, float* v, int i, int j, float h_m, float valid) {
    int idx = i * NY + j;
    h[idx] = h_m / 0.8f;
    v[idx] = valid;
}

void fill(float* h, float* v, float h_m, float valid) {
    for (int k = 0; k < N; ++k) { h[k] = h_m / 0.8f; v[k] = valid; }
}

int failures = 0;

void check(const char* name, bool ok, const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::printf("[%s] %s  %s\n", ok ? "PASS" : "FAIL", name, buf);
    if (!ok) ++failures;
}

// 直走判定：跟航点直冲（vx 高且 |wz| 小）。
bool is_straight(const Result& r) { return r.vx > 0.3f && std::fabs(r.wz) < 0.2f; }

// 墙阵：正前方 x≈1.2~1.7m（i∈[8,9]），y∈[-0.5,0.5]（j∈[4,7]）放一堵障碍，两侧留空可绕。
void put_front_wall(float* h, float* v, float h_m) {
    fill(h, v, 0.0f, 1.0f);
    for (int i = 8; i <= 9; ++i)
        for (int j = 4; j <= 7; ++j)
            set_cell(h, v, i, j, h_m, 1.0f);
}
}  // namespace

int main() {
    Params p;
    // 机器人位姿原点，朝向 +X（yaw=0），航点在正前 2m。
    const double x = 0.0, y = 0.0, yaw = 0.0, tx = 2.0, ty = 0.0, target_vx = 0.5;

    // (1) 平地：直走
    {
        float h[N], v[N];
        fill(h, v, 0.0f, 1.0f);
        Result r = plan_local(h, v, p, x, y, yaw, tx, ty, target_vx);
        check("平地跟航点", is_straight(r),
              "vx=%.2f wz=%.2f (期望 vx>0.3 且 |wz|<0.2)", r.vx, r.wz);
    }

    // (2) 正前墙 h=0.8（clip 满 -> Block）：直冲被改写
    {
        float h[N], v[N];
        put_front_wall(h, v, 0.8f);
        Result r = plan_local(h, v, p, x, y, yaw, tx, ty, target_vx);
        check("正前墙避让", !is_straight(r),
              "vx=%.2f wz=%.2f (期望 vx<=0.3 或 |wz|>=0.2)", r.vx, r.wz);
    }

    // (3) 同位置可爬台阶 h=0.35（0~0.4m -> Step）：不判墙，仍直走
    {
        float h[N], v[N];
        put_front_wall(h, v, 0.35f);
        Result r = plan_local(h, v, p, x, y, yaw, tx, ty, target_vx);
        check("可爬台阶不偏", is_straight(r),
              "vx=%.2f wz=%.2f (期望 vx>0.3 且 |wz|<0.2，与墙阵不同)", r.vx, r.wz);
    }

    // (4) 单阈值边界：h=0.40 恰好等于 wall_height_m，应判 Step（严格 > 才 Block）
    {
        float h[N], v[N];
        put_front_wall(h, v, 0.40f);
        Result r = plan_local(h, v, p, x, y, yaw, tx, ty, target_vx);
        check("0.4m 边界判台阶", is_straight(r),
              "vx=%.2f wz=%.2f (期望仍直走，h=0.4 不 > 0.4)", r.vx, r.wz);
    }

    // (5) 全无效图 -> 降级 kInactiveDegraded
    {
        float h[N], v[N];
        fill(h, v, 0.0f, 0.0f);
        Result r = plan_local(h, v, p, x, y, yaw, tx, ty, target_vx);
        check("全无效降级",
              r.status == autonav_local_plan::Status::kInactiveDegraded,
              "status=%d (期望 kInactiveDegraded)", (int)r.status);
    }

    std::printf("\n%s (%d 失败)\n", failures == 0 ? "全部通过" : "存在失败", failures);
    return failures == 0 ? 0 : 1;
}
