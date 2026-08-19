#pragma once
// autonav_local_planner.hpp — AutoNav 内 LiDAR 高度图局部规划器（纯函数，无 ROS 依赖）
//
// 在 AutoNav 几何命令 (fwd, side, wz) 算出之后、写入 UserCommand 之前调用：
// 读 /S10_HEIGHTMAP 的 policy 高度图 (2,16,12)，对速度方向做避障改写。
// 只改运行时 AutoNav 写出的命令，不进入教师训练（TH 仍见朴素 cmd_raw）。
//
// 判别规则（用户拍板，单阈值）：
//   h >  wall_height_m (0.40m)   -> 墙   (Block, 绕开)
//   0 < h <= 0.40m               -> 台阶/可爬 (Step, 顶着走)
//   h <  pit_m (-0.30m)          -> 深坑 (Pit, 绕开)
//   [-0.30, 0]                   -> 平地/缓下坡 (Free)
//   valid == 0                   -> 未知 (Unknown, 减速)
//
// 高度图契约见 configs/heightmap.yaml：h_norm = clip(z-ground_ref, [-0.4,0.8]) / 0.8，
// C-order 展平 idx = i*12 + j（i=x 轴 0..15，j=y 轴 0..11），共 384 = 192 高度 + 192 掩码。

#include <array>
#include <cmath>
#include <cstdint>

namespace autonav_local_plan {

// 对齐 configs/heightmap.yaml（T00 冻结）
inline constexpr int   kPolicyNx = 16;
inline constexpr int   kPolicyNy = 12;
inline constexpr int   kNumCells = kPolicyNx * kPolicyNy;   // 192
inline constexpr float kCellM    = 0.25f;   // policy 格 0.25m
inline constexpr float kXMinM    = -0.8f;   // 高度图 x 轴起点（后）
inline constexpr float kYMinM    = -1.5f;   // 高度图 y 轴起点（右）
inline constexpr float kHeightDivisorM = 0.80f;   // 归一化除数，回乘得相对高度米

struct Params {
    float horizon_s      = 1.5f;     // 积分时长 (s)
    float dt_s           = 0.1f;     // 积分步长 (s) -> 15 点
    float max_wz         = 1.0f;     // 转向段原地转角速度 (rad/s)，对齐键盘满转
    float wall_height_m  = 0.40f;    // 高于此判墙（用户拍板）
    float pit_m          = -0.30f;   // 低于此判深坑
    float wall_cost      = 1000.0f;  // 撞墙，主导
    float pit_cost       = 60.0f;    // 冲下平台
    float unknown_cost   = 20.0f;    // 逐点累积 -> 无效带被绕
    float out_cost       = 15.0f;    // 越出视场
    float step_cost      = 0.0f;     // 可爬台阶/高台：不惩罚（顶着走）；仅墙/坑/未知才绕
    float progress_w     = 40.0f;    // 每米朝航点进度
    float steer_w        = 5.0f;     // 每 rad 转向惩罚（抑抖/抑绕圈）
    float speed_w        = 1.5f;     // 每 m/s 偏离目标速度
    float blocked_thresh = 400.0f;   // 最佳代价仍高 -> vx=0 原地转
    float min_valid_ratio = 0.15f;   // 有效格占比过低 -> 降级
};

enum class CellClass : uint8_t { kFree, kBlock, kStep, kPit, kUnknown, kOut };
enum class Status   : uint8_t { kInactiveDegraded, kActive, kBlocked };

struct Result {
    float vx   = 0.0f;
    float side = 0.0f;   // 恒 0：避障转向靠 wz，不产生侧移
    float wz   = 0.0f;
    Status status = Status::kActive;
};

// 输入 h_norm/valid 各 192 个 float（已从 Float32MultiArray 解出，非空）。
// (x,y,yaw) 机器人世界位姿；(tx,ty) 目标航点世界坐标；target_vx 几何目标速度。
inline Result plan_local(const float* h_norm, const float* valid,
                         const Params& p,
                         double x, double y, double yaw,
                         double tx, double ty, double target_vx)
{
    // (1) 预分类：单阈值，O(192)，无邻域
    std::array<CellClass, kNumCells> cls;
    int valid_count = 0;
    for (int i = 0; i < kPolicyNx; ++i) {
        for (int j = 0; j < kPolicyNy; ++j) {
            int idx = i * kPolicyNy + j;
            if (valid[idx] == 0.0f) { cls[idx] = CellClass::kUnknown; continue; }
            ++valid_count;
            float h = h_norm[idx] * kHeightDivisorM;   // 回乘得相对地面高度米 [-0.4, 0.8]
            if (h > p.wall_height_m)       cls[idx] = CellClass::kBlock;
            else if (h < p.pit_m)          cls[idx] = CellClass::kPit;
            else if (h > 0.0f)             cls[idx] = CellClass::kStep;
            else                           cls[idx] = CellClass::kFree;
        }
    }

    // (2) 候选集：vx 按目标速度分档，Δθ（原地转角）固定 9 档
    const float vx_nom[4]  = {0.0f, 0.2f, 0.45f, 0.7f};
    const float vx_maze[3] = {0.0f, 0.15f, 0.35f};
    const bool  maze = (target_vx < 0.45f);   // maze 段 target_vx=0.35
    const float* vx_cand = maze ? vx_maze : vx_nom;
    const int   n_vx = maze ? 3 : 4;
    // Δθ 候选：原地转角，0 + 4 档正负比例，覆盖 [0, 1.0] rad（满转档）。
    const float dth_frac[4] = {0.15f, 0.35f, 0.6f, 1.0f};
    std::array<float, 9> dth_cand;
    dth_cand[4] = 0.0f;
    for (int k = 0; k < 4; ++k) {
        dth_cand[3 - k] = -dth_frac[k];
        dth_cand[5 + k] =  dth_frac[k];
    }
    const int   n_dth = 9;

    const float EPS = 1e-3f;
    const double cosY = std::cos(yaw), sinY = std::sin(yaw);

    double best_score = -1e30;
    float  best_vx = 0.0f, best_dth = 0.0f, best_cost = 0.0f;

    for (int iv = 0; iv < n_vx; ++iv) {
        const float vx = vx_cand[iv];
        for (int id = 0; id < n_dth; ++id) {
            const float dth = dth_cand[id];

            // (3) 轨迹代价：先原地转 Δθ、再沿新朝向直线（turn-then-go 两段式）+ 前向探针
            const float t_turn = std::fabs(dth) / p.max_wz;   // 原地转向耗时
            const float t_go   = std::max(0.0f, p.horizon_s - t_turn);
            const float L      = vx * t_go;                   // 直线段距离
            const float sgn    = (dth >= 0.0f) ? 1.0f : -1.0f;
            const float cosD   = std::cos(dth), sinD = std::sin(dth);
            const float xr_end = L * cosD, yr_end = L * sinD;

            float cost = 0.0f;
            for (float t = p.dt_s; t <= p.horizon_s + EPS; t += p.dt_s) {
                // t < t_turn 原地转（位置原点，朝向扫 0→dth）；否则沿 dth 直线
                const float tl = (t < t_turn) ? 0.0f : (t - t_turn);
                const float th = (t < t_turn) ? (sgn * p.max_wz * t) : dth;
                const float xr = vx * tl * std::cos(th);
                const float yr = vx * tl * std::sin(th);
                for (float s : {0.3f, 0.5f, 0.7f, 1.0f}) {   // 前向探针：原地转段也能"看见"扫向方向
                    const float px = xr + s * std::cos(th);
                    const float py = yr + s * std::sin(th);
                    const int ci = (int)std::floor((px - kXMinM) / kCellM);
                    const int cj = (int)std::floor((py - kYMinM) / kCellM);
                    if (ci < 0 || ci >= kPolicyNx || cj < 0 || cj >= kPolicyNy) {
                        cost += p.out_cost;
                    } else {
                        switch (cls[ci * kPolicyNy + cj]) {
                            case CellClass::kBlock:   cost += p.wall_cost;    break;
                            case CellClass::kPit:     cost += p.pit_cost;     break;
                            case CellClass::kUnknown: cost += p.unknown_cost; break;
                            case CellClass::kStep:    cost += p.step_cost;    break;
                            default: break;   // kFree 0
                        }
                    }
                }
            }

            // (4) 航点进度：直线段末端转世界系
            const double wx_end = x + xr_end * cosY - yr_end * sinY;
            const double wy_end = y + xr_end * sinY + yr_end * cosY;
            const double d0 = std::hypot(tx - x, ty - y);
            const double d1 = std::hypot(tx - wx_end, ty - wy_end);
            const double progress = d0 - d1;   // >0 表示朝航点靠近

            // (5) 得分：避障硬约束优先，progress/steer/speed 为软偏好
            const double score = -static_cast<double>(cost)
                               + p.progress_w * progress
                               - p.steer_w * std::fabs(dth)
                               - p.speed_w * std::fabs(vx - target_vx);
            if (score > best_score) {
                best_score = score;
                best_vx = vx;
                best_dth = dth;
                best_cost = cost;
            }
        }
    }

    // (6) 兜底
    Result r;
    const bool dead_lock = (best_vx < EPS && std::fabs(best_dth) < EPS);
    if (best_cost > p.blocked_thresh || (dead_lock && target_vx > EPS)) {
        // 完全挡住，或「只有原地不动才安全」的原地死锁：交还几何命令，
        // 让机器人至少动起来，由几何 + teleport / wall_push 兜底。
        r.status = Status::kBlocked;
        return r;
    }
    // 输出当前帧动作：有转角 -> 这一帧原地转（wz 满转）；无转角 -> 直线。
    r.vx = (std::fabs(best_dth) < EPS) ? best_vx : 0.0f;
    r.wz = (std::fabs(best_dth) < EPS) ? 0.0f : (best_dth > 0.0f ? p.max_wz : -p.max_wz);
    r.side = 0.0f;
    if (valid_count < p.min_valid_ratio * kNumCells) r.status = Status::kInactiveDegraded;
    else                                             r.status = Status::kActive;
    return r;
}

}  // namespace autonav_local_plan
