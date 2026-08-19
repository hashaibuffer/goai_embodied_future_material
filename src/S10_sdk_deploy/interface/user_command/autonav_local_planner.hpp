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
    float max_wz         = 1.0f;     // 转向候选上限 (rad/s)，对齐键盘满转
    float wall_height_m  = 0.40f;    // 高于此判墙（用户拍板）
    float pit_m          = -0.30f;   // 低于此判深坑
    float wall_cost      = 1000.0f;  // 撞墙，主导
    float pit_cost       = 60.0f;    // 冲下平台
    float unknown_cost   = 20.0f;    // 逐点累积 -> 无效带被绕
    float out_cost       = 15.0f;    // 越出视场
    float step_cost      = 0.0f;     // 可爬台阶/高台：不惩罚（顶着走）；仅墙/坑/未知才绕
    float progress_w     = 40.0f;    // 每米朝航点进度
    float steer_w        = 5.0f;     // 每 rad/s 转向惩罚（抑抖/抑绕圈）
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

    // (2) 候选集：vx 按目标速度分档，wz 固定 9 档
    const float vx_nom[4]  = {0.0f, 0.2f, 0.45f, 0.7f};
    const float vx_maze[3] = {0.0f, 0.15f, 0.35f};
    const bool  maze = (target_vx < 0.45f);   // maze 段 target_vx=0.35
    const float* vx_cand = maze ? vx_maze : vx_nom;
    const int   n_vx = maze ? 3 : 4;
    // wz 候选由 max_wz 生成：0 + 4 档正负比例，覆盖 [0, max_wz]（含满转）。
    const float wz_frac[4] = {0.15f, 0.35f, 0.6f, 1.0f};
    std::array<float, 9> wz_cand;
    wz_cand[4] = 0.0f;
    for (int k = 0; k < 4; ++k) {
        wz_cand[3 - k] = -p.max_wz * wz_frac[k];
        wz_cand[5 + k] =  p.max_wz * wz_frac[k];
    }
    const int   n_wz = 9;

    const float EPS = 1e-3f;
    const double cosY = std::cos(yaw), sinY = std::sin(yaw);

    double best_score = -1e30;
    float  best_vx = 0.0f, best_wz = 0.0f, best_cost = 0.0f;

    for (int iv = 0; iv < n_vx; ++iv) {
        const float vx = vx_cand[iv];
        for (int iw = 0; iw < n_wz; ++iw) {
            const float wz = wz_cand[iw];

            // (3) 轨迹代价：车体系圆弧积分 + 前向探针
            float cost = 0.0f;
            float xr_end = 0.0f, yr_end = 0.0f;
            for (float t = p.dt_s; t <= p.horizon_s + EPS; t += p.dt_s) {
                const float th = wz * t;
                float xr, yr;
                if (std::fabs(wz) < EPS) { xr = vx * t; yr = 0.0f; }
                else {
                    xr = vx / wz * std::sin(wz * t);
                    yr = vx / wz * (1.0f - std::cos(wz * t));
                }
                xr_end = xr; yr_end = yr;
                for (float s : {0.3f, 0.5f, 0.7f, 1.0f}) {   // 前向探针：vx=0 原地转也能"看见"转向方向
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

            // (4) 航点进度：弧末端转世界系
            const double wx_end = x + xr_end * cosY - yr_end * sinY;
            const double wy_end = y + xr_end * sinY + yr_end * cosY;
            const double d0 = std::hypot(tx - x, ty - y);
            const double d1 = std::hypot(tx - wx_end, ty - wy_end);
            const double progress = d0 - d1;   // >0 表示朝航点靠近

            // (5) 得分：避障硬约束优先，progress/steer/speed 为软偏好
            const double score = -static_cast<double>(cost)
                               + p.progress_w * progress
                               - p.steer_w * std::fabs(wz)
                               - p.speed_w * std::fabs(vx - target_vx);
            if (score > best_score) {
                best_score = score;
                best_vx = vx;
                best_wz = wz;
                best_cost = cost;
            }
        }
    }

    // (6) 兜底
    Result r;
    const bool dead_lock = (best_vx < EPS && std::fabs(best_wz) < EPS);
    if (best_cost > p.blocked_thresh || (dead_lock && target_vx > EPS)) {
        // 完全挡住，或「只有原地不动才安全」的原地死锁：交还几何命令，
        // 让机器人至少动起来，由几何 + teleport / wall_push 兜底。
        r.status = Status::kBlocked;
        return r;
    }
    r.vx = best_vx;
    r.wz = best_wz;
    r.side = 0.0f;
    if (valid_count < p.min_valid_ratio * kNumCells) r.status = Status::kInactiveDegraded;
    else                                             r.status = Status::kActive;
    return r;
}

}  // namespace autonav_local_plan
