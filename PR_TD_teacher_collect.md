# TD 官方教师采集与 TH/Issue #5 分工调整

## 变更概述

- 修正 TD 地形风险计算：使用中央轮足走廊的连续局部高差，降低侧墙、单点噪声和无效高度图造成的误判。
- 将 `cmd_terrain` 调整为轻量改写：平地最多加速 5%，高风险最多减速 5%，横向偏移上限 0.03，不削弱转向速度。
- 增加 `S10_TD_REWRITE_MODE=off|on` A/B 开关，用于官方 AutoNav 基线与 TD 规则对比。
- C++ 在线实现与 Python 离线实现保持一致。
- 将台阶、上下高差和墙迷宫路段移交 TH；TD 只负责平地/缓坡官方教师数据。

## 路段分工

- TD：平地、缓坡，以及完整路线覆盖。
- TH：`6→7`、`15→16`、`16→17`、`17→18`、`20→21`、`22→23`、`23→24`、`27→28` 和 `28–32`。
- Issue #5：运行时 AutoNav 使用 LiDAR 高度图做局部绕墙规划；不训练第二网络，也不负责台阶 locomotion。

## 验证结果

- `pytest -q`：66 passed
- `colcon build --packages-select drdds s10_sdk_deploy`：成功
- 两轮最终 TD 数据：50,412 条
- 全路线 waypoint 0–32 覆盖
- TH 困难段覆盖完整
- `command_contract_ok: true`
- `teacher_source` 全部为 official

## 验收说明

TD 不再要求官方盲策略在台阶或迷宫中提供成功标签；这些标签由 TH 的 privileged teacher 提供。正式训练使用最终规则的 `20260819_final_seed0_*` 与 `20260819_final_seed1_*` 数据，不混入旧版慢速采集数据。
