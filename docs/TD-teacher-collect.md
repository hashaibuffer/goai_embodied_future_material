# TD — 地形教师与蒸馏数据采集

- **Status:** doing
- **Depends on:** TA AutoNav collect、TB `/S10_HEIGHTMAP`、TC 冻结策略合约

## 数据因果关系

```text
cmd_raw + LiDAR -> terrain rewrite -> cmd_terrain
teacher_obs(57, cmd_terrain) -> official policy -> action_teacher
student_obs = proprio(57, cmd_raw, previous_action) + LiDAR(384)
```

同一次官方推理输出同时用于机器人控制和 `action_teacher` 标签。学生和教师观测均在更新 `last_action` 前构造，避免未来动作泄漏。

## 运行

```bash
export S10_AUTONAV_MODE=collect
export S10_START_JITTER_X=0.02
export S10_START_JITTER_Y=-0.01
export S10_START_JITTER_YAW=0.03
ros2 run s10_sdk_deploy rl_deploy --teacher-collect
```

另一个终端运行：

```bash
python3 training/distillation/collect.py \
  --output results/datasets/$(date +%Y%m%d)_seed0.npz \
  --seed 0
```

TD 只扩展 TA 的状态发布；卡死、翻滚、越界、传送和 checkpoint 仍由 TA 原实现负责。

## 当前验证记录（2026-08-19）

- `pytest -q`：57 passed。
- `colcon build --packages-select drdds s10_sdk_deploy`：2 packages finished。
- 端到端短采集：官方 57→16 教师、TA、TB、MuJoCo 与 NPZ 落盘同时运行。
- 生成并通过 `allow_pickle=False` 校验的短测数据集：831 条和 320 条；短测高度图样本全部有效。
- 观测到 `cmd_raw` 与 `cmd_terrain` 最大差异约 0.94，证明地形改写和命令平滑链路生效。
- SIGINT/SIGTERM 下采集器与 `rl_deploy` 均可正常退出。

## 尚未完成的实采验收

- 跑完全图并覆盖 `results/fail_segments.md` 中的主要失败 waypoint。
- 为主要失败段采集同位置成功对照。
- 汇总平地、危险地形、迷宫和传送后续段的数据量与成功率。

完成这些长时间实采项目后，才能把 TD 状态从 `doing` 更新为 `done`。
