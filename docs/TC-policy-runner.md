# TC — 自研 16 维策略 runner 骨架

> 通俗版：官方程序里有一段代码负责"收集机器人状态 → 喂给模型 → 拿到 16 个关节命令 → 发给仿真"。我们把它换成自己写的版本：观测拼成 441 维（本体 57 维 + 高度图 384 维），先塞一个假模型证明链路能通，之后换成真模型。

- **Status:** done
- **Priority:** P0
- **Owner:** ThinkPad-3
- **Machine:** Jazzy ThinkPad
- **Depends on:** T00 contracts
- **Unblocks:** 加载 TE 训练出的真模型；TG eval-submit

## 目标

在官方 `rl_deploy` 的路径上换上我们自己的策略 runner：输入 441 维观测，输出 16 维关节命令。**先用假模型**（随机输出或单位映射）证明"加载 + 发布 `/JOINTS_CMD`"这条路是通的。正式提交默认不再指向官方 `policy.onnx`。

> 为什么先做骨架：真模型（TE）和真雷达（TB）都要时间，而"能不能加载模型、拼对观测"这件事和模型内容无关。先把壳搭好，等模型好了直接换上就行。

这张卡**不等待**真雷达和真模型：高度图先用全 0，命令先用 TA 的（TA 没就绪就用 0），能站稳不崩即可。

## 范围（这张卡要做的事）

1. 按 `configs/policy.yaml` 拼观测：官方 57 维（角速度、投影重力、速度命令、关节位置、关节速度、上次动作）+ 高度图 384 维。
2. 单位转换：IMU 在 ROS 里是"度"，进策略前转成"弧度"，和官方 `S10PolicyRunner` 保持一致。
3. 这些**保持官方默认值，不要改**：关节在策略里的排列顺序 `policy_order`、`action_scale_robot` 缩放系数、默认姿态、PD 增益（`kp/kd`）、策略运行频率（约 50 Hz，即每隔 4 个 5ms 控制周期跑一次）。
4. 加载 `models/terrain_locomotion.onnx`（现在先放一个占位模型）。
5. 做一个 `--controller proprio_clone | learned` 开关：
   - `proprio_clone`（纯本体克隆）：只依赖本体状态，不看雷达——把高度图的 validity 全置 0。
   - `learned`（带感知）：用真实雷达高度图。
6. 写单元测试：维度对不对、展平顺序对不对、模型输入输出名对不对。

## 不在范围

- 训练网络（那是 TE）。
- 实现雷达（那是 TB）。
- 改导航逻辑（TA 负责出速度命令接口；TC 只消费高度图和本体状态）。如果 AutoNav 还没合入，runner 也要能在命令为 0 时让机器人站稳不崩。

## 建议步骤

1. 对照 `s10_policy_runner.hpp`，先抄它的前 57 维拼法，再在后面接上高度图。
2. 用 PyTorch 导出一个零权重或很小的 ONNX 占位模型，形状 `[1,441] → [1,16]`。
3. 改模型路径，明确默认不用官方文件。
4. 和 TA 约定速度命令怎么进来：推荐 AutoNav 继续写 `UserCommand`（官方接口），高度图走 ROS 话题，这样不用多加一层延迟。
5. 写 `tests/test_policy_io.py` 测试。

## 交付

```text
src/s10_terrain_policy/               # 自研 runner 代码
models/terrain_locomotion.onnx        # 占位模型，TE 出真模型后覆盖
models/model_card.md                  # 先写维度说明，后补训练信息
configs/policy.yaml                   # 策略 I/O 约定
tests/test_policy_io.py               # 输入输出测试
```

## 完成定义

- [x] 假 ONNX 能被 `rl_deploy` 加载，runner 的 16 维命令经 `SetJointCommand` 发布到 `/JOINTS_CMD`。
- [x] 观测维度 441，字段顺序与 YAML 完全一致。
- [x] 默认启动路径不是官方 `policy.onnx`。
- [x] 度/弧度转换、关节顺序有测试或注释对照表。

## 验证记录（2026-08-18）

- `pytest -q`：26 passed（包含实际编译执行的 C++ 观测拼接与动作解码测试）。
- `tests/onnx_smoke.cpp`：验证 `obs [1,441] -> actions [1,16]`，零模型输出全部为 0。
- TC 已迁入 `goai_embodied_future_material`，构建不再依赖外部 `s10-terrain-aware-policy` 路径。
- 运行时链路核对：4 个 5 ms 控制周期执行一次推理，结果经 `SetJointCommand` 发布到 `/JOINTS_CMD`。
- 已提供 `scripts/tc_joint_cmd_smoke.py`；使用 `--use-simulator` 消费官方 MuJoCo 的健康关节、IMU 与位姿状态完成验收。
- 官方 MuJoCo 最终验收通过：成功加载 `S10_track.xml`（16 DoF），状态机进入 `rl_control`，烟测观察到 `TerrainPolicyRunner` 在 `/JOINTS_CMD` 发布命令。

## 风险

| 现象 | 处理 |
| --- | --- |
| 假模型让机器人倒下 | 可接受，这张卡不要求走赛道；合流后换真模型 |
| 高度图话题还没到、消息无效或超过 250 ms 未更新 | 用全 0 + validity 0，别让策略线程卡死或继续使用陈旧地形 |
