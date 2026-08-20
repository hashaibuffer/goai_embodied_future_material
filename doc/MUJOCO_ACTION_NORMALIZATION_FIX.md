# MuJoCo 教师动作归一化故障说明（含 2026-08-20 更正）

## 最新结论（2026-08-20，方案 B，当前生效）

`model_43100.pt` / `model_44196.pt` 的 actor 输出**不是** `[-1,1]` 归一化动作，也**不应该**被裁剪到
`[-1,1]`。正确契约是：

```text
actor(obs)
  -> Gaussian deterministic mean（原始 MLP 输出，即 a_raw）
  -> clip 到 ±100（对齐 Isaac rsl_rl clip_actions=100，仅安全上限，非归一化）
  -> action_scale_robot 缩放 + default_pose_robot 偏置
  -> 关节/轮子控制器
```

`a_raw` 的量级可以是 `3`、`4`、`8` 等，不同关节之间的幅度差异（例如台阶动作里左前腿 knee `4.5`
vs 右前腿 knee `3.3`）是策略要表达的真实信息。之前把这份输出强制 clip 到 `[-1,1]`，会把两者都压成
`±1`，抹掉左右腿的差异化动作，是导致学生策略上台阶时"一腿正常、一腿卡住"的直接原因。

因此：

- 导出/采集/数据集/学生训练**统一使用原始动作契约**，字段名为 `teacher_action_raw`（不再是
  `teacher_action_norm`）；
- 唯一的裁剪是 `±100` 安全上限，用来防止异常 checkpoint 输出发散，**不是**归一化步骤；
- 学生推理时对自己的输出做同样的解释：`raw_action -> ×action_scale_robot -> +default_pose_robot`，
  不能假设标签已经落在 `[-1,1]`。

本节之前（2026-08-19 之前）的调查把 `clip[-1,1]` 当作"MuJoCo `a_norm` 协议所需的安全适配层"，
这是**错误结论**，已被下方"历史误判与纠正过程"记录并推翻，仅作追溯参考，不再作为当前操作依据。

## 为什么之前会得出 `clip[-1,1]` 的错误结论

最初观察到 `model_43100.pt` 原始输出的绝对值可以到 `~11`，直接乘 `action_scale_robot` 会产生
`~2-3 rad` 的关节目标偏移或 `~55 rad/s` 的轮速目标，导致机器人瞬间摔倒。当时唯一已知的"边界"是
MuJoCo 控制器要求动作有限、不发散，于是采用了 `clip[-1,1]` 作为临时安全约束，并未逐行核对 Isaac
`rsl_rl` 的 `vecenv_wrapper.step()` 和 `JointPositionActionCfg` 的真实裁剪范围。

后续交叉核对 `rl_training` 的 `rsl_rl_ppo_cfg.py`（`clip_actions = 100`）、
`rough_env_cfg.py`（`self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}`）、IsaacLab
`isaaclab_rl/rsl_rl/vecenv_wrapper.py`（在送入环境前按 `clip_actions` 裁剪）和
`isaaclab/envs/mdp/actions/joint_actions.py`（`processed = raw_action * scale + default_joint_pos`，
裁剪发生在乘 scale、加默认位置之后）后确认：Isaac 训练/部署链路里裁剪上限是 `±100`，且裁剪对象是
"即将乘 scale 的原始动作"，从未出现过 `[-1,1]` 这个边界。真机 `terrain_policy_math.hpp::DecodeAction`
同样没有 `[-1,1]` 裁剪。`[-1,1]` 完全是 MuJoCo 采集脚本单方面加上去的错误约束。

## Isaac 侧实际行为（更正后）

```text
actor(obs)
  -> Gaussian deterministic mean（原始 MLP 输出）
  -> rsl_rl VecEnvWrapper: clip_actions=100（±100 安全上限）
  -> JointPositionAction: raw_action * action_scale_robot + default_pose_robot
  -> （环境侧 clip 配置里也标注了 ±100，但此时数值已经是关节目标空间，通常不会触发）
  -> 关节/轮子控制器
```

当前代码没有 `tanh`，也没有 `[-1,1]` 的 Isaac 动作截断；唯一存在的截断是 `±100`。

## 已实施修复（方案 B）

### ONNX 导出（`tools/export_s10_teacher_onnx.py`）

- `--action-postprocess clip`（默认）：在 ONNX 图内执行对称 clip，**默认上限改为 `100`**
  （`--action-limit 100`），对齐 Isaac `clip_actions=100`；
- `--action-postprocess tanh`：仍保留，供特殊场景手动选用，默认不使用；
- `--action-postprocess none`：仅用于诊断。

导出时用 128 条随机 `1413` 维输入检查：

- 输出全部有限；
- 输出绝对值不超过 `--action-limit`（默认 `100`）。

sidecar 现在记录：

```json
{
  "action_contract": "raw",
  "action_contract_note": "actions are NOT normalized to [-1,1]; scale by action_scale_robot + default_pose_robot after clipping to clip_actions=100",
  "action_postprocess": "clip",
  "action_range": [-100.0, 100.0]
}
```

### MuJoCo 采集（`scripts/collect_mujoco_d_priv.py`）

- `ACTION_RAW_LIMIT = 100.0 + 1e-3`（原 `ACTION_NORM_LIMIT = 1.0 + 1e-3`）；
- 非有限值直接失败；任一绝对值大于 `100.00...` 直接失败；
- `decode_action_norm` 重命名为 `decode_action_raw`，语义澄清为"输入是未归一化的原始动作"；
- `recorder.append` 写入字段由 `teacher_action_norm` 改为 `teacher_action_raw`。

### 数据集（`src/s10_terrain_perception/d_priv_dataset.py`）

- `SCHEMA_VERSION` 由 `1` 升至 `2`（字段名变更，不兼容旧 shard）；
- `DPrivRecorder` / `validate_d_priv` 全部字段名 `teacher_action_norm` -> `teacher_action_raw`。

### 其他

- `tools/inspect_d_priv.py`：`action_abs_max` 统计改用 `teacher_action_raw`；
- `tests/test_mujoco_privileged_collector.py`：函数名/字段名同步改名，测试全部通过（28 passed）。

## 正确导出命令（方案 B，当前）

```bash
conda activate isaaclab511
cd /home/hashai/Projects/DeepRobotics/goai_embodied_future_material

python tools/export_s10_teacher_onnx.py \
  --checkpoint /home/hashai/Projects/DeepRobotics/rl_training/logs/rsl_rl/deeprobotics_s10_privileged_teacher/2026-08-20_08-56-52/model_43100.pt \
  --output artifacts/teacher_model_43100_1413_raw.onnx \
  --action-postprocess clip \
  --action-limit 100
```

然后先做短采集：

```bash
python3 scripts/collect_mujoco_d_priv.py \
  --teacher-onnx artifacts/teacher_model_43100_1413_raw.onnx \
  --samples 200 \
  --command 0.5 0.0 0.0 \
  --output results/datasets/D_priv_model_43100_raw_smoke.npz \
  --log-every 20

python3 tools/inspect_d_priv.py \
  --require-training-labels \
  results/datasets/D_priv_model_43100_raw_smoke.npz
```

预期看到 `teacher_action_raw` 的绝对值可以明显大于 `1`（例如 `2~8` 量级），且不同关节/左右腿之间
不再被压平成相同的 `±1`；`action_abs_max` 应远小于 `100`（否则说明命中了安全上限，需要复查 checkpoint
或观测契约）。

## 使用边界

1. `±100` 是安全上限截断，不是归一化；下游（学生训练、runner）必须按 `action_scale_robot +
   default_pose_robot` 解释 `teacher_action_raw`，禁止再假设它落在 `[-1,1]` 并二次裁剪；
2. 批量采集 D_priv 前仍必须检查机器人站立、直行、转向和动作饱和率（此处"饱和"指贴近 `±100`，
   而非贴近 `±1`）；
3. 旧字段名 `teacher_action_norm` 和函数名 `decode_action_norm` 已废弃，`SCHEMA_VERSION=2` 之前的
   shard 与当前代码不兼容，禁止混用；
4. 如果要求严格复现 Isaac 教师，仍需要进一步核对 Isaac 与 MuJoCo `1413` 维观测分布是否完全一致，
   本次修复只解决了动作后处理契约问题。

## 历史验证记录（clip[-1,1] 阶段，已废弃，仅供追溯）

- 安全导出 `model_43100.pt` 成功；输入为 `1413`，输出为 `16`。
- ONNX Runtime 对拍最大误差：`1.19e-6`。
- 安全 ONNX 随机推理输出绝对值最大值：`1.0`（因为强制 clip 到 `[-1,1]`，掩盖了真实幅度差异）。
- `--action-postprocess none` 对当前 checkpoint 按预期因超出 `[-1,1]` 被拒绝。
- MuJoCo 采集器回归测试：`5 passed`。
- fake collector smoke：通过。

> 以上记录对应的是已被推翻的 `clip[-1,1]` 方案，保留仅为说明问题演进过程；当前实际生效的是本文档
> 顶部"最新结论（方案 B）"一节。
