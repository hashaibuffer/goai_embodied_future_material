# MuJoCo 教师动作归一化故障说明

## 结论

`model_43100.pt` 不能把 actor 的最后一层输出直接当作 MuJoCo 的 `a_norm`。
该 checkpoint 的 Gaussian deterministic 输出是未约束的策略均值；MuJoCo 再把它当作 `[-1,1]` 归一化动作乘以关节动作 scale，会生成极端目标，导致 S10 快速跌倒。

典型后果：

- `a_norm ≈ 11`；
- 轮速目标约为 `55 rad/s`；
- 腿关节目标偏移约为 `2～3 rad`。

## Isaac 侧实际行为

当前 S10 训练配置和 RSL-RL 5.4.1 的动作链路是：

```text
actor(obs)
  -> Gaussian deterministic mean（原始 MLP 输出）
  -> Isaac 环境入口 clip_actions=100
  -> 动作项乘 action scale
  -> 关节/轮子控制器
```

当前代码没有 `tanh`，也没有 `[-1,1]` 的 Isaac 动作截断。也就是说，`clip[-1,1]` 不是当前 Isaac 训练动作后处理的复现，而是 MuJoCo `a_norm` 协议所需的安全适配层。

## 已实施修复

### ONNX 导出

`tools/export_s10_teacher_onnx.py` 现在支持：

- `--action-postprocess clip`：在 ONNX 图内执行对称 clip；
- `--action-postprocess tanh`：在 ONNX 图内执行 `tanh`；
- `--action-postprocess none`：仅用于诊断，不适合当前 checkpoint 采集。

导出默认使用 `clip[-1,1]`，并在写入 ONNX 前用 128 条随机 `1413` 维输入检查：

- 输出全部有限；
- 输出绝对值不超过 `1.00001`。

检查失败时禁止生成可用于采集的模型。sidecar 会记录：

```json
{
  "action_contract": "a_norm",
  "action_contract_adapter": true,
  "action_postprocess": "clip",
  "action_range": [-1.0, 1.0]
}
```

### MuJoCo 采集

`collect_mujoco_d_priv.py` 在每个教师推理周期检查 16 维动作：

- 非有限值直接失败；
- 任一绝对值大于 `1.00001` 直接失败；
- 错误信息会提示重新导出，而不是静默截断后继续采集。

这样可以防止未适配的 raw Gaussian ONNX 写入错误标签或把机器人打倒。

## 正确导出命令

```bash
conda activate isaaclab511
cd /home/hashai/Projects/DeepRobotics/goai_embodied_future_material

python tools/export_s10_teacher_onnx.py \
  --checkpoint /home/hashai/Projects/DeepRobotics/rl_training/logs/rsl_rl/deeprobotics_s10_privileged_teacher/2026-08-20_08-56-52/model_43100.pt \
  --output artifacts/teacher_model_43100_1413.onnx \
  --action-postprocess clip \
  --action-limit 1
```

然后先做短采集：

```bash
python3 scripts/collect_mujoco_d_priv.py \
  --teacher-onnx artifacts/teacher_model_43100_1413.onnx \
  --samples 200 \
  --command 0.5 0.0 0.0 \
  --output results/datasets/D_priv_model_43100_smoke.npz \
  --log-every 20

python3 tools/inspect_d_priv.py \
  --require-training-labels \
  results/datasets/D_priv_model_43100_smoke.npz
```

## 验证记录

- 安全导出 `model_43100.pt` 成功；输入为 `1413`，输出为 `16`。
- ONNX Runtime 对拍最大误差：`1.19e-6`。
- 安全 ONNX 随机推理输出绝对值最大值：`1.0`。
- `--action-postprocess none` 对当前 checkpoint 按预期因超出 `[-1,1]` 被拒绝。
- MuJoCo 采集器回归测试：`5 passed`。
- fake collector smoke：通过。

## 使用边界

当前 `clip[-1,1]` 解决的是 MuJoCo 控制协议和安全性问题，不等于 Isaac 训练侧的 `clip_actions=100`。因此：

1. 该 ONNX 可以用于继续做 MuJoCo 冒烟和短闭环验证；
2. 批量采集 D_priv 前必须检查机器人站立、直行、转向和动作饱和率；
3. 如果要求严格复现 Isaac 教师，应进一步对齐 Isaac 与 MuJoCo 的 `1413` 维观测分布，或重新训练明确采用 `[-1,1]` 动作契约的教师；
4. 禁止把旧的 raw ONNX 直接交给 MuJoCo 控制器。
