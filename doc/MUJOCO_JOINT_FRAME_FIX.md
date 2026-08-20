# MuJoCo 教师关节坐标系错误变换故障说明

## 结论

`mujoco_teacher.py` 此前对 MuJoCo `data.qpos`/`data.qvel` 错误套用了
`POS_OFFSET_RAD` / `JOINT_DIR`，这套变换实际只属于**真实机器人电机编码器**
相对 URDF 零点的标定偏移（参见
`src/S10_sdk_deploy/interface/robot/hardware/s10_interface.hpp`），与 MJCF
仿真无关。MJCF 关节零点定义与 Isaac Lab/URDF 一致，`qpos`/`qvel` 本身
就是 policy(published) 空间的关节角/角速度，不需要任何变换。

这是"真实 ONNX 教师采集约 11 步即倒地、fake smoke 却能通过"故障的**根因**。

## 故障现象

- fake smoke（零动作）能跑完，因为零动作不触发观测/动作转换的数值分歧；
- 真实 `model_43100.pt` 导出的（安全 clip）ONNX 教师，采集约 11 个控制周期后机器人倒地；
- 动作在第一步就已经饱和在 `[-1, 1]`；
- 排查发现：起立完成、机器人物理姿态完全正常（`base_z=0.407`）时，按旧公式
  算出的 `joint_pos` 策略观测相对 `DEFAULT_POLICY` 偏差最大达到 **2.7 rad**，
  属于严重分布外（OOD）输入。

## 错误变换 vs 正确变换

`s10_interface.hpp`（真机硬件驱动，权威来源）里两个方向的转换互为逆运算：

```cpp
// 写入（下发目标 published -> 电机编码器 raw）
raw = (published - offset) * dir;
// 读取（电机编码器 raw -> published，见 dds_interface.hpp Handler）
published = raw * dir + offset;
```

而 `mujoco_teacher.py`（连同 `mujoco_simulation_ros2.py`）此前用的是：

```python
# 写入（错误）
raw = published * dir + offset
# 读取（错误）
published = (raw - offset) * dir
```

这套公式只有在把 MuJoCo `qpos` 当作"电机编码器 raw 值"时才有意义，但 MJCF
零点本身已经等同于 published 空间，不应再叠加这层变换。用
`GetHipYPosByHeight`/`GetKneePosByHeight`（IK 直接算出的物理关节角）与
`DEFAULT_ROBOT=[0,-0.3,0.6,0,...]` 数值核对一致，进一步证明 `DEFAULT_ROBOT`
本身已经是 published/policy 空间，而非 raw 空间。

## 已实施修复

`src/s10_terrain_perception/mujoco_teacher.py`：

1. `state_from_mujoco()`：观测侧不再对 `qpos[7:23]`/`qvel[6:22]` 做
   `(raw - POS_OFFSET_RAD) * JOINT_DIR` 变换，直接使用原始值。
2. `published_targets_to_raw()`：动作侧不再对策略解码出的目标关节角/角速度做
   `* JOINT_DIR + POS_OFFSET_RAD` 变换，直接透传给 MuJoCo PD 控制器。

`POS_OFFSET_RAD` / `JOINT_DIR` 常量本身保留在文件中（仍可能被其它真机相关
代码路径引用），但两处观测/动作转换函数不再使用它们。

## 验证记录

用本地 `model_43100.pt` 重新导出的安全（`clip[-1,1]`）ONNX 教师，修复后跑
200 条真实教师 rollout：

- `base_z` 全程稳定在 `0.39～0.41m`（正常站立高度），无早停；
- 200 条样本全部采集完成，`labels_usable: true`；
- 动作仍有约 48% 维度饱和在 `±1`（主要是轮速通道，`action_scale=5.0`，
  正常工作范围本就贴近限幅），不影响站立与行走稳定性。

对照修复前：相同 ONNX、相同起立逻辑，约 11 步后机器人倒地。

## 使用边界 / 后续建议

1. 本修复解决的是 MuJoCo 采集侧坐标系 bug，不改变 Isaac 训练侧任何逻辑；
2. 建议后续批量采集前，仍按 `MUJOCO_ACTION_NORMALIZATION_FIX.md` 的流程
   检查动作饱和率、站立/直行/转向表现；
3. `results/datasets/D_priv_model_43100_smoke.npz`（11 条、倒地前旧数据）
   已确认不可用，不纳入版本库，需要重新采集。
