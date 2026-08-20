# S10 MuJoCo 特权教师数据采集交接手册

本文档给接手采集的同事使用。目标是：Isaac Lab 只训练教师；教师 checkpoint 冻结后，全部 rollout、比赛 LiDAR、教师真值扫描和 `D_priv` 落盘都在 CPU MuJoCo 中完成，不占用 Ubuntu 5080。

## 0. 同事拿到 PT 后的最短路径

1. 确认交付包同时包含 `model_N.pt`、`rl_training` commit、训练迭代数和 B2 GUI 验收结论；缺一项先补齐。
2. 在 `isaaclab511` 中按第 3 节导出 ONNX；sidecar 必须显示输入 1413、输出 16、`empirical_normalization=false`，有 ONNX Runtime 时还必须显示 `torch_onnx_parity.checked=true`。
3. 在 CPU 采集机检出本 collector 对应 commit，按第 4 节安装依赖并跑 fake smoke；fake 文件只验证链路，不能训练。
4. 按第 5 节用真 ONNX 采 200 条，执行带 `--require-training-labels` 的检查，再做一次短 GUI 回放。
5. 只有短闭环通过后，才按第 6 节分地形、分 episode 批采；每个 shard 单独检查并连同 sidecar 和验收记录交给 TE。

这五步不需要 Isaac Sim、ROS、RViz 或 CUDA；只有第 2 步需要能读取 PT 的 PyTorch 环境。

## 1. 不可改变的协议

每个样本包含：

```text
teacher_obs[1413] = base_lin_vel_body[3] + official_proprio[57] + privileged_height[1353]
student_obs[441]  = official_proprio[57] + lidar_height[192] + validity[192]
label[16]         = teacher ONNX 输出的原始动作 a_raw，禁止提前 decode
```

教师高度扫描严格复现 Isaac Lab：

- yaw-only 网格，`x=[-0.8,3.2]`、`y=[-1.6,1.6]`、分辨率 `0.1 m`；包含端点，`41x33=1353`。
- 展平顺序是 y 外层、x 内层；索引 0 为 `(-0.8,-1.6)`，索引 1 为 `(-0.7,-1.6)`。
- 每个网格点从 `base_z+20 m` 向世界 `-Z` 发射独立射线，只命中静态赛道，排除机器人和 overlay。
- 输入值为 `clip(base_z-hit_z-0.5,-1,1)`；20 m 只用于射线起点，不进入数值。
- 教师推理不加训练随机噪声。学生 384 维必须来自比赛 `4344` 线 MuJoCo LiDAR 和 `heightmap.py`，绝不能复制教师真值。

控制周期：MuJoCo `0.001 s`，教师 `0.02 s`，学生 LiDAR `20 Hz`。动作顺序和 scale 与官方 runner 相同，decode 只在仿真控制器执行一次。
ONNX 的 `actions` 是教师 actor 的**原始动作**（`a_raw`），对齐 Isaac 训练侧 `clip_actions=100`；**不是** `[-1,1]` 归一化动作。采集器不会为异常大的输出静默截断，超过 `±100` 安全上限会立即失败。

> 2026-08-20 更正：此前版本要求 `a_norm∈[-1,1]` 并在导出/采集时强制 clip 到 `±1`。
> 交叉核对 Isaac `rsl_rl` `vecenv_wrapper.step()`（`clip_actions=100`）、S10 `rough_env_cfg.py`
> 的 `JointPositionActionCfg(scale=..., clip=(-100,100))` 和真机 `terrain_policy_math.hpp::DecodeAction`
> （无 `[-1,1]` 裁剪）后确认：`clip[-1,1]` 不是训练/真机契约的一部分，而是采集链路里凭空加上去的
> 错误约束，会把左右腿等幅度差异（例如台阶动作里 knee 4.5 vs 3.3）削平成相同的 `±1`，导致学生
> 学不到正确的差异化动作。现在字段名统一改为 `teacher_action_raw`，裁剪上限对齐 `clip_actions=100`。

### 1.1 每个控制周期如何组装 `teacher_obs[1413]`

采集器在每个 `0.02 s` 控制边界使用**同一个 MuJoCo 状态快照**完成以下步骤。不要把不同时间的速度、扫描和动作拼在同一个样本中。

| 切片 | 维度 | 内容 | 处理规则 |
| --- | ---: | --- | --- |
| `obs[0:3]` | 3 | `base_lin_vel_body` | 从 MuJoCo `base_link` 的 body velocity 读取，保持 body frame，不能改成世界系 |
| `obs[3:60]` | 57 | `official_proprio` | 与官方 runner 完全同序：`base_ang_vel*0.25`、投影重力、`cmd_raw`、16 维关节位置、16 维关节速度、上一拍 `a_raw[16]` |
| `obs[60:1413]` | 1353 | `privileged_height` | 41×33 个 yaw-only 垂直射线，按 y 外层、x 内层展平 |

其中 `official_proprio[6:9]` 必须是裁剪后的 `cmd_raw`，不能填 `cmd_terrain`；`official_proprio` 中的 `last_action` 必须是上一拍教师输出的原始动作 `a_raw`（未经 `[-1,1]` 归一化，与 `clip_actions=100` 同一空间），不是关节角、轮速或已经 decode 的目标。

教师高度扫描的单元值严格为：

```text
hit_z = base_z + 20.0 - ray_distance
height[i] = clip(base_z - hit_z - 0.5, -1.0, 1.0)
         = clip(ray_distance - 20.5, -1.0, 1.0)
```

射线起点的 `+20 m` 只是为了覆盖地形，不得把 20 m 作为观测偏置再次加入。只打开静态赛道 geom group，关闭机器人和 overlay group；无命中的单元保持 `-1.0`，并由 `privileged_hit_fraction` 记录质量。

### 1.2 一条样本的完整时序

```text
MuJoCo 状态快照
  -> 读取 base/link、关节状态和上一拍 a_raw
  -> 计算 official_proprio[57]
  -> 发射 1353 条 privileged vertical rays
  -> 拼接 teacher_obs = [3 + 57 + 1353] = 1413
  -> ONNX Runtime 输入 obs[1,1413]，得到 actions[1,16]
  -> 同时保存 student_obs[441] 和 teacher_action_raw[16]
  -> 将本拍 a_raw 更新为下一拍的 last_action
  -> 仅在控制器中 decode 一次，并用 PD 跑 20 个 physics step
  -> 在下一控制边界重复
```

`student_obs[441]` 使用同一状态附近最近一次比赛 LiDAR 编码的 `16×12` height 和 `16×12` validity；它不能由 1353 条 teacher 真值射线重采样得到。教师真值只用于产生动作标签，不能写入学生输入或最终学生 ONNX。

### 1.3 采集前必须确认的模型身份

`export_s10_teacher_onnx.py` 只接受 privileged teacher actor。导出前确认：

- checkpoint 是冻结后的 S10 teacher，不是官方 57 维 policy，也不是学生 441 维网络；
- actor 第一层输入为 `1413`，最后一层输出为 `16`；
- checkpoint 没有需要外置的 observation normalizer，或 normalizer 已被明确封装进导出模型；
- **确认动作后处理**：当前 RSL-RL Gaussian deterministic 输出是 actor 的原始均值，不是 `tanh`。这个原始均值就是 Isaac 训练/真机部署实际使用的动作空间：Isaac `rsl_rl` 入口按 `clip_actions=100` 截断，环境侧 `JointPositionActionCfg` 再乘 `action_scale_robot` 并叠加 `default_pose_robot`；真机 `terrain_policy_math.hpp::DecodeAction` 同样不做 `[-1,1]` 裁剪。因此导出时使用 `--action-postprocess clip --action-limit 100`，只做安全上限截断（对齐 Isaac `clip_actions=100`），**不得**再压缩到 `[-1,1]`，否则会把不同关节间的原始幅度差异（例如台阶动作里 knee 4.5 vs 3.3）削平成相同的边界值，导致学生学不到差异化动作。
- `teacher_action_raw` 是未经 `[-1,1]` 归一化的教师原始动作（仅裁剪到 `±100` 安全上限），后续不得在数据集写入阶段 decode，也不得被下游代码误当作 `[-1,1]` 归一化值再次裁剪。

如果输入维度不是 `1413`，立即停止；不要通过补零、截断或复制高度图“修正”维度。

### 1.4 采集起点必须先起立，不能把坐姿直接喂给教师

比赛真机和官方 `rl_deploy` 都是先跑纯运动学 `StandUpState`（无网络、固定 PD 目标），起立完成后才切入 `RLControlMode` 开始跑策略。采集器必须复刻同一时序，否则第一拍就会把出生姿态 `JOINT_INIT_RAW`（蹲姿，`hipy≈-1.16`、`knee≈2.76`）直接喂给按站姿 `DEFAULT_ROBOT`（`hipy≈-0.3`、`knee≈0.6`）训练的教师：

```text
坐姿观测（偏差最大约 1.7 rad）
  -> 教师 actor 原始输出严重饱和（16 维中 15 维打到 ±1）
  -> clip/tanh 后动作仍然是极端值
  -> 目标关节瞬间跳变最大约 1.9 rad，轮速跳变 ±5 rad/s
  -> base_z 在 7 个控制周期内跌破 stop_base_z=0.08
```

`scripts/collect_mujoco_d_priv.py` 在 `initialize()` 之后、正式采集循环之前，会调用 `mujoco_teacher.run_stand_up()`：

- 起点：`JOINT_INIT_RAW`（与 `initialize()` 写入的坐姿一致）；
- 终点：`mujoco_teacher.stand_up_target_raw()`，其值就是 `DEFAULT_ROBOT` 本身（**不经过** `decode_action_raw`/`published_targets_to_raw` 二次变换——`DEFAULT_ROBOT` 已经是 raw/MJCF 空间的物理关节角，可用 `GetHipYPosByHeight/GetKneePosByHeight` 逆运动学核实：`h=0.48 -> hipy=-0.284, knee=0.568`，与 `DEFAULT_ROBOT` 的 `[-0.3, 0.6]` 吻合；若再套一次 `JOINT_DIR/POS_OFFSET_RAD` 会把目标推出 MJCF 硬限位，例如 hipy 会跳到 `±2.83 rad`，超过 `±2.53 rad` 限位）；
- 时长/增益：`STAND_UP_DURATION_S=3.0s`，腿 `kp=120/kd=2`，轮 `kp=0/kd=0.6`，与 C++ `StandUpState`/`s10_control_parameters.cpp` 一致；
- 三次样条插值位置和速度（复刻 `GetCubicSplinePos/GetCubicSplineVel`）；
- 起立阶段不查询教师、不写任何 `D_priv` 样本，只做纯运动学 PD 收敛。

起立结束后 `base_z` 应稳定在 `0.40` 左右（地面支撑反力下略低于 `stand_height_=0.48`），低于 `0.30` 会打印 warning，提示检查 MJCF/actuator 配置。

**这一步与教师/学生模型训练完全解耦**：`run_stand_up` 只是固定运动学轨迹，不依赖策略权重；真机比赛时同理，起步先走 repo 自带的 `StandUpState`，与用哪个训练版本的教师/学生策略无关。

## 1.5 零动作不代表"能站稳"

用 `--fake-policy`（`ZeroPolicy`，恒定输出零动作）做 smoke 时，`base_z` 会缓慢沉到约 `0.08` 附近并稳定（零动作只给出静态目标 `DEFAULT_ROBOT`，没有任何主动平衡补偿；真实教师策略会持续输出小的修正动作维持平衡，类似人站立时的踝关节微调）。因此：

- fake smoke 的 `base_z` 触底**不代表 StandUp 或采集协议有 bug**，只代表零动作本身无法维持站姿；
- 采集器对 `--fake-policy` 跳过 `--stop-base-z` 提前终止检查，只验证协议格式（维度、dtype、字段），샘플数量会跑满 `--samples`；
- 真正验证"教师能否站稳/行走"必须使用第 5 节的真 ONNX 200 条 smoke，并结合 GUI 回放确认。

如果输入维度不是 `1413`，立即停止；不要通过补零、截断或复制高度图"修正"维度。

## 2. 教师负责人需要交付什么

最小交付包：

1. 验收通过的 `model_N.pt`，不要给仍在训练或“慢滚低坎”的旧 checkpoint。
2. checkpoint 对应的 `rl_training` commit ID、训练迭代数和 B2 GUI 验收结论。
3. 最好同时提供本工具导出的 `teacher_model_N_1413.onnx` 与自动生成的 `.onnx.json`。
4. 文件 SHA256；导出脚本会自动写入 sidecar，采集 NPZ 也会记录 ONNX SHA256。

当前教师配置 `empirical_normalization=False`。如果以后打开 observation normalizer，必须先修改导出器把 normalizer 包进 ONNX；禁止直接沿用当前导出命令。

## 3. 从 PT 导出教师 ONNX

在有 `torch` 和 `onnx` 的机器上执行即可，不启动 Isaac Sim。推荐直接使用既有环境：

```bash
conda activate isaaclab511
cd /path/to/goai_embodied_future_material
python tools/export_s10_teacher_onnx.py \
  --checkpoint /absolute/path/model_N.pt \
  --output artifacts/teacher_model_N_1413.onnx \
  --action-postprocess clip --action-limit 100
```

导出器会强制检查：

- actor 输入必须是 `1413`；
- actor 输出必须是 `16`；
- tensor 名必须导出为 `obs` / `actions`；
- opset 17、batch 动态；
- 在导出前用 128 条随机 `1413` 维输入检查输出有限且绝对值不超过 `--action-limit`（默认 `100`，对齐 Isaac `clip_actions=100`）；
- 若环境有 `onnxruntime`，自动完成 3 条 probe 的 Torch/ONNX 数值对拍；
- 同目录生成 `teacher_model_N_1413.onnx.json`，包含 PT/ONNX SHA256、后处理方式、动作范围（`action_contract: "raw"`）和对拍最大误差。
- `--action-postprocess none` 跳过安全裁剪，仅用于诊断，不得进入采集；`--action-limit` 只是一个安全上限截断（防止异常 checkpoint 输出发散），**不是** `[-1,1]` 归一化——导出的 `actions` 仍是教师原始动作幅度（`teacher_action_raw`），下游必须按原始幅度乘 `action_scale_robot` 解释。

如果出现 `expected 1413->16`，拿到的是普通 57 维策略、错误 checkpoint 或教师观测配置发生了变化，禁止继续采集。

兼容性基线：2026-08-19 已用 RSL-RL 5.4.1 实际 checkpoint（`actor_state_dict/mlp.*`，网络 `1413-256-256-128-16`）完成导出和数值对拍。该结论只证明工具兼容 checkpoint 格式，不代表该 checkpoint 已通过 B2 质量门控。

## 4. Jazzy 5080、ThinkPad Jazzy 与 Windows 11 采集机准备

采集不需要 ROS、Isaac Lab、CUDA、显示器或 GPU。Jazzy 5080、ThinkPad Jazzy 和 Windows 11 使用同一套 Python 脚本和 NPZ schema；只有虚拟环境激活和路径写法不同。

### 4.1 Ubuntu 24.04 / Jazzy 5080 训练机

PT 导出可直接使用已有 `isaaclab511`；为避免改动 Isaac 训练环境，批量采集推荐单独 venv：

```bash
cd /path/to/goai_embodied_future_material
python3 -m venv .venv-collect
source .venv-collect/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<2" pyyaml mujoco onnxruntime pytest
python -m pytest -q tests/test_mujoco_privileged_collector.py \
  tests/test_heightmap.py tests/test_heightmap_step.py tests/test_heightmap_blocked.py
```

若不希望新建 venv，也可在 `isaaclab511` 中安装缺少的 `mujoco pyyaml`；但不要为采集启动 Isaac Sim。headless collector 不创建 OpenGL renderer，通常不需要设置 `MUJOCO_GL`。

### 4.2 ThinkPad / ROS 2 Jazzy（无 Isaac Lab）

ThinkPad 只有 MuJoCo、`goai_embodied_future_material` 和 `s10-terrain-aware-policy` 两个仓库，这是完整支持的采集节点，不要安装 Isaac Lab。业务命令全部在 goai 仓执行；s10 仓只用于核对 T00/TH/TE 契约和 taskcard。

正常交接时，ThinkPad 应收到 `teacher_model_N_1413.onnx`、`.onnx.json`、B2 验收记录和 collector commit，不必收到 PT。使用现有 MuJoCo Python 环境时先补依赖并自检：

```bash
cd ~/Projects/DeepRobotics/goai_embodied_future_material
python3 -m pip install "numpy<2" pyyaml mujoco onnxruntime pytest
python3 -c "import mujoco, onnxruntime, yaml; print('collector deps OK')"
python3 -m pytest -q tests/test_mujoco_privileged_collector.py \
  tests/test_heightmap.py tests/test_heightmap_step.py tests/test_heightmap_blocked.py
```

若同事只拿到 `model_N.pt`，也不需要 Isaac Lab；可在独立 venv 安装 CPU `torch onnx onnxruntime` 后执行第 3 节导出器。为减少 ThinkPad 配置和传输风险，默认仍由训练负责人在 `isaaclab511` 导出并同时交付 ONNX 与 sidecar。

ThinkPad 的 fake smoke、200 条真教师 smoke 和批量采集命令与第 5～6 节完全相同。它可以长期 CPU headless 采集，不占用 5080；不要从 s10 仓寻找 collector，也不要启动 ROS/RViz。

### 4.3 Windows 11 / PowerShell

推荐 64 位 Python 3.11。在 PowerShell 中执行：

```powershell
cd C:\work\goai_embodied_future_material
py -3.11 -m venv .venv-collect
Set-ExecutionPolicy -Scope Process Bypass
.\.venv-collect\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "numpy<2" pyyaml mujoco onnxruntime pytest
python -m pytest -q tests/test_mujoco_privileged_collector.py tests/test_heightmap.py tests/test_heightmap_step.py tests/test_heightmap_blocked.py
```

Windows 也可以从 PT 导出 ONNX，不需要 Isaac Lab：

```powershell
python -m pip install torch onnx
python tools/export_s10_teacher_onnx.py `
  --checkpoint "D:\checkpoints\model_N.pt" `
  --output "artifacts\teacher_model_N_1413.onnx"
```

PowerShell 换行符是反引号 `` ` ``，不是 Bash 的 `\`。含空格的路径必须加双引号；文档里的 `/tmp/FILE.npz` 在 Windows 上改为例如 `C:\temp\FILE.npz`。

### 4.4 三类机器交接规则

- `.pt`、`.onnx`、`.onnx.json` 和 pickle-free `.npz` 可在 Jazzy 5080、ThinkPad Jazzy 和 Windows 11 间直接传递。
- 每台采集机都要记录 collector commit；不同 commit 的 shard 不能不加说明地混合。
- Windows 使用 `onnxruntime` CPU provider，不要为这条链安装 CUDA 版 Runtime。
- 数据回传后再跑一次 `tools/inspect_d_priv.py --require-training-labels FILE.npz`，确认文件未损坏且不是 fake shard。

### 必跑的无教师 smoke

这一步使用零动作假 policy，只验接口，输出绝不能交给学生训练：

```bash
python3 scripts/collect_mujoco_d_priv.py \
  --fake-policy --samples 10 \
  --output /tmp/D_priv_FAKE_SMOKE.npz
python3 tools/inspect_d_priv.py /tmp/D_priv_FAKE_SMOKE.npz
```

必须看到：

- `student_obs: [10,441]`；
- `teacher_action_raw: [10,16]`；
- `teacher_source=fake_mujoco_smoke`；
- `metadata.labels_usable=false`；
- `privileged_hit_fraction` 接近 1.0。

## 5. 拿到教师后的首次闭环 smoke

先采 100～500 条，不要直接跑十万条：

```bash
python3 scripts/collect_mujoco_d_priv.py \
  --teacher-onnx artifacts/teacher_model_N_1413.onnx \
  --samples 200 --command 0.5 0.0 0.0 \
  --output results/datasets/D_priv_model_N_smoke.npz \
  --log-every 20
python3 tools/inspect_d_priv.py \
  --require-training-labels results/datasets/D_priv_model_N_smoke.npz
```

启动时会拒绝错误的 ONNX 名称和维度。每条真数据标记为 `teacher_source=privileged_mujoco`，metadata 中必须有 `labels_usable=true`、teacher SHA256、赛道 XML SHA256 和 collector commit。

首次 smoke 验收：

1. 实际样本数等于请求数，没有因 `base_z<0.08 m` 提前停止。
2. `privileged_hit_fraction.min >= 0.99`；否则检查出生点、赛道 mesh 和 geom group。
3. action 全部有限，且不是全零/常量。
4. action 的绝对值全部不超过 `1.00001`；否则说明拿到的是未适配 ONNX，立即停止。
5. base 没有快速下坠或爆飞。
6. 用同一 ONNX 做一次短 GUI/官方仿真回放，确认站立、前进和转向至少与 Isaac 基本一致；不通过时禁止批采。

## 6. 分困难路段采集

可直接指定出生点 `[x,y,z,yaw(rad)]`，每个路段输出独立 shard：

```bash
python3 scripts/collect_mujoco_d_priv.py \
  --teacher-onnx artifacts/teacher_model_N_1413.onnx \
  --start -15.02 36.50 0.85 1.57 \
  --command 0.6 0.0 0.0 --samples 3000 \
  --terrain-id regular_stairs_up --episode-id 1 \
  --output results/datasets/D_priv_model_N_stairs_up_001.npz
```

复杂操作可用命令表。CSV 每行为 `起始样本,vx,vy,wz`，按样本号升序：

```csv
0,0.60,0.00,0.00
120,0.45,0.00,0.35
220,0.60,0.00,0.00
```

```bash
python3 scripts/collect_mujoco_d_priv.py \
  --teacher-onnx artifacts/teacher_model_N_1413.onnx \
  --start 21.66 29.57 1.34 0.05 \
  --command-schedule configs/collect/high_step.csv \
  --samples 4000 --terrain-id high_step --episode-id 2 \
  --output results/datasets/D_priv_model_N_high_step_002.npz
```

输入命令会冻结裁剪到比赛范围：`vx[-1,1]`、`vy[-0.6,0.6]`、`wz[-1,1]`。同一 checkpoint 的不同地形、不同随机种子应写不同 NPZ，禁止覆盖旧 shard。

## 7. NPZ 字段和下游交付

必需字段：

| 字段 | dtype / shape | 语义 |
| --- | --- | --- |
| `student_obs` | `float32 [N,441]` | 官方 57 + 比赛 LiDAR 高度/mask |
| `teacher_action_raw` | `float32 [N,16]` | 教师原始动作（未 decode，未归一化到 `[-1,1]`，仅裁剪到 `±100`） |
| `command_raw` | `float32 [N,3]` | 学生实际看到的命令，禁止 `cmd_terrain` |
| `teacher_source` | string `[N]` | 真数据必须是 `privileged_mujoco` |
| `waypoint_id` | `int32 [N]` | 最近赛道 waypoint，仅元数据 |
| `terrain_id` | string `[N]` | shard 对应地形/路段 |
| `episode_id`, `step_id` | `int32 [N]` | 时序定位 |
| `base_pose_wxyz` | `float32 [N,7]` | 质检/回放，不进入学生输入 |
| `privileged_hit_fraction` | `float32 [N]` | 教师扫描完整性 |
| `metadata_json` | JSON string | checkpoint/XML/commit/SHA/周期/可用性 |

交给 TE 的文件：

```text
D_priv_model_N_*.npz
teacher_model_N_1413.onnx.json
每个 shard 的 inspect 输出或汇总 JSON
checkpoint 的 B2 验收记录
```

TE 只读取 `student_obs` 和 `teacher_action_raw` 训练 `441->16`；学生需要按教师同样的方式解释标签（乘一次 `action_scale_robot` 再叠加 `default_pose_robot`），不能假设标签已经在 `[-1,1]`。其他字段用于分层采样、失败定位和防止混入假数据。交付前对每个 shard 执行 `tools/inspect_d_priv.py --require-training-labels FILE.npz`；该选项会拒绝 fake source 和 `labels_usable=false`。

## 8. 性能与故障处理

本机测得官方 2329 geoms 上 1353 条教师竖直射线约 `15.5 ms/帧`，全部在 CPU。比赛 LiDAR 的 4344 条 `mj_multiRay` 以 20 Hz 更新，通常是主要瓶颈；采集器无 ROS、无 RViz、无实时 sleep，可直接跑到机器允许的速度。

- `No module named onnxruntime`：`python3 -m pip install onnxruntime`。
- 教师输入不是 1413：拿错模型，停止采集。
- hit fraction 低：出生点在赛道外、mesh/group 配置错误或竖直射线没有地面。
- 样本数不足：机器人跌落触发 `--stop-base-z`；先修闭环，不要关闭保护硬采。
- 首拍 15/16 维动作饱和到 ±1、机器人立即塌陷：坐姿 `JOINT_INIT_RAW` 直接喂给了按站姿训练的教师，产生 OOD 输入。原因通常是 `run_stand_up` 未能有效起立，或者 `stand_up_target_raw()` 因二次坐标变换算出了超出 MJCF 限位（hipy `±2.8 rad` 超过 `±2.53 rad`）的目标。修复方法：确认 `stand_up_target_raw()` 直接返回 `DEFAULT_ROBOT`，不再经过 `published_targets_to_raw`；起立结束后 `base_z` 应在 0.40 左右（低于 0.30 会打印 warning）。
- 模型在 Isaac 能走、MuJoCo 立即摔：这是 sim-to-sim 失败，不能把动作当强标签；先检查关节方向、默认位姿、控制周期和动作是否重复 decode。
- 只想测接口：必须使用 `--fake-policy`，并确保产物路径含 `FAKE`；其 metadata 会标记不可训练。

## 9. 比赛合规边界

1353 维 mesh 真值只进入离线教师 ONNX，绝不进入 `student_obs`、学生 ONNX或比赛 runner。比赛运行时仍只有 TB 的 MuJoCo 模拟 LiDAR生成 `16x12x2` 高度图，最终学生直接输出 16 维动作。
