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
label[16]         = teacher ONNX 输出的归一化 a_norm，禁止提前 decode
```

教师高度扫描严格复现 Isaac Lab：

- yaw-only 网格，`x=[-0.8,3.2]`、`y=[-1.6,1.6]`、分辨率 `0.1 m`；包含端点，`41x33=1353`。
- 展平顺序是 y 外层、x 内层；索引 0 为 `(-0.8,-1.6)`，索引 1 为 `(-0.7,-1.6)`。
- 每个网格点从 `base_z+20 m` 向世界 `-Z` 发射独立射线，只命中静态赛道，排除机器人和 overlay。
- 输入值为 `clip(base_z-hit_z-0.5,-1,1)`；20 m 只用于射线起点，不进入数值。
- 教师推理不加训练随机噪声。学生 384 维必须来自比赛 `4344` 线 MuJoCo LiDAR 和 `heightmap.py`，绝不能复制教师真值。

控制周期：MuJoCo `0.001 s`，教师 `0.02 s`，学生 LiDAR `20 Hz`。动作顺序和 scale 与官方 runner 相同，decode 只在仿真控制器执行一次。

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
  --output artifacts/teacher_model_N_1413.onnx
```

导出器会强制检查：

- actor 输入必须是 `1413`；
- actor 输出必须是 `16`；
- tensor 名必须导出为 `obs` / `actions`；
- opset 17、batch 动态；
- 若环境有 `onnxruntime`，自动完成 3 条 probe 的 Torch/ONNX 数值对拍；
- 同目录生成 `teacher_model_N_1413.onnx.json`，包含 PT/ONNX SHA256 和对拍最大误差。

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
- `teacher_action_norm: [10,16]`；
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
4. base 没有快速下坠或爆飞。
5. 用同一 ONNX 做一次短 GUI/官方仿真回放，确认站立、前进和转向至少与 Isaac 基本一致；不通过时禁止批采。

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
| `teacher_action_norm` | `float32 [N,16]` | 未 decode 的教师动作 |
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

TE 只读取 `student_obs` 和 `teacher_action_norm` 训练 `441->16`；其他字段用于分层采样、失败定位和防止混入假数据。交付前对每个 shard 执行 `tools/inspect_d_priv.py --require-training-labels FILE.npz`；该选项会拒绝 fake source 和 `labels_usable=false`。

## 8. 性能与故障处理

本机测得官方 2329 geoms 上 1353 条教师竖直射线约 `15.5 ms/帧`，全部在 CPU。比赛 LiDAR 的 4344 条 `mj_multiRay` 以 20 Hz 更新，通常是主要瓶颈；采集器无 ROS、无 RViz、无实时 sleep，可直接跑到机器允许的速度。

- `No module named onnxruntime`：`python3 -m pip install onnxruntime`。
- 教师输入不是 1413：拿错模型，停止采集。
- hit fraction 低：出生点在赛道外、mesh/group 配置错误或竖直射线没有地面。
- 样本数不足：机器人跌落触发 `--stop-base-z`；先修闭环，不要关闭保护硬采。
- 模型在 Isaac 能走、MuJoCo 立即摔：这是 sim-to-sim 失败，不能把动作当强标签；先检查关节方向、默认位姿、控制周期和动作是否重复 decode。
- 只想测接口：必须使用 `--fake-policy`，并确保产物路径含 `FAKE`；其 metadata 会标记不可训练。

## 9. 比赛合规边界

1353 维 mesh 真值只进入离线教师 ONNX，绝不进入 `student_obs`、学生 ONNX或比赛 runner。比赛运行时仍只有 TB 的 MuJoCo 模拟 LiDAR生成 `16x12x2` 高度图，最终学生直接输出 16 维动作。
