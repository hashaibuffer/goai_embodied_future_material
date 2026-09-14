# model99 官方连续台阶复现（2026-09-14）

## 最新显示与性能设置

### NORMAL 无历史跨阶重入修复

之前轮下高度触发只适用于保留 `paused_climb_mode` 的暂停恢复；NORMAL 无记录时仍依赖
前向 Detector，因此漏掉车身下面的台阶。现在 NORMAL 有独立 Low 入口：正向兼容命令、
四轮踏面有效、两前轮均接触且轮心与踏面距离合理、前踏面比至少一个后踏面高 ≥0.04 m，
总高度差小于 Low/High 分界，连续至少三帧后进入 LOW_STEP_SEQUENCE。
不要求暂停历史或前方目标。平地、下阶、零命令、后退、转向、无前轮接触或缺失踏面不会触发；
这仍是踏面高度启发式，不是通用坡道/台阶分类器。

以真实前轮踏面 Z 锁定后轮完成高度，以机身当前位置作为进度参考点（不伪称检测到物理棱边）。
终端和 JSONL 的 `last_entry_reason` / `entry` 区分 `normal_treads`、`paused_treads`、
`forward_detector`。原 play 命令无需新增参数。诊断可用 `--forget-router-on-resume`
在恢复前人为清空 Router 历史，仅用于验证新入口，不用于正常部署。

已移除全部高度图点绘制。旧 `--height-debug-layer` 和 `--actor-height-debug-vis` 参数
仍可解析，但只提示已停用，不会绘制任何高度点；下文彩色点说明仅为历史诊断记录。
Detector 走廊/净空线仍可使用。`--viewer-hz 30` 限制界面同步和 overlay 重建频率，
不降低策略、物理或射线更新频率；`--actor-threads 1` 避免多个 ONNX 会话默认线程池争抢 CPU。
高度 JSONL 默认不启用；启用后用 `--height-debug-every 100` 每 2 秒仿真时间采样一次，
不再随 `--log-every` 状态打印间隔变化。恢复旧采样密度可设为 25。

轻量 play 命令（省去高度可视化及大体积 JSONL）：

```bash
python scripts/play_mujoco_teacher.py \
  --xml models/mjcf/S10_track_lidar.xml \
  --router-bundle artifacts/s10_teacher_router/teacher_router_low_command_events_model99.json \
  --start 33.165 15.18 2.09 1.500695 --vx-limit 1.0 \
  --low-support-surface --low-height-corridor-half-width .25 --low-height-x-range -.4 1.2 \
  --viewer-hz 30 --actor-threads 1 --real-time
```

官方地图 100 步 headless cProfile 采样中，原始高度扫描累计 2.38 s、分层选面累计 2.56 s，
二者约 49 ms/策略步（包含 profiler 开销）；物理积分累计约 0.23 s。说明瓶颈不全在渲染。
本次没有为提速减少射线、降采样或改变选面逻辑；也未以 headless 结果宣称 GUI FPS 提升数值。

已实现可开关的分层表面选择与 XY clamp，并在官方地图、真实扫描、完整 Router 下
完成两种起步位姿的 headless 到顶检查。未重新训练、未更改地图碰撞体或注入理想观测。
这不是重复试验成功率，也不包含途中停稳/重新出发或 normal 1 m/s 验收。

## 当前修复与验证

三层 debug 分别记录原始物理命中（橙色）、选择后的物理表面（绿色）、最终 Actor
有效高度（粉色）。最后一层包含边缘延拓和高度编码，不能把它当作真实射线命中。
JSONL 同时记录世界 XY、两组 geom ID、转换误差和发生表面替换的网格数量。

官方居中起步 t=2.0 s，同一采样位置 `(33.4301,16.8478)` 的原始命中 Z=3.59077，
所选台阶 Z=1.94077，相差 **1.65 m**：确实命中了上层楼板，而非把正常台阶坐标转换错。
坐标转换仍为 `height = base_z - hit_z - 0.5`；分层场景与整体 Z 平移测试验证其一致性。

修复从机身下方支撑面出发，逐层投射，仅接受朝上的、局部高度连续且相邻连线无遮挡的表面。
额外检测候选点是否藏在实体内部，避免把墙底下的地面误当作自由空间；未找到连通面时保留
原始障碍。Detector 使用同一选面结果，其净空射线不变；Low Actor 使用选面后的 XY
延拓高度图。最新适配中 NORMAL、LOW、HIGH 共用这套 Z 选面与 XY 延拓，范围外真实
Z 不直接进入这三个 Actor；只有 RECOVERY 输入保持原样。原始 41×33 网格形状、排列和归一化均不变。

当前候选范围为机身 yaw 坐标 **X=[-0.4,1.2] m，Y=±0.25 m**。范围外复制最近边界值，
非网格对齐边界采用线性插值，不缩放网格。仅对被 Actor 和 Detector 消费的网格做额外
分层投射。最大相邻台阶差为 0.18 m，单射线最多 8 次交点；不是任意地形适用性保证。

| 最终 headless 检查 | 到顶 | 仿真时间 | 最终机身 Z |
| --- | --- | --- | --- |
| 官方地图，完整 Router，居中起步，选面+XY | 是 | 5.86 s | 3.031 m |
| 官方地图，完整 Router，waypoint 25 起步，选面+XY | 是 | 5.98 s | 3.033 m |
| 开阔 14 级楼梯，完整 Router，选面+XY | 是 | 5.82 s | 3.079 m |
| 官方地图，直接 Low，仅 XY、不修选面 | 否，跌落 | 11.98 s | 1.802 m |

结果文件为 `results/model99_stair_diagnosis/verified_{router_center,router_wp25,open,xyonly}.json`。
到顶判据为 `base_y >= 21.375 && base_z >= 3.0`，不是停稳判据。waypoint 起步终点存在
约 0.22 m 横向偏移，停止瞬间一轮未接触，仍需人工验收平台停稳与路线衔接。
额外消融中只修 Z 仍卡阶；组合修复的 X 前界 0.8 m 和后界 -0.6 m 也各通过一次直接 Low
检查，因此推荐范围是已验证候选，而非唯一最优范围。

### 手动复核命令

在 GoAI 仓库根目录执行；所有新处理默认关闭，显式加参数启用：

```bash
python scripts/play_mujoco_teacher.py \
  --xml models/mjcf/S10_track_lidar.xml \
  --router-bundle artifacts/s10_teacher_router/teacher_router_low_command_events_model99.json \
  --start 33.165 15.18 2.09 1.500695 \
  --vx-limit 0.4 \
  --low-support-surface \
  --low-height-corridor-half-width 0.25 \
  --low-height-x-range -0.4 1.2 \
  --detector-debug-vis --height-debug-layer all \
  --height-debug-log results/model99_height_debug.jsonl \
  --real-time
```

键盘前进；图层太密时将 `all` 改为 `raw`、`selected` 或 `actor`。日志按 `--log-every`
采样并追加到文件。去掉 `--start` 从默认地图起点开始。`--vx-limit 0.4` 是本次 Low 对照
速度，不是 normal 1 m/s 测试设置。

### Headless 复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/diagnose_s10_low_stairs.py \
  --router --support-surface --half-width 0.25 --x-range -0.4 1.2 \
  --start-x 33.165 --yaw 1.500695 --steps 600 \
  --output results/model99_stair_diagnosis/recheck_router_wp25.json
```

## 修复前的诊断记录

### 后续最小 Router 适配与 NORMAL 回归

`--low-height-corridor-half-width` / `--low-height-x-range` 保留旧参数名兼容命令，
现在同时控制 NORMAL、LOW、HIGH 的 XY 延拓；`--low-support-surface` 也同时对三者
启用 Z 选面。默认仍关闭，NORMAL/HIGH 命令不受 Low 0.6 m/s 限速。

Router 新增四轮正下方的物理踏面射线（FL、FR、HL、HR），从轮心向下投射，拒绝底面、
无命中及超过 0.3 m 的远处表面。零命令仍交给 Recovery，但保存尚未完成的爬阶目标。
重新前进时，若两前轮踏面均比至少一个后轮高 0.04 m、连续确认三帧，则恢复原爬阶模式，
不要求前向 Detector 再发现已经到车身下面的台阶。转向/后退取消恢复；后轮已经同层则不
强制爬阶。完成判据同时核对物理踏面与接触，前后仍跨层时不会仅凭历史轮心记录退出。
此逻辑只恢复曾经进入过的爬阶任务，不凭坡度凭空触发 HIGH。

Headless 检查：官方居中起步、y≥20.5 m 跨阶时停令 1 s，Recovery 后恢复 Low 并到顶
（`router_pause_treads.json`）；最终平台入口停令 1 s 后也到顶，此时后轮已上来，正常由
NORMAL 接续（`router_pause_final_tread.json`）。平地 NORMAL 输入 1 m/s 连续运行 5 s，
后段机身前向速度约 0.966–0.969 m/s，未跌倒（`normal_xy_1mps.json`）。

**未通过的边界情况**：waypoint 25 偏侧起步、跨阶停令 3 s 时，Recovery 自身后滑并失去
部分轮下支撑，随后未能到顶（`router_pause3_wp25.json`）。这是已观察到的停稳能力限制，
不能将当前 Router 修复宣称为所有暂停时长/位姿验收通过；本次不扩展修改 Recovery 观测或补训。

```bash
python scripts/diagnose_s10_low_stairs.py --router --support-surface \
  --half-width .25 --x-range -.4 1.2 --pause-at-y 20.5 --pause-seconds 1 \
  --steps 750 --output results/model99_stair_diagnosis/recheck_pause.json
python scripts/diagnose_s10_low_stairs.py --scene flat --router --support-surface \
  --half-width .25 --x-range -.4 1.2 --command-vx 1 --steps 250 \
  --output results/model99_stair_diagnosis/recheck_normal_1mps.json
```

### 原始对照

固定 model99 ONNX、0.4 m/s 前进命令、MuJoCo 1 ms / 策略 20 ms、默认 PD。
起点统一居中 `(33.6,15.18,2.09)`，朝向 +Y；全程直接运行 Low，排除
Detector、Router 超时、AutoNav 和键盘中断。每组为一次确定性诊断，不是成功率评估。

| 碰撞场景 | Actor 高度观测 | 结果 |
| --- | --- | --- |
| 官方地图 | 原始完整观测 | 约 2.5 s 偏向左侧并卡住，6 s 时 y≈16.05、z≈2.09 |
| 开阔 14 级箱体楼梯，0.075 m / 0.42 m | 原始完整观测 | 约 6 s 到达顶部 |
| 官方地图（物理不变） | 人工合成开阔楼梯观测 | 约 6 s 到达顶部 |

第三组是 oracle 消融：使用已知世界坐标生成理想楼梯高度，绝不能作为部署修复
或比赛通关证据。诊断只检验到顶，不检验停稳；继续前进可能驶出顶部平台。
临时把侧面网格复制为中央列、或限制到中央 ±0.4 m 的实验只能改善前段，仍会卡住。

结论：完整地形观测的差异足以触发失败；不能归因于 12 s 超时、AutoNav 减速，
也不能仅凭 Detector 射线正常认定 Actor 高度观测等价。当时尚未将影响精确隔离到某组网格。
开阔楼梯通过也不证明两个物理引擎完全一致，但证明本部署链路具有爬这类楼梯的能力。

现有 `--detector-debug-vis` 只显示检测走廊、候选棱、落脚面和前向净空射线。
Actor 看的是 41×33、横向 ±1.6 m 的完整高度图，能包含另一侧楼梯和侧面低地。
新增 `--actor-height-debug-vis` 显示全部 1353 个清洗后高度采样：中央 ±0.4 m 为蓝色，
外侧为粉色。它只影响可视化，不改策略输入。显示的是 Actor 有效高度，填补网格不一定是真实表面。

复现（在仓库根目录，使用安装了 mujoco、onnxruntime 的 Python）：

```bash
python scripts/diagnose_s10_low_stairs.py --scene official --observation native --output results/model99_stair_diagnosis/official_native.json
python scripts/diagnose_s10_low_stairs.py --scene open --observation native --output results/model99_stair_diagnosis/open_native.json
python scripts/diagnose_s10_low_stairs.py --scene official --observation synthetic-stairs --output results/model99_stair_diagnosis/official_synthetic-stairs.json
```

JSON 记录命令、位姿、速度和实际 Actor 高度输入。后续修复必须在官方场景、native
观测下验收，并复测原开阔楼梯成功率、途中停稳重启，不能用合成观测替代验收。
仅调整 Router/T6 无法解决本次 Low 持续接管时的观测问题。现已优先完成上述部署端修复；
若后续多路线验收仍有失败，再考虑训练端同步选面与 XY 观测处理，不应直接启动补训。
