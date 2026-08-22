# FullRoute V1 可复现采集

本流程把本轮“全部重新采集”的方案冻结为配置文件
`configs/collect/fullroute_v1.json`。它不会读取或混入旧的 turning_v2、
turning_v3 数据，默认输出到独立目录
`results/datasets/fullroute_v1_20260822/`。

## 固定契约

- 教师输入/输出：`1413 -> 16`，学生输入：`441`。
- 正式标签每个 policy frame 都重新做 privileged height scan，即
  `scan-every=1`，50 Hz。
- 官方航点接受半径固定为 0.20 m。
- 官方连续路段使用 `official_route_v1` 控制律；急弯前 1.2 m 内减速，
  其余路段按配置中的 0.6–1.0 m/s 运行。
- 每个 shard 记录配置哈希、教师 SHA-256、代码提交、XML 哈希和确定性 seed。
- 默认只是 dry-run；只有显式加入 `--execute` 才会启动 MuJoCo 和写数据。
- 已存在的 shard 不会被覆盖；`--resume` 只跳过同时存在数据和 summary 的 shard。

## 配额

| 阶段 | 类别 | 样本 |
|---|---:|---:|
| train | 平地 | 30k |
| train | 坡道 | 20k |
| train | 台阶 | 30k |
| train | 转向 | 25k |
| train | 官方分段 | 25k |
| train | 恢复 | 10k |
| validation | 全类别 | 12k |
| test | 全类别 | 8k |
| DAgger R1/R2/R3 | 学生访问状态 | 30k / 25k / 18k |

台阶 30k 进一步固定为：普通 4k、WP17–18 6k、WP22–23 5k、
WP25–26 6k、WP27–28 9k。

## 1. 环境与模型校验

在仓库根目录执行：

```bash
python3 -c "import mujoco, numpy, onnxruntime; print('dependencies OK')"
sha256sum artifacts/teacher_c4_model_1700_1413_raw100.onnx
python3 -m py_compile scripts/collect_mujoco_d_priv.py \
  scripts/collect_fullroute_v1.py scripts/audit_fullroute_v1.py
```

如果换教师，后续所有正式 shard 必须全部使用同一个教师文件。不能在一次
数据集中混合教师哈希。

## 2. 必做 dry-run

```bash
python3 scripts/collect_fullroute_v1.py \
  --teacher artifacts/teacher_c4_model_1700_1413_raw100.onnx
```

预期输出：train=140000、validation=12000、test=8000，总计 160000。
这一步不启动仿真，也不创建数据目录。

## 3. 教师 33 点门槛

当前 C4 教师尚未被证明能连续跑完 33 点，因此不能直接开始正式标签采集。
先单独采集一次完整路线 probe：

```bash
python3 scripts/collect_fullroute_v1.py \
  --teacher artifacts/teacher_c4_model_1700_1413_raw100.onnx \
  --phase probe --workers 1 --allow-ungated-probe --execute
```

probe 完成后检查日志中没有塌陷、长时间停滞或倒退，并找到输出 shard 对应的
`.npz.summary.json`。只有 summary 同时满足以下条件，才会被正式采集器接受：

- `route_complete=true`
- `waypoint_range=[0,32]`
- `teacher_scan_every=1`
- `waypoint_reach_radius_m=0.20`
- 教师 SHA-256 与正式采集使用的教师一致

例如：

```bash
find results/datasets/fullroute_v1_20260822/probe \
  -name '*.npz.summary.json' -print
```

若 probe 未通过，先修教师或导航并更换 `dataset_id`，不要用失败教师生成正式
标签，也不要手工修改 gate summary。

## 4. 正式采集

将下面的 `GATE_JSON` 替换为上一步真实 summary 路径：

```bash
python3 scripts/collect_fullroute_v1.py \
  --teacher artifacts/teacher_c4_model_1700_1413_raw100.onnx \
  --gate-report GATE_JSON --workers 1 --execute
```

本设备建议先用 `--workers 1`；确认 CPU、内存和仿真实时性稳定后最多升到 2。
并行数只影响墙钟时间，不改变每个 shard 的 seed、采样频率或标签。

中断后使用完全相同的命令并加 `--resume`。不要删除 plan 或 schedule；它们是
复现实验的一部分。

## 5. 审计与训练清单

采集结束后，对本次 plan 运行：

```bash
python3 scripts/audit_fullroute_v1.py \
  results/datasets/fullroute_v1_20260822/collection_plan__test_train_validation.json
```

审计会检查 schema、样本数、教师哈希、NaN/Inf、塌陷、动作越界、前进命令下
明显倒退/长时间停滞、路线完成状态、0.20 m 判定和 scan-every=1。任何 shard
失败时命令返回非零；生成的 manifest 只列出通过项。训练程序只能读取 manifest
中的 `accepted_shards`，不能直接 glob 整个数据目录。

## 6. DAgger

每轮先冻结该轮学生 ONNX，再用它驱动状态访问，教师仍对每一帧提供 privileged
标签：

```bash
python3 scripts/collect_fullroute_v1.py \
  --teacher artifacts/teacher_c4_model_1700_1413_raw100.onnx \
  --student PATH_TO_FROZEN_STUDENT.onnx \
  --phase dagger_r1 --gate-report GATE_JSON --workers 1 --execute
```

R2、R3 分别把 phase 改为 `dagger_r2`、`dagger_r3`，并使用上一轮训练后冻结
的新学生。每轮单独审计、单独保存 plan/manifest；不要把失败 shard 或旧学生访问
数据自动并入下一轮。

## 复现所需文件

向另一台训练机移交时至少包含：

- `configs/collect/fullroute_v1.json`
- `scripts/collect_mujoco_d_priv.py`
- `scripts/collect_fullroute_v1.py`
- `scripts/audit_fullroute_v1.py`
- 本次教师 ONNX 及其 SHA-256
- 输出目录中的 collection plan、`_schedules/`、`_logs/`、summary 和 manifest
- 本次 Git commit ID，以及 plan 中记录的 XML/config 哈希

只要这些哈希、依赖版本和命令一致，任务展开顺序、每个 episode id、seed、命令
schedule 和输出文件名就是确定的。
