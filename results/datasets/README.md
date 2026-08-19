# TD 官方教师数据集

本目录保存 `training/distillation/collect.py` 生成的官方 ONNX 教师采集数据。TD 只覆盖平地/缓坡路径；台阶、急升降和末段复杂地形交由 TH 的 privileged teacher（D_priv）处理。数据不使用 pickle，采用压缩 NPZ，超过 `chunk_size` 时拆分为 `*_partNNNN.npz`。

## 正式数据

- `20260819_final_seed0_part0000.npz` … `part0005.npz`：25,910 条，6 个分块。
- `20260819_final_seed1_part0000.npz` … `part0004.npz`：24,502 条，5 个分块。
- 正式合计：**50,412 条**。此前 `20260819_seed0` … `seed4` 的旧参数采集为历史数据，不计入本正式合计，也不要与正式集混合后重复计数。
- 标签统计（正式合计）：`pre_failure=8,636`，`success=15,410`，`post_teleport=4,357`；`teacher_source` 全部为 `official`（编码 0）。

## 字段与训练契约

- `obs_student [N,441]`：57 维本体观测 + 384 维高度图（16×12×2）。
- `obs_teacher [N,57]`：官方教师实际推理输入，命令槽使用平滑后的 `cmd_terrain`。
- `action_teacher [N,16]`：同一次官方 ONNX 推理得到的 16 维动作标签。
- `cmd_raw/cmd_terrain [N,3]`：原始导航命令与地形规则改写后的命令。命令裁剪为 `vx∈[-1,1]`、`vy∈[-0.6,0.6]`、`wz∈[-1,1]`。
- `wp_id/next_wp_id/episode_id/success/pre_failure/failure_code`：复用 TA collect 的导航、卡死和传送记录；`next_wp_id` 覆盖 0–33（航点 0–32 及终点 33）。
- `post_teleport`：checkpoint 传送后的续跑窗口。传送样本不计为成功对照。
- `contrast_label`：0=普通样本，1=失败前样本，2=同路段未经传送的真实成功对照。
- `metadata_json`：采集 Git commit、随机种子、模型路径/哈希和配置哈希，保证可追溯。

## 路段覆盖与分工

正式数据覆盖完整 TA 航线目标 0–32，并包含失败前样本与真实成功对照。以下复杂路段只保留路线覆盖和交接记录，不作为 TD 的教师规则训练目标：

- 台阶/高差及复杂段：6→7、15→16、16→17、17→18、20→21、22→23、23→24、27→28。
- 末段迷宫：航点 28–32；运行时避墙由 Issue #5 的 AutoNav LiDAR 局部规划处理，不在 TD 中训练第二个网络。
- `results/fail_segments.md` 中保留 TA 的历史失败清单；TD 对上述 TH 段不以“失败段必须成功对照”作为放行条件，困难地形成功样本由 TH 单独采集。

## 雷达/高度图参数

参数来源分别为 `configs/lidar.yaml`、`configs/heightmap.yaml`：

- 高度图话题 `/S10_HEIGHTMAP`，坐标系 `robot_horizontal`；范围 `x=[-0.8,3.2] m`、`y=[-1.5,1.5] m`，分辨率 `0.10 m`，原始网格 40×30，下采样策略网格 16×12。
- 高度图两通道：裁剪/归一化高度（范围 `[-0.40,0.80] m`，除数 0.80）与 validity；未知格 validity=0；共 384 个展平值。
- LiDAR 话题 `/S10_SIM_LIDAR`，`lidar_front_site`，量程 0.10–6.00 m；水平 FOV 180°/181 rays，垂直 FOV 55°/24 rays，更新 20 Hz；噪声标准差 0.01 m，dropout 概率 0.03。

## 验证

```bash
python3 training/distillation/validate_dataset.py results/datasets/20260819_final_seed0_part0000.npz
python3 training/distillation/dataset_coverage.py results/datasets/20260819_final_seed*.npz
pytest -q
```

检查器使用 `np.load(..., allow_pickle=False)`，可在 Windows 和 ROS 2/Jazzy 环境运行。
