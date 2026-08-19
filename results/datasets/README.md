# TD 官方教师数据集

本目录保存 `training/distillation/collect.py` 生成的、无需 pickle 的压缩 NPZ。
大于 `chunk_size` 的采集会写成 `*_part0000.npz`、`*_part0001.npz`。

核心字段：

- `obs_student [N,441]`：57 维本体观测使用 `cmd_raw`，后接 384 维 LiDAR 高度图。
- `obs_teacher [N,57]`：官方教师实际推理观测，命令槽使用 `cmd_terrain`。
- `action_teacher [N,16]`：同一次官方 ONNX 推理的原始动作，也是实际控制标签。
- `cmd_raw/cmd_terrain [N,3]`：用于审计地形改写是否生效。
- `wp_id/episode_id/success/pre_failure/failure_code`：来自复用的 TA collect 链路。
- `focus_segment`：该样本是否属于 TA `fail_segments.md` 指定的重点路段。
- `post_teleport`：TA 使用原 checkpoint 传送后的 5 秒正常续跑样本。
- `contrast_label`：0 普通、1 失败前、2 同路段成功对照。

传送后自动进入 waypoint 的样本不会被标成成功对照；成功对照必须来自未处于 `post_teleport` 窗口的真实通过轨迹。
- `metadata_json`：Git 提交、随机种子、官方模型和配置 SHA-256。

跨平台检查：

```bash
python3 training/distillation/validate_dataset.py results/datasets/<file>.npz
```

跨多个数据块验收 TD 完成定义：

```bash
python3 training/distillation/dataset_coverage.py results/datasets/*.npz
```

该命令直接读取 `results/fail_segments.md`。任何重点 waypoint 缺少失败前样本或成功对照时返回非零状态，TD 不得标记为 `done`。

检查器使用 `np.load(..., allow_pickle=False)`，可在 Windows 和 Jazzy 机器运行。
