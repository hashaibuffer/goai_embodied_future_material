# TD 官方教师数据集

本目录保存 `training/distillation/collect.py` 生成的、无需 pickle 的压缩 NPZ。
大于 `chunk_size` 的采集会写成 `*_part0000.npz`、`*_part0001.npz`。

核心字段：

- `obs_student [N,441]`：57 维本体观测使用 `cmd_raw`，后接 384 维 LiDAR 高度图。
- `obs_teacher [N,57]`：官方教师实际推理观测，命令槽使用 `cmd_terrain`。
- `action_teacher [N,16]`：同一次官方 ONNX 推理的原始动作，也是实际控制标签。
- `cmd_raw/cmd_terrain [N,3]`：用于审计地形改写是否生效。
- `wp_id/episode_id/success/pre_failure/failure_code`：来自复用的 TA collect 链路。
- `metadata_json`：Git 提交、随机种子、官方模型和配置 SHA-256。

跨平台检查：

```bash
python3 training/distillation/validate_dataset.py results/datasets/<file>.npz
```

检查器使用 `np.load(..., allow_pickle=False)`，可在 Windows 和 Jazzy 机器运行。
