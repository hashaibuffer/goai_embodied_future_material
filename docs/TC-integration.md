# TC 自研策略集成

TC 的 runner、配置、占位模型、测试和验收脚本现已直接维护在本仓库中。`s10-terrain-aware-policy` 不再是编译或运行依赖。

## 目录

- `src/s10_terrain_policy/`：441 维观测和 16 维动作的自研 runner。
- `configs/policy.yaml`：冻结的策略 I/O、关节顺序和 PD 参数。
- `models/terrain_locomotion.onnx`：当前占位模型；教师/蒸馏模型完成后原位替换。
- `scripts/tc_joint_cmd_smoke.py`：ROS 2 `/JOINTS_CMD` 端到端烟测。
- `tests/test_tc_runner.py`：runner 合约和 SDK 集成测试。

## 构建

```bash
cd ~/goai_embodied_future_material
source /opt/ros/jazzy/setup.bash
colcon build --packages-select drdds s10_sdk_deploy
```

无需设置 `S10_TERRAIN_POLICY_ROOT`。默认模型由仓库内的 `models/terrain_locomotion.onnx` 提供，也可以通过 `--model-path` 显式覆盖。

## 运行

```bash
source install/setup.bash
ros2 run s10_sdk_deploy rl_deploy \
  --controller proprio_clone \
  --model-path "$PWD/models/terrain_locomotion.onnx"
```

`proprio_clone` 强制高度图 validity 为 0；`learned` 消费 `/S10_HEIGHTMAP`。真模型尚未替换占位模型前，只用于验证加载、观测和关节命令链路。
