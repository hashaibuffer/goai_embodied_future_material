# S10四教师Router的MuJoCo整赛道测试

2026-09-17：保留原HIGH model900，HIGH Actor输入使用
`vx_actor=min(vx_user, 0.4)`；1.0映射到0.4 m/s，小指令不放大，零/倒车及vy/wz不变。
Router/Detector及NORMAL预热仍使用原始命令，退出HIGH后NORMAL恢复原始1.0。
bundle字段`high_forward_command_max_mps=0.4`，旧bundle缺字段也默认0.4；生成器同名CLI可配置。
0.4采用保留的model900/model49路线已有测试输入，不是全速度范围最优或GoAI GUI成功率保证。
只改部署命令，不训练、不替换ONNX。容错model49仍保留，尚未替换当前GoAI的model900。

这条链路只用于在GoAI仓库的真实MuJoCo比赛场景中验证冻结教师，不加载Isaac赛道复刻。
地图不写进bundle，也没有默认地图；每次启动必须显式传`--xml`。

## 运行合同

- NORMAL：T1 `model_17999`
- HIGH_CLIMB：T4a `model_900`
- LOW_STEP_SEQUENCE：当前已验收基线为T5 `model_800`；新一轮checkpoint是在该基座上训练的
  轮—棱有界残差，外部ONNX输入协议不变
- RECOVERY：Recovery候选 `model_900`
- Router安全分界：`<0.16 m`为Low，`>=0.16 m`为High
- Router继续读取操作者原始命令；只有Low Actor收到的前进速度会裁剪到训练上限`0.6 m/s`，
  NORMAL与High不裁剪，Low退出后原命令立即恢复。
- 每个Actor独立保持128维GRU状态；每拍只执行当前接管的Actor，其他休眠Actor持续保持零状态，切换进入专家时不会继承无关路段历史。
- 当前棱的四轮支撑分别连续确认3拍；四轮都曾确认后才允许寻找下一级或进入Recovery。
- Recovery收到新的非零人工命令会立即交还NORMAL，最长占用5秒。

GoAI的非对称观测严格按训练顺序组装：

```text
command[3] + proprio[57] + height_map[1,41,33]
```

MuJoCo射线扫描的原始扁平顺序是`(y=33, x=41)`（x最快变化）；送入Actor前必须显式转置成训练协议的`(1, x=41, y=33)`。Detector仍使用原始`(y,x)`几何网格，两者不能共用一次无语义的`reshape`。

旧版单MLP播放器使用的扁平`obs[1413]`仍然保留，两种ONNX协议不会混用。

## 启动

```bash
cd /home/hashai/Projects/DeepRobotics/goai_embodied_future_material

python3 scripts/play_mujoco_teacher.py \
  --xml models/mjcf/S10_track_lidar.xml \
  --router-bundle artifacts/s10_teacher_router/teacher_router_bundle.json \
  --detector-debug-vis \
  --terrain-id official_track \
  --real-time
```

启动时终端必须明确打印：

```text
xml=/.../models/mjcf/S10_track_lidar.xml
router_bundle=/.../artifacts/s10_teacher_router/teacher_router_bundle.json
```

不写`--xml`会直接报错，不会暗中选择官方地图。

## 快速切换测试路段

播放窗口运行时，在启动脚本的终端里轻按`n`，机器人会传送到路线顺序中的下一个
`track_waypoint_*`：

- 第一次按`n`时，以机器人当前位置最近的waypoint为当前点，传送到其后继点；继续按则严格按ID递增，终点后回到0。
- waypoint的Z是落脚面高度；传送后机身默认位于其上方`0.42 m`，四肢直接采用正常站立姿态。
- 机身朝向该waypoint之后的下一段路线；在终点则保持最后一段的进入方向。
- 传送同时清零机身/关节速度、人工速度命令和上一拍动作，并重置四个Actor的GRU及Router状态，旧路段状态不会带到新测试点。
- 可用`--waypoint-base-clearance`调整机身相对落脚面的高度，例如
  `--waypoint-base-clearance 0.44`。一般不需要修改默认值。

这是人工测试捷径，不代表机器人通过了被跳过的路段，也不要把含有传送跳变的录制数据当作
连续轨迹训练样本。其他按键仍为：`w/s`前后、`a/d`转向、`q/e`横移、`h`原地恢复、
`r`回起点、`p`暂停。

## Detector辅助线

- 青色矩形：前方台阶搜索走廊。
- 黄色粗线：检测到的候选棱。
- 绿色矩形：通过检查、允许锁存的落脚区域。
- 水平细线：三层净空射线；绿色为畅通，红色为命中。
- 红色粗线/矩形：几何上像台阶，但被判定为悬空楼板并拒绝。

三层净空射线不是简单的“前面有没有东西”：

```text
低层命中 + 上层畅通 -> 实体台阶立面
低层畅通 + 上层命中 -> 悬空楼板，拒绝锁存
低层命中 + 上层命中 -> 台阶位于低净空区域，保留台阶候选并显示净空告警
```

终端同步输出当前`mode`、`detector`、`overhead`、`headroom`和`rise`。其中
`headroom=1`只表示上层通道被占据；`overhead=1`才表示该候选已被悬空障碍规则拒绝。

## 重新导出

导出器显式导出GRU的`h_prev/h_next`，并对Torch/ONNX做数值一致性检查。它会根据checkpoint中
是否存在`low_residual.*`键自动选择Base或Low残差结构。当前只接受带
`low_residual.contract_version=4`、末层`[12,64]`的动作协同残差；旧v1/v2/v3均明确拒绝，不可重解释。
已验收Base model_800继续兼容，未验收新残差不会自动替换bundle。

v4每轮输出`[前后, 上下, 轮速]`，共12路，用`raw/(1+abs(raw))`产生双向有界请求。近棱共同门
允许前后轴同时动作和支撑协同，不再由前轴确认、后轴优先或净空归零互锁。名义棱距和轮位仍来自
同一高度图/本体输入。在`default_q + parent_action * scale`的**父策略目标角**上计算Jacobian：
前后/竖直请求各最多±4 cm，平滑缩小到每关节修正小于0.50 rad；仅朝直膝方向限制不超过原膝角
绝对值45%，不把正常屈膝抬轮也锁住。HipX修正为0，轮速修正最多±2 rad/s，按5 rad/s/action映射。
这些是目标修正上界，不是物理位移或通行率保证；std继承且冻结。RL父网络、动作16和Actor输入1413
不变。sidecar architecture为`frozen-base-coordination-low-residual-v4`。全部计算仍只使用同一份
`command/proprio/height_map`，不需要MuJoCo额外提供射线或状态机输入：

```bash
/home/hashai/桌面/miniconda3/envs/isaaclab511/bin/python \
  tools/export_s10_asymmetric_teacher_onnx.py \
  --checkpoint /absolute/path/model.pt \
  --output artifacts/s10_teacher_router/name.onnx
```

四个ONNX准备好后，用`tools/create_s10_teacher_router_bundle.py`生成带SHA256校验的bundle。
bundle只声明模型和Detector/Router合同，不声明地图。生成器默认写入
`low_forward_command_max_mps=0.6`；需要显式重建时可传
`--low-forward-command-max-mps 0.6`，旧bundle未带该字段时运行时也按0.6兼容读取。

## 当前边界

净空射线目前只参与确定性Detector/Router判断，不增加Actor观测维度，因此现有教师checkpoint
保持可用。向下高度图仍是Actor输入；如果整赛道测试证明低楼板会持续污染Actor高度图，即使
Router已经拒绝错误棱，下一步也应把“可行走表面高度”和“上方占用”拆成两个传感语义，而不是
继续调整High/Low高度阈值。
