# S10四教师Router的MuJoCo整赛道测试

2026-09-18：主入口`teacher_router_bundle.json`已改为当前正式接线，不再绑定旧LOW
model800；它与`teacher_router_low_command_events_model99.json`内容一致，LOW实际绑定
`low_command_events_milestone_model99.onnx`。同时修正两处Router交接：四轮已确认且处于同一
支撑面时，顶端松键/命令结束直接回NORMAL；连续台阶前轮已到下一踏面、后轮刚完成当前踏面时，
允许推进LOW successor，不再被`split_support`阻塞。终端与JSONL中的
`last_transition_reason`可区分正常完成、successor推进、零命令和超时。

当前保留原HIGH model900。HIGH Actor只把正向`vx`裁剪到bundle中的
`high_forward_command_max_mps=0.6`；小指令不放大，零命令、倒车以及`vy/wz`不变。
Router、Detector及NORMAL预热仍使用操作者原始命令。这个裁剪是Actor输入适配，不代表实际
运动速度恒为0.6 m/s，也不构成GoAI GUI通过结论。容错model49仍是待验收候选，未替换model900。

这条链路只用于在GoAI仓库的真实MuJoCo比赛场景中验证冻结教师，不加载Isaac赛道复刻。
地图不写进bundle，也没有默认地图；每次启动必须显式传`--xml`。

## 运行合同

- NORMAL：`model_275`
- HIGH_CLIMB：T4a `model_900`
- LOW_STEP_SEQUENCE：Low command-events `model_99`
- RECOVERY：Recovery候选 `model_900`
- Router安全分界：`<0.16 m`为Low，`>=0.16 m`为High
- Router继续读取操作者原始命令。LOW Actor使用锁存的世界穿越方向生成其训练时的
  `vx/vy/wz`输入，平移速度不超过`0.6 m/s`；HIGH Actor只把正向`vx`上限设为`0.6 m/s`。
  NORMAL、Router、Detector和NORMAL预热仍读取原始命令。
- LOW和HIGH都使用连通支撑面高度图。LOW按`0.18 m`最大相邻层差选择连续踏面；HIGH及
  Detector按其`0.45 m`检测上限选择连续踏面，避免把0.23 m高台退化成悬空结构的原始首击面。
- 每个Actor独立保持128维GRU状态。当前Actor运行；NORMAL在成功交接等待和RECOVERY期间预热，
  其他休眠Actor清零，避免继承无关技能历史。
- 当前棱的四轮支撑分别连续确认3拍。检测到下一低台阶后，即使前后轴正分处相邻踏面，也会
  推进LOW锁存目标；只有没有successor且四轮处于同一支撑面时才完成到NORMAL。
- LOW是纯前进过阶专家。LOW期间出现横移、转向、零命令或倒车时立即交给NORMAL，不进入
  RECOVERY，也不继续锁存新的LOW successor。
- LOW仅在12秒尝试超时时进入RECOVERY；HIGH仍保留零命令、不兼容命令和超时恢复路径。
  RECOVERY连续稳定0.2秒后回NORMAL，最长占用5秒。成功登顶和LOW命令交接不经过RECOVERY。

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
  --low-support-surface \
  --low-height-corridor-half-width 0.25 \
  --low-height-x-range -0.4 1.2 \
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

终端同步输出当前`mode`、`entry`、`transition`、`detector`、`overhead`、`headroom`和`rise`。其中
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
