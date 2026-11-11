# deploy_mujoco

基于 MuJoCo 物理引擎的 R2、G1 与 RW skater 强化学习策略部署及仿真环境。

## 项目简介

本项目在 MuJoCo 中运行 R2 v1.2 双足行走、G1 地形感知与动作跟踪、RW Skater 轮滑速度策略。

支持：

- R2 TorchScript、G1 SuperMimic PyTorch、G1 AME 和 RW Skater ONNX 策略部署
- 手柄（Joystick）实时控制机器人速度指令
- 关节状态日志记录（CSV 格式，兼容 PlotJuggler）
- 两种观测历史打包模式（`time` / `term`）
- 实时仿真与时间同步

## 目录结构

```
deploy_mujoco/
├── r2_deploy.py              # 主部署脚本（v1 配置 / 支持关节日志）
├── r2_term.py                # 部署脚本（v2 配置 / term 观测打包模式）
├── sim2sim/
│   ├── supermimic_sim2sim.py # S0 动作回放与 S1 → S0 在线仿真入口
│   ├── supermimic_planner.py # 用实测状态与地形生成 S1 参考动作
│   ├── evaluate_supermimic_terrain.py # 批量地形泛化评测
│   ├── supermimic_terrain_gif.py      # 导出轨迹与足底接触动画
│   └── rw_skater_sim2sim.py  # RW Skater 速度跟踪
├── asset/sp_g1/              # SuperMimic 训练用 MJCF、网格和冻结配置
├── config/
│   ├── rw_skater.yaml        # 轮滑策略选择与运行配置
│   ├── r2_v1.yaml            # 机器人配置 v1
│   └── r2_v2.yaml            # 机器人配置 v2
├── assets/
│   ├── skater_v2/            # RW Skater 模型和网格
│   └── r2_v1_2/
│       ├── mjcf/
│       │   └── r2_v1_2.xml   # MuJoCo 模型描述文件
│       ├── urdf/
│       │   ├── r2_v1_2.urdf  # URDF 模型文件
│       │   └── param.csv     # 模型参数
│       └── meshes/           # 机器人各部件的 STL 网格文件
├── policy/                   # 训练好的策略文件
│   ├── scan80000/            # S0 跟踪权重、平地、低台阶与高台示例
│   ├── s1_flow_w256/         # S1 动作规划权重
│   ├── skater_*/             # push_gait 和 swizzle 权重及训练参数
│   ├── policy.onnx / policy.pt  # 当前使用的策略
│   └── r2_flat_*/            # 各版本训练策略及参数备份
├── deploy_mujoco_logs/
│   └── joint_log.csv         # 关节状态日志输出
└── README.md
```

## SuperMimic Scan80000

在仓库根目录激活 `rw_sim2sim`。S0 和 S1 权重通过 Git LFS 分发，首次使用时安装 Git LFS 并运行 `git lfs pull` 下载权重。默认示例是 9.08 秒的低台阶动作，运行时会同时加载参考动作和对应的地形。

使用 `sim2sim/supermimic_sim2sim.py` 复现训练配置。根目录的 `scan80000_deploy.py` 保留了早期实验使用的机器人模型和统一动作缩放，供回溯对照。

```bash
conda activate rw_sim2sim
mjpython sim2sim/supermimic_sim2sim.py                 # macOS 窗口默认显示策略高程图
mjpython sim2sim/supermimic_sim2sim.py --show_ref      # 同时叠加参考机器人
mjpython sim2sim/supermimic_sim2sim.py --show_ref --no-heightmap  # 只看动作对照
python sim2sim/supermimic_sim2sim.py --check           # 无窗口完整回放
python sim2sim/supermimic_sim2sim.py --check --seconds 6 --motion policy/scan80000/motions/walk2_subject3.npz
```

窗口默认显示高程图，可用 `--no-heightmap` 关闭。地面和台阶上的彩点对应当前策略观测里的 187 个采样点。右下角俯视图显示同一帧，前方（+X）朝上，左侧（+Y）朝左，白框是根部投影。颜色表示根部到地形的垂直距离，单位是米。偏黄表示较近，偏蓝表示较远；固定色标为 0.4–1.2 m，面板底部显示本帧实际范围。达到策略 5 m 截断值的格子显示为灰色，三维场景省略对应的点。

`--show_ref` 在同一地形坐标系内叠加半透明粉色 G1。它按当前参考帧的根部位置、朝向和全部关节角做正向运动学，便于观察落脚、姿态和位置漂移。

机器人 MJCF 和 35 个网格取自 InstinctMJ 的 `03563ee`。目录 `asset/sp_g1/params/` 保存了冻结包 `stage2_usable_scan_80000_20261106` 的原始训练配置。入口从这些文件读取站姿、逐关节动作缩放、PD、关节惯量和仿真步长。权重位于 `policy/scan80000/model_80000.pt`，SHA256 为 `999b528d19f598eb6f2c12449649c47cdc26993790f70d49e06ec0919ce990c6`。

`--motion` 接受 planner 数据集导出的 50 Hz NPZ，使用 `base_pos_w`、`base_quat_w`（wxyz）、`joint_pos`、`joint_names` 和 `framerate`。`boxes` 各行保存地块坐标系下的中心、wxyz 四元数及完整长宽高，包含地板和障碍；缺省时使用水平地面。单地块回放采用训练标称参数，从第 0 帧开始，不施加训练时的随机扰动或观测噪声。

无窗口模式输出根部三维位置误差（米）和逐关节平均绝对误差（弧度）。超过根部高度或重力投影阈值时会记录首次越限时刻并返回非零退出码。较长动作可用 `--seconds` 截取，播放至片尾时自动结束。

### 在线 S1 planner → S0

`--planner` 加载 `policy/s1_flow_w256/ckpt_045000.pt` 的 EMA 权重与训练归一化参数。S1 直接使用 SuperMimic 源码里的推理实现，需要将含 `perspmimic/` 的训练仓库根目录加入 `PYTHONPATH`。此版本使用训练仓库的 `2e3da54` 提交验证。权重来自第二轮 `flow_w256` 第 45000 步，SHA256 为 `05eeb6ee7cff8c452459574e6dd4cb6772ff0ea7b062ccc787813ed7a289ac99`。

```bash
conda activate rw_sim2sim
export PYTHONPATH="/path/to/supermimic${PYTHONPATH:+:$PYTHONPATH}"
# 平地向前 2 m。粉色 G1 显示当前 S1 参考，绿色箭头标出目标位置和朝向。
mjpython sim2sim/supermimic_sim2sim.py --planner --goal 2 0 0 --show_ref
# 在低台阶场景中，把目标设在起点前方 6 m
python sim2sim/supermimic_sim2sim.py --planner --motion policy/scan80000/motions/low_step.npz \
  --goal 6 0 0 --seconds 16 --seed 0 --check
```

`--goal` 的三个值依次是相对初始朝向的前向距离（米）、左向距离（米）和最终偏航角（度）。场景 NPZ 提供起始姿态和地形，机器人从零速度启动。S1 每 0.2 秒读取最近 4 帧实际机器人状态、31×61 的局部地形图和目标，按 50 Hz 生成未来 64 帧。重规划时保留上一计划中已承诺的 20 帧。S0 跟踪这些参考，窗口显示的是 S0 自己使用的 187 点高程图。

默认评测 12 秒、随机种子 0。每个 S0 控制步都会将实际姿态与当前参考比较。根部高度绝对误差超过 0.25 m 或机身坐标系下单位重力向量的差值范数超过 0.8 时，在线仿真会提前结束。S1 重规划时，如果局部高程图、历史位置或中间目标的地形查询超出地块，也会提前结束。`--check` 在跑满时长、全程未越限且终点水平误差小于 0.30 m、朝向误差小于 20° 时返回成功。中途到达目标会继续仿真。

要观察机器人跌倒后或高程图越界后的行为，给在线模式加 `--no-early-stop`。下面的命令在 macOS 打开 95 cm 高台场景。

```bash
mjpython sim2sim/supermimic_sim2sim.py --terrain-demo climb --box-height .95 --seed 0 --no-early-stop
```

实时窗口会一直运行到手动关闭。可加 `--seconds 30` 在 30 秒仿真时间结束。无窗口检查和 GIF 导出仍有默认时长，地形演示为 16 秒，普通 planner 为 12 秒。越过跟踪阈值后，S1 和 S0 继续闭环运行。地形覆盖不足期间，S0 跟完最近一次有效计划后保持最后的参考姿态；首次规划就越界时使用初始姿态。S1 每 0.2 秒重新查询，覆盖恢复后继续生成动作。非有限数值和 MuJoCo 仿真异常仍会报错。

`--metrics-json` 的 `stop` 记录实际结束原因，`first_violation_s` 分别记录跟踪越限和地形越界的首次时刻（秒）。连续模式中只要发生过越限，`--check` 仍返回失败。批量泛化脚本 `sim2sim/evaluate_supermimic_terrain.py` 同样接受 `--no-early-stop`，汇总中的 `trials_with_violation` 统计发生过越限的试次。

在 `rw_sim2sim` 环境用种子 0、1、2 运行以上两个场景，六次均未达标。表中为执行结束时实际机器人的水平目标误差。

| 场景 | 种子 0 | 种子 1 | 种子 2 |
|---|---:|---:|---:|
| 平地向前 2 m，12 s | 3.74 m | 5.32 m | 2.31 m |
| 低台阶向前 6 m，16 s | 9.43 m | 4.44 m | 5.43 m（13 s 时高程图越界） |

平地种子 0 在 8.22 s 时最接近目标，距离 0.38 m，之后继续偏离。相同起点、目标和种子下，直接执行 S1 生成姿态的运动学对照在 12 s 时误差为 0.29 m。

### 台阶和高台可视化

激活上面的 `rw_sim2sim` 环境并设置 `PYTHONPATH` 后，在仓库根目录运行：

```bash
mjpython sim2sim/supermimic_sim2sim.py --terrain-demo step
mjpython sim2sim/supermimic_sim2sim.py --terrain-demo climb
python sim2sim/supermimic_sim2sim.py --terrain-demo climb --seed 0 \
  --gif sim2sim_output/supermimic/climb_seed0.gif
```

`step` 加载 35.2 厘米台阶，`climb` 加载 76 厘米高台。后者的场景文件 `policy/scan80000/motions/climb_76.npz` 来自 `planner_v1_20261109/motions/train/train__php__motions-low-climb-76__forward_walk0.6s_step1.2s.npz`。两者默认从静止开始，目标在前方 6 米，运行 16 秒。可用 `--seed`、`--seconds` 和 `--goal` 覆盖。

窗口默认叠加粉色 S1 参考机器人、S0 高程采样点和绿色目标箭头。`--gif` 使用离屏渲染，并在画面右侧显示俯视轨迹和足底接触时间。脚踩上台面的判据为法向力大于 1 N、接触法线竖直分量的绝对值至少 0.8、接触点距台面高度小于 2.5 厘米。双脚先后上台，再各自在障碍另一侧接触地面时，记录落地时间。渲染代码位于 `sim2sim/supermimic_terrain_gif.py`。

### 改变箱高和验证泛化

`--box-height` 以米指定障碍最高台面相对地面的高度。组合箱体以地板上表面为基准在竖直方向等比缩放，0 表示只保留地板。`--box-offset-y` 以米沿场景的世界 Y 轴平移整组障碍。起点和目标仍在原位置。MuJoCo 碰撞体、S1 地形输入、S0 高程图和足底判定使用同一组修改后的箱体；源 NPZ 只读。

```bash
# 观察高台改成 70 cm、整体横移 +25 cm 后的闭环行为
mjpython sim2sim/supermimic_sim2sim.py --terrain-demo climb --box-height .70 --box-offset-y .25
# 观察 75 cm 高台沿世界 Y 轴平移 0.5 m 后的动作，直到关闭窗口
mjpython sim2sim/supermimic_sim2sim.py --terrain-demo climb --box-height .75 --box-offset-y .5 --seed 0 --no-early-stop
# 单次无窗口运行，记录真实接触、最近及最终目标距离、朝向和停止原因
python sim2sim/supermimic_sim2sim.py --terrain-demo step --box-height .55 --seed 0 \
  --check --metrics-json sim2sim_output/supermimic/step55.json
# 配对实验：原始地形、只改高度、只改横向位置。每种设置使用相同的一组种子。
python sim2sim/evaluate_supermimic_terrain.py --workers 3 --output sim2sim_output/supermimic/sweep.json
# 自定义高度与种子；--offsets 0 表示只测原始位置
python sim2sim/evaluate_supermimic_terrain.py --scenes climb --heights .55 .70 .82 .95 --offsets 0 \
  --seeds 0 1 2 --workers 3 --output sim2sim_output/supermimic/climb_heights.json
# 原始留出布局：读取 planner 数据集 index.json 中 test 划分的 step、climb 样本
python sim2sim/evaluate_supermimic_terrain.py --heldout-dataset /path/to/planner_v1_20261109 \
  --seeds 0 1 2 --workers 3 --output sim2sim_output/supermimic/heldout.json
```

配对实验使用前方 6 m 的固定目标。留出测试只用起始姿态和箱体几何选目标，方向从起点指向箱体中心的均值，终点落在最远箱边外 1.2 m，最终朝向同一方向。两个协议都在 16 秒时按相同阈值结算。报告中的 `crossed` 记录两脚是否踩上台面再落到另一侧，`goal_reached` 记录终点距离和朝向是否达标，`task_success` 要求两者同时满足。平地对照的 `crossed` 为 `null`。提前停止的试次也计入分母。`--check` 的退出码沿用原来的终点判据。

2026-11-11 实测 42 条配对试次和 60 条留出试次。台阶布局在 20、30、35.2、40 cm 各翻越 3/3 次，55 cm 为 0/3。高台布局在 55、70、76.4、82、95 cm 依次为 0/3、3/3、3/3、1/3、0/3。原高下向世界 Y 轴正负方向各平移 25 cm，两种布局共翻越 12/12 次。20 个留出布局每个运行 3 个种子：台阶翻越 31/36、完整任务 0/36，高台翻越 12/24、完整任务 5/24。

## 环境配置

### 依赖

- Python 3.8+
- **PyTorch**（R2 TorchScript 和 SuperMimic 推理）
- **ONNX Runtime**（G1 AME 和 RW Skater 推理）
- **MuJoCo**（`mujoco` Python 绑定，必须）
- NumPy、PyYAML（必须）
- Pillow（SuperMimic 高程图面板）
- Pygame（手柄控制，可选）

> **说明**：R2 使用 `torch.jit.load`；G1 AME 使用训练仓库 exporter 生成的 ONNX，并通过
> `onnxruntime` 推理。

### 方式一：conda / virtualenv（推荐）

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

或使用 conda：

```bash
conda create -n deploy_mujoco python=3.10 -y
conda activate deploy_mujoco

pip install -r requirements.txt
```

### 方式二：直接安装

```bash
pip install -r requirements.txt
```

### 验证安装

```bash
python -c "import mujoco; import torch; print('MuJoCo:', mujoco.__version__); print('PyTorch:', torch.__version__)"
```

## 快速开始

### 1. 使用默认配置运行

```bash
# v1 配置（带关节日志）
python r2_deploy.py

# v2 配置（term 观测模式）
python r2_term.py
```

### 2. 指定策略文件

```bash
python r2_deploy.py --policy policy/model_20000.pt
```

### 3. 指定观测历史打包模式

```bash
# time 模式：按时间顺序逐帧堆叠
python r2_deploy.py --time

# term 模式（默认）：按观测项分组打包
python r2_deploy.py --term
```

### 4. 自定义日志参数

```bash
python r2_deploy.py \
  --policy policy/policy.pt \
  --log-csv deploy_mujoco_logs/my_log.csv \
  --log-duration 30.0
```

### 5. G1 AME 策略

AME 部署使用训练仓库 `scripts/rsl_rl/exporter.py` 导出的 ONNX。默认策略为
`policy/ame1.onnx`，同时提供带全局地形特征分支的 `policy/ame2.onnx`。两个预训练 checkpoint
使用原始 `0.05 m` 网格：观测为 96 维本体状态加 `33×21×3=2079` 维局部地形坐标。

```bash
# 无界面核对模型、关节顺序、高程图和策略维度
python ame_deploy.py --check

# 运行并显示 21x33 高程采样点、打印 z 数值
python ame_deploy.py --heightmap-debug

# 同时显示从 torso_link 上方 20 m 发出的射线
python ame_deploy.py --heightmap-debug --heightmap-rays

# 指定 ONNX、速度指令或启用手柄
python ame_deploy.py --policy /path/to/policy.onnx --cmd 0.5 0.0 0.0
python ame_deploy.py --joystick
```

AME 的 29 个观测/动作关节按训练时 PhysX articulation 顺序组织，再通过关节名映射到 MJCF；
不能直接使用 MJCF 文件中的关节遍历顺序。高程图只对世界地形射线检测，会排除机器人自身几何体。
AME 默认使用 `assets/g1/mjcf/g1_29dof_ame.xml`：惯量和关节结构保持 G1 29DoF，碰撞体使用
capsule/sphere 足底与肢体近似，避免三角 mesh 接触和点状足底造成的 sim2sim 失稳。
默认命令还会复现 PLAY 配置的 heading 控制：目标朝向为 0 rad，使用 0.5 的增益动态生成 yaw-rate；
显式传入 `--cmd` 或 `--joystick` 时则使用直接速度指令。

训练地形也可以直接从命令行切换。默认 `play_stakes` 对应训练代码
`G1RoughEnvCfg_PLAY` 的固定梅花桩；其余名称来自 `FINETUNE_ROUGH_TERRAINS_CFG`：

```bash
# 训练 Play 配置的梅花桩
python ame_deploy.py --terrain play_stakes --heightmap-debug

# 正/反金字塔楼梯
python ame_deploy.py --terrain pyramid_stairs --terrain-difficulty 0.5 --heightmap-debug
python ame_deploy.py --terrain pyramid_stairs_inv --terrain-difficulty 0.5 --heightmap-debug

# 训练中的其他地形
python ame_deploy.py --terrain stakes1
python ame_deploy.py --terrain stakes2
python ame_deploy.py --terrain stakes3
python ame_deploy.py --terrain hf_gaps
python ame_deploy.py --terrain stonebridge
python ame_deploy.py --terrain rails
```

每块地形为8×8m，水平分辨率0.05m、垂直量化0.005m。`--terrain-difficulty` 范围为
0到1；`--terrain-seed` 用于复现随机桩高和地面高度。训练中基于 mesh 的正/反金字塔楼梯和
rails 在 MuJoCo 中使用同尺寸 box 几何，避免 hfield 在垂直边缘产生斜坡观测。

### 6. RW skater_v2 速度策略

`sim2sim/rw_skater_sim2sim.py` 运行 rw_lab `Rw-Lab-Skater_*` 任务训练的 RW 轮滑策略。模型
`assets/skater_v2` 来自 `rw_mode/skater_v2`（commit 2dba443），策略来自 `rw_policy/skater`，
配置为 `config/rw_skater.yaml`。目前收录的 `push_gait`（2026-04-21）和 `swizzle`（2026-04-17）
都是在 skater_v2 上训练的；rw_policy 中 `rw_wheel/*` 的策略基于旧 rw_wheel 模型（踝关节结构和
右踝 pitch 轴向不同），在 skater_v2 上基本站不住，不要直接用本脚本跑。

观测每帧 84 维（`ang_vel×0.2, gravity, cmd, joint_pos_rel, joint_vel×0.05, last_action`），5 帧
历史按观测项分组展开为 420 维；动作 25 维，`target = default + 0.25 × action`，控制频率 50 Hz。
默认关节角、PD 增益和力矩上限都取自策略的 `params/env.yaml`；关节按名字映射到 MJCF，4×2 个轮子
关节为被动关节。

```bash
# 无界面核对模型、观测维度和 ONNX
python sim2sim/rw_skater_sim2sim.py --check

# 打开 viewer（macOS 需用 mjpython），↑/↓ 调 vx，←/→ 调 yaw-rate，Enter 复位
mjpython sim2sim/rw_skater_sim2sim.py --cmd 0.3,0,0
mjpython sim2sim/rw_skater_sim2sim.py --policy-name swizzle --joystick

# 无界面按段跑指令并打印稳态跟踪（vx,vy,wz,秒；负数要写成 --cmd=-0.4,0,0,6）
python sim2sim/rw_skater_sim2sim.py --headless --cmd 0,0,0,4 --cmd 0.4,0,0,8 --cmd 0.25,0,0.4,6
```

速度指令会被裁剪到各策略的训练范围（`push_gait`: vx −0.1~0.5，`swizzle`: vx −0.6~0.6，
yaw-rate 均为 ±0.5）。轨迹 CSV 默认写入 `sim2sim_output/rw_skater/`，可用 `--no-log` 关闭。
新增策略时在配置的 `policies` 下加一项即可。

## 配置说明

配置文件位于 `config/` 目录，使用 YAML 格式。以 `r2_v2.yaml` 为例：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `xml_path` | MuJoCo 模型路径 | `/path/to/r2_v1_2.xml` |
| `simulation_duration` | 仿真总时长（秒） | `120.0` |
| `simulation_dt` | 仿真时间步长 | `0.005` |
| `control_decimation` | 控制频率降采样因子 | `4` |
| `kps` / `kds` | PD 控制器的比例/微分增益 | `[100, ...]` |
| `default_angles` | 各关节默认角度（初始姿态） | `[-0.1, ...]` |
| `action_scale` | 策略输出动作缩放 | `0.25` |
| `num_actions` | 动作维度（受控关节数） | `12` |
| `num_obs` | 观测维度 | `45` |
| `obs_history_length` | 观测历史帧数 | `5` |
| `policy_joints` | 策略控制的关节列表 | 12 个腿关节 |

### 机器人关节

R2 v1.2 策略控制 12 个关节（6 自由度/腿）：

| 左腿 | 右腿 |
|------|------|
| left_hip_pitch_joint | right_hip_pitch_joint |
| left_hip_roll_joint | right_hip_roll_joint |
| left_hip_yaw_joint | right_hip_yaw_joint |
| left_knee_joint | right_knee_joint |
| left_ankle_pitch_joint | right_ankle_pitch_joint |
| left_ankle_roll_joint | right_ankle_roll_joint |

## 控制架构

```
手柄/指令 → 速度命令(cmd)
                │
                ▼
┌──────────────────────────────────────┐
│           观测构建                    │
│  omega + 重力方向 + cmd + 关节状态    │
│  + 关节速度 + 上一帧动作              │
│         → 观测历史堆叠                │
└──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│       神经网络策略推理 (50Hz)         │
│         → 目标关节角度增量             │
└──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│       PD 控制器 (250Hz)               │
│   (target_q - q) × Kp + (target_dq - dq) × Kd  │
│         → 关节力矩                    │
└──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│      MuJoCo 物理仿真 (200Hz)          │
└──────────────────────────────────────┘
```

- 仿真频率：200 Hz（dt = 0.005s）
- 策略推理频率：50 Hz（control_decimation = 4）
- 手柄指令更新频率：10 Hz

## 手柄控制

连接手柄后，脚本自动检测并使用：

| 轴 | 功能 | 范围 |
|----|------|------|
| 左摇杆 Y （轴 1） | 前进/后退速度 | -2.0 ~ 2.0 m/s |
| 左摇杆 X （轴 0） | 左右速度 | -1.0 ~ 1.0 m/s |
| 右摇杆 X （轴 3） | 偏航角速度 | -1.0 ~ 1.0 rad/s |

## 关节日志

运行 `r2_deploy.py` 时自动记录关节数据到 CSV 文件，包含：

- 各关节位置（pos）
- 各关节速度（vel）
- 各关节力矩（tor）
- 各关节目标位置（target_pos）

日志格式兼容 PlotJuggler，可直接导入进行数据分析和可视化。

## 策略文件

`policy/` 目录包含多个训练版本的策略：

- `r2_flat_*` 系列：平地步态策略的不同训练版本
- 每个版本包含 `exported/`（ONNX + TorchScript）、`params/`（训练参数）
- 根目录的 `policy.onnx` / `policy.pt` 为当前使用的最新策略

## 两个部署脚本的区别

| 特性 | r2_deploy.py | r2_term.py |
|------|-------------|------------|
| 默认配置 | r2_v1.yaml | r2_v2.yaml |
| 观测打包方式 | 可切换（`--time` / `--term`） | 固定 term 分组 |
| 关节日志 | ✅ 支持 | ❌ 不支持 |
| 关节映射 | 通过执行器 ID 查找 | 简化版映射 |
| 观测历史 | 可配置长度 | 固定 5 帧 |

## License

Internal research project.
