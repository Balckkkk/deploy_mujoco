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
│   ├── supermimic_sim2sim.py # G1 Scan80000 参考动作跟踪
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
│   ├── scan80000/            # SuperMimic 权重和两个示例动作
│   ├── skater_*/             # push_gait 和 swizzle 权重及训练参数
│   ├── policy.onnx / policy.pt  # 当前使用的策略
│   └── r2_flat_*/            # 各版本训练策略及参数备份
├── deploy_mujoco_logs/
│   └── joint_log.csv         # 关节状态日志输出
└── README.md
```

## SuperMimic Scan80000

在仓库根目录激活 `rw_sim2sim`。默认示例是 9.08 秒的低台阶动作，运行时会同时加载参考动作和对应的地形。

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
