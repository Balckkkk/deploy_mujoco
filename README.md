# deploy_mujoco

基于 MuJoCo 物理引擎的 R2 双足机器人强化学习策略部署与仿真环境。

## 项目简介

本项目用于在 MuJoCo 仿真器中部署和测试训练好的强化学习行走策略。机器人模型为 R2 v1.2 —— 一款 12 自由度的双足机器人，通过 PD 控制器 + 神经网络策略实现实时运动控制。

支持：

- TorchScript（`.pt`）和 ONNX（`.onnx`）格式的策略模型
- 手柄（Joystick）实时控制机器人速度指令
- 关节状态日志记录（CSV 格式，兼容 PlotJuggler）
- 两种观测历史打包模式（`time` / `term`）
- 实时仿真与时间同步

## 目录结构

```
deploy_mujoco/
├── r2_deploy.py              # 主部署脚本（v1 配置 / 支持关节日志）
├── r2_term.py                # 部署脚本（v2 配置 / term 观测打包模式）
├── config/
│   ├── r2_v1.yaml            # 机器人配置 v1
│   └── r2_v2.yaml            # 机器人配置 v2
├── assets/
│   └── r2_v1_2/
│       ├── mjcf/
│       │   └── r2_v1_2.xml   # MuJoCo 模型描述文件
│       ├── urdf/
│       │   ├── r2_v1_2.urdf  # URDF 模型文件
│       │   └── param.csv     # 模型参数
│       └── meshes/           # 机器人各部件的 STL 网格文件
├── policy/                   # 训练好的策略文件
│   ├── policy.onnx / policy.pt  # 当前使用的策略
│   └── r2_flat_*/            # 各版本训练策略及参数备份
├── deploy_mujoco_logs/
│   └── joint_log.csv         # 关节状态日志输出
└── README.md
```

## 环境配置

### 依赖

- Python 3.8+
- **PyTorch**（策略推理通过 `torch.jit.load` 加载 TorchScript 模型，必须）
- **MuJoCo**（`mujoco` Python 绑定，必须）
- NumPy、PyYAML（必须）
- Pygame（手柄控制，可选）

> **说明**：策略加载完全走 `torch.jit.load`，ONNX 文件仅为训练时导出备份，当前不使用 `onnxruntime`，无需安装。

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
pip install mujoco torch numpy pyyaml pygame
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
