#!/usr/bin/env python3
"""MuJoCo sim2sim runner for RW skater_v2 velocity policies trained in rw_lab.

The policy consumes 5 frames of proprioception packed term-major (all frames of
base_ang_vel, then all frames of projected_gravity, ...), 84 values per frame,
and outputs 25 joint position offsets in Isaac articulation order.  The passive
wheel joints are not actuated.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
import os
import re
import sys
import time
from typing import Any

# EGL only exists on Linux; macOS rejects MUJOCO_GL=egl at import time.
if sys.platform.startswith("linux") and ("--headless" in sys.argv or "--check" in sys.argv):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config/rw_skater.yaml"

KEY_ENTER = 257
KEY_KP_ENTER = 335
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265

# Policy observation terms per frame, in the order used by the rw_lab env cfg.
OBS_TERMS = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "last_action")


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path, policy_name: str | None = None, policy_path: str | None = None) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    name = policy_name or config["default_policy"]
    if name not in config["policies"]:
        raise ValueError(f"Unknown policy '{name}', available: {', '.join(config['policies'])}")
    entry = config["policies"][name]
    config["policy_name"] = name
    config["policy_path"] = policy_path if policy_path is not None else entry["path"]
    config["command_ranges"] = entry["command_ranges"]
    for key in ("model_path", "policy_path", "output_dir"):
        config[key] = str(resolve_path(config[key]))
    return config


def values_from_rules(
    joint_names: list[str],
    rules: list[dict[str, Any]],
    *,
    default: float | None,
    label: str,
) -> np.ndarray:
    values: list[float] = []
    for joint_name in joint_names:
        value = default
        matched = False
        for rule in rules:
            if re.fullmatch(str(rule["pattern"]), joint_name):
                value = float(rule["value"])
                matched = True
                break
        if not matched and default is None:
            raise ValueError(f"No {label} rule matched {joint_name}")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def setup_joint_mappings(
    model: mujoco.MjModel,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    actuator_indices: list[int] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if joint_id < 0:
            raise ValueError(f"Policy joint not found in MuJoCo model: {joint_name}")
        if actuator_id < 0:
            raise ValueError(f"Actuator not found in MuJoCo model: {joint_name}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        actuator_indices.append(int(actuator_id))
    return (
        np.asarray(qpos_indices, dtype=np.int32),
        np.asarray(qvel_indices, dtype=np.int32),
        np.asarray(actuator_indices, dtype=np.int32),
    )


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz
    return np.array(
        [2.0 * (-qz * qx + qw * qy), -2.0 * (qz * qy + qw * qx), 1.0 - 2.0 * (qw * qw + qz * qz)],
        dtype=np.float32,
    )


def world_to_body(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat_wxyz, dtype=np.float64))
    return rotation.reshape(3, 3).T @ np.asarray(vector, dtype=np.float64)


def pack_history_term_major(history: deque[dict[str, np.ndarray]]) -> np.ndarray:
    """Isaac Lab flatten_history_dim with concatenate_terms: each term's history is contiguous."""
    return np.concatenate(
        [np.concatenate([frame[term] for frame in history]) for term in OBS_TERMS]
    ).astype(np.float32)


class OnnxPolicy:
    def __init__(self, path: Path, num_obs: int, num_actions: int) -> None:
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name
        expected = int(model_input.shape[-1])
        if expected != num_obs:
            raise ValueError(f"Policy expects {expected} observations, runtime builds {num_obs}")
        self.num_actions = num_actions

    def infer(self, obs: np.ndarray) -> np.ndarray:
        action = self.session.run([self.output_name], {self.input_name: obs.reshape(1, -1)})[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.num_actions:
            raise ValueError(f"Policy returned {action.shape[0]} actions, expected {self.num_actions}")
        return action


class VelocityCommand:
    """vx, vy, yaw-rate command clipped to the policy's training ranges."""

    def __init__(self, initial: list[float], ranges: dict[str, list[float]], step: list[float]) -> None:
        self.low = np.array([ranges["lin_vel_x"][0], ranges["lin_vel_y"][0], ranges["ang_vel_z"][0]], dtype=np.float32)
        self.high = np.array([ranges["lin_vel_x"][1], ranges["lin_vel_y"][1], ranges["ang_vel_z"][1]], dtype=np.float32)
        self.step = np.asarray(step, dtype=np.float32)
        self.value = np.zeros(3, dtype=np.float32)
        self.set(initial)

    def set(self, value: list[float] | np.ndarray) -> None:
        self.value[:] = np.clip(np.asarray(value, dtype=np.float32), self.low, self.high)

    def nudge(self, axis: int, sign: float) -> None:
        value = self.value.copy()
        value[axis] = round(float(value[axis] + sign * self.step[axis]), 3)
        self.set(value)

    def from_axes(self, axes: np.ndarray) -> None:
        """Map stick deflection in [-1, 1] onto the asymmetric training range."""
        axes = np.clip(axes, -1.0, 1.0)
        self.set(np.where(axes >= 0.0, axes * self.high, -axes * self.low))

    def __str__(self) -> str:
        return f"vx={self.value[0]:+.2f} vy={self.value[1]:+.2f} wz={self.value[2]:+.2f}"


class JoystickInput:
    """Optional polling joystick; axes match the existing R2/AME deployments."""

    def __init__(self) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            pygame.quit()
            raise RuntimeError("No joystick device found")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def axes(self) -> np.ndarray:
        self.pygame.event.pump()
        return np.array(
            [-self.joystick.get_axis(1), -self.joystick.get_axis(0), -self.joystick.get_axis(3)],
            dtype=np.float32,
        )

    def close(self) -> None:
        self.pygame.quit()


class RwSkaterRuntime:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(config["model_path"])
        self.model.opt.timestep = float(config["simulation_dt"])
        self.data = mujoco.MjData(self.model)
        self.decimation = int(config["control_decimation"])
        self.policy_dt = self.model.opt.timestep * self.decimation
        self.joint_names = list(config["policy_joints"])
        self.num_actions = int(config["num_actions"])
        if len(self.joint_names) != self.num_actions:
            raise ValueError(f"policy_joints has {len(self.joint_names)} entries, expected {self.num_actions}")
        self.qpos_indices, self.qvel_indices, self.actuator_indices = setup_joint_mappings(
            self.model, self.joint_names
        )
        self.default_joint_pos = values_from_rules(
            self.joint_names, config["default_joint_pos_rules"], default=0.0, label="default_joint_pos"
        )
        self.kp = values_from_rules(self.joint_names, config["stiffness_rules"], default=None, label="stiffness")
        self.kd = values_from_rules(self.joint_names, config["damping_rules"], default=None, label="damping")
        self.effort_limit = values_from_rules(
            self.joint_names, config["effort_limit_rules"], default=None, label="effort_limit"
        )
        self.action_scale = float(config["action_scale"])
        self.obs_scales = {term: float(config["obs_scales"][term]) for term in OBS_TERMS}
        self.history_length = int(config["history_length"])
        self.num_obs = self.history_length * (9 + 3 * self.num_actions)
        self.policy = OnnxPolicy(Path(config["policy_path"]), self.num_obs, self.num_actions)
        self.command = VelocityCommand(
            config.get("initial_command", [0.0, 0.0, 0.0]),
            config["command_ranges"],
            config.get("command_step", [0.1, 0.1, 0.1]),
        )
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_q = self.default_joint_pos.copy()
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=self.history_length)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:3] = np.asarray(self.config.get("init_base_pos", self.model.qpos0[:3]), dtype=np.float64)
        self.data.qpos[self.qpos_indices] = self.default_joint_pos
        self.data.qvel[:] = 0.0
        self.last_action[:] = 0.0
        self.target_q = self.default_joint_pos.copy()
        mujoco.mj_forward(self.model, self.data)
        # Isaac Lab fills the history buffer with the first observation after reset.
        frame = self.observation_frame()
        self.history.clear()
        for _ in range(self.history_length):
            self.history.append(frame)

    def observation_frame(self) -> dict[str, np.ndarray]:
        joint_pos = self.data.qpos[self.qpos_indices].astype(np.float32) - self.default_joint_pos
        joint_vel = self.data.qvel[self.qvel_indices].astype(np.float32)
        raw = {
            # Free-joint angular velocity in qvel is already expressed in the body frame.
            "base_ang_vel": self.data.qvel[3:6].astype(np.float32),
            "projected_gravity": projected_gravity(self.data.qpos[3:7]),
            "velocity_commands": self.command.value.copy(),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "last_action": self.last_action.copy(),
        }
        return {term: raw[term] * self.obs_scales[term] for term in OBS_TERMS}

    def policy_step(self) -> np.ndarray:
        self.history.append(self.observation_frame())
        action = self.policy.infer(pack_history_term_major(self.history))
        self.target_q = self.default_joint_pos + self.action_scale * action
        self.last_action = action
        return action

    def physics_step(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_indices]
        dq = self.data.qvel[self.qvel_indices]
        torque = np.clip(self.kp * (self.target_q - q) - self.kd * dq, -self.effort_limit, self.effort_limit)
        self.data.ctrl[self.actuator_indices] = torque
        mujoco.mj_step(self.model, self.data)
        return torque.astype(np.float32)

    def base_state(self) -> dict[str, float]:
        lin_vel_b = world_to_body(self.data.qpos[3:7], self.data.qvel[0:3])
        gravity = projected_gravity(self.data.qpos[3:7])
        tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1.0, 1.0))))
        return {
            "x": float(self.data.qpos[0]),
            "y": float(self.data.qpos[1]),
            "z": float(self.data.qpos[2]),
            "vx": float(lin_vel_b[0]),
            "vy": float(lin_vel_b[1]),
            "wz": float(self.data.qvel[5]),
            "tilt_deg": tilt,
        }

    def has_fallen(self) -> bool:
        state = self.base_state()
        return (
            state["z"] < float(self.config["fall_height"])
            or state["tilt_deg"] > float(self.config["fall_tilt_deg"])
            or not np.isfinite(self.data.qpos).all()
        )

    def run_check(self) -> None:
        self.reset()
        obs = pack_history_term_major(self.history)
        action = self.policy.infer(obs)
        frame = self.history[-1]
        print(f"Model: {self.config['model_path']} nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")
        print(f"Policy: {self.config['policy_name']} -> {self.config['policy_path']}")
        print(f"Timing: sim_dt={self.model.opt.timestep}, decimation={self.decimation}, policy_dt={self.policy_dt}")
        print(
            "Observation terms per frame: "
            + ", ".join(f"{term}={frame[term].shape[0]}" for term in OBS_TERMS)
            + f" x history {self.history_length} = {obs.shape[0]}"
        )
        print(
            f"Command ranges: vx={self.command.low[0]:+.2f}..{self.command.high[0]:+.2f}, "
            f"wz={self.command.low[2]:+.2f}..{self.command.high[2]:+.2f}"
        )
        print(
            f"Action: shape={action.shape}, min={action.min():.4f}, "
            f"max={action.max():.4f}, mean={action.mean():.4f}"
        )
        print("rw_skater sim2sim check passed.")

    def open_logger(self, enabled: bool) -> tuple[Any, csv.DictWriter | None]:
        if not enabled:
            return None, None
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.config['policy_name']}_trajectory.csv"
        file = path.open("w", newline="", encoding="utf-8")
        fields = ["time", "cmd_vx", "cmd_vy", "cmd_wz", "x", "y", "z", "vx", "vy", "wz", "tilt_deg", "torque_max"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        print(f"Logging trajectory: {path}")
        return file, writer

    def log_row(self, writer: csv.DictWriter | None, torque: np.ndarray) -> None:
        if writer is None:
            return
        row = {"time": f"{self.data.time:.4f}", "torque_max": f"{np.abs(torque).max():.4f}"}
        row.update({f"cmd_{key}": f"{value:.3f}" for key, value in zip(("vx", "vy", "wz"), self.command.value)})
        row.update({key: f"{value:.5f}" for key, value in self.base_state().items()})
        writer.writerow(row)

    def run_headless(self, *, segments: list[tuple[np.ndarray, float]], log: bool) -> bool:
        """Run command segments and print steady-state tracking; returns False if the robot fell."""
        self.reset()
        log_file, writer = self.open_logger(log)
        counter = 0
        fallen = False
        try:
            for command, duration in segments:
                self.command.set(command)
                samples: list[dict[str, float]] = []
                end_time = self.data.time + duration
                while self.data.time < end_time - 1e-9:
                    torque = self.physics_step()
                    counter += 1
                    if counter % self.decimation == 0:
                        self.policy_step()
                        self.log_row(writer, torque)
                        samples.append(self.base_state())
                        if self.has_fallen():
                            fallen = True
                            break
                steady = samples[len(samples) // 2 :] or samples
                mean = {key: float(np.mean([s[key] for s in steady])) for key in ("vx", "vy", "wz")}
                status = f"FELL at t={self.data.time:.2f}s" if fallen else "ok"
                print(
                    f"cmd [{self.command}] {duration:.1f}s -> {status}: "
                    f"vx={mean['vx']:+.3f} vy={mean['vy']:+.3f} wz={mean['wz']:+.3f} "
                    f"z_min={min(s['z'] for s in samples):.3f} tilt_max={max(s['tilt_deg'] for s in samples):.1f}deg"
                )
                if fallen:
                    break
        finally:
            if log_file is not None:
                log_file.close()
        print(f"Headless run finished: sim_time={self.data.time:.2f}s, fallen={fallen}")
        return not fallen

    def run_viewer(self, *, duration: float, realtime: bool, log: bool, joystick: JoystickInput | None) -> None:
        self.reset()
        print(f"Policy: {self.config['policy_name']} -> {self.config['policy_path']}")
        print("Keys: Up/Down vx, Left/Right yaw rate, Enter reset pose and zero command")
        print(f"Command: {self.command}")
        pending: deque[int] = deque()

        def key_callback(keycode: int) -> None:
            pending.append(keycode)

        log_file, writer = self.open_logger(log)
        counter = 0
        wall_start = time.perf_counter()
        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
                # Pace against an absolute wall-clock anchor so sleep overshoot does not accumulate.
                anchor_wall, anchor_sim = time.perf_counter(), self.data.time
                while viewer.is_running() and self.data.time < duration:
                    while pending:
                        self.handle_key(pending.popleft())
                    torque = self.physics_step()
                    counter += 1
                    if counter % self.decimation == 0:
                        if joystick is not None:
                            self.command.from_axes(joystick.axes())
                        self.policy_step()
                        self.log_row(writer, torque)
                        # Rendering at the 50 Hz policy rate keeps macOS viewer close to realtime.
                        viewer.sync()
                    if realtime:
                        if self.data.time < anchor_sim:  # reset rewound the clock
                            anchor_wall, anchor_sim = time.perf_counter(), self.data.time
                        sleep_time = (self.data.time - anchor_sim) - (time.perf_counter() - anchor_wall)
                        if sleep_time > 0.0:
                            time.sleep(sleep_time)
        finally:
            if log_file is not None:
                log_file.close()
        wall_elapsed = time.perf_counter() - wall_start
        ratio = self.data.time / wall_elapsed if wall_elapsed > 0.0 else 0.0
        print(f"Viewer run finished: sim={self.data.time:.2f}s wall={wall_elapsed:.2f}s ratio={ratio:.3f}x")

    def handle_key(self, keycode: int) -> None:
        if keycode == KEY_UP:
            self.command.nudge(0, 1.0)
        elif keycode == KEY_DOWN:
            self.command.nudge(0, -1.0)
        elif keycode == KEY_LEFT:
            self.command.nudge(2, 1.0)
        elif keycode == KEY_RIGHT:
            self.command.nudge(2, -1.0)
        elif keycode in (KEY_ENTER, KEY_KP_ENTER):
            self.command.set([0.0, 0.0, 0.0])
            self.reset()
            print("Reset")
        else:
            return
        print(f"Command: {self.command}")


def parse_segments(values: list[str] | None, default_duration: float) -> list[tuple[np.ndarray, float]]:
    """Parse --cmd entries of the form 'vx,vy,wz[,seconds]'."""
    if not values:
        return [(np.zeros(3, dtype=np.float32), default_duration)]
    segments = []
    for value in values:
        parts = [float(item) for item in value.split(",")]
        if len(parts) not in (3, 4):
            raise ValueError(f"--cmd expects vx,vy,wz[,seconds], got '{value}'")
        duration = parts[3] if len(parts) == 4 else default_duration
        segments.append((np.asarray(parts[:3], dtype=np.float32), duration))
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RW skater_v2 velocity policy MuJoCo sim2sim")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to rw_skater YAML config")
    parser.add_argument("--policy-name", default=None, help="Policy entry in the config (push_gait, swizzle)")
    parser.add_argument("--policy", default=None, help="Override exported ONNX policy path")
    parser.add_argument(
        "--cmd",
        action="append",
        default=None,
        help="Command 'vx,vy,wz[,seconds]'. Viewer uses the first as initial command; "
        "headless runs each as a segment (repeatable)",
    )
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration in seconds")
    parser.add_argument("--check", action="store_true", help="Validate model, observation layout, and ONNX")
    parser.add_argument("--headless", action="store_true", help="Run without MuJoCo viewer")
    parser.add_argument("--joystick", action="store_true", help="Use joystick axes for vx, vy, and yaw rate")
    parser.add_argument("--no-realtime", action="store_true", help="Do not sleep to wall-clock timestep")
    parser.add_argument("--no-log", action="store_true", help="Disable trajectory CSV logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config), args.policy_name, args.policy)
    if args.policy is not None:
        config["policy_path"] = str(resolve_path(args.policy))
    duration = float(args.duration if args.duration is not None else config["simulation_duration"])
    segments = parse_segments(args.cmd, 10.0 if args.headless and args.duration is None else duration)
    runtime = RwSkaterRuntime(config)
    if args.check:
        runtime.run_check()
    elif args.headless:
        if not runtime.run_headless(segments=segments, log=not args.no_log):
            sys.exit(1)
    else:
        runtime.command.set(segments[0][0])
        joystick = JoystickInput() if args.joystick else None
        try:
            realtime = bool(config.get("realtime", True)) and not args.no_realtime
            runtime.run_viewer(duration=duration, realtime=realtime, log=not args.no_log, joystick=joystick)
        finally:
            if joystick is not None:
                joystick.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
