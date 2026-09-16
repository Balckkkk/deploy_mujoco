#!/usr/bin/env python3
"""Run the distilled G1 depth policy in the matched MuJoCo scene."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Select a headless backend before importing MuJoCo. GUI runs keep MuJoCo's
# default GLFW backend.
if "--headless" in sys.argv or "--check" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml

from depth_camera import DepthCameraConfig, MujocoDepthCamera, colorize_policy_depth
from g1_obstacle_scene import ObstacleSceneConfig, build_obstacle_model, reset_obstacles
from g1_perceptive_policy import ACTION_DIM, G1PerceptivePolicy, build_proprioception


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config/g1_perceptive.yaml"
INFERENCE_MODES = ("vault", "climb", "sequential")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("model_path", "isaac_reference_path", "policy_checkpoint"):
        config[key] = str(resolve_path(config[key]))
    config["motion_files"] = {
        key: str(resolve_path(value)) for key, value in config["motion_files"].items()
    }
    return config


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Rotate the unit world gravity vector into the torso frame."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    return np.array(
        (
            2.0 * (w * y - z * x),
            -2.0 * (z * y + w * x),
            1.0 - 2.0 * (w * w + z * z),
        ),
        dtype=np.float32,
    )


def setup_joint_mappings(
    model: mujoco.MjModel, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    actuator_indices: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise ValueError(f"Missing joint/actuator mapping for {name}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        actuator_indices.append(int(actuator_id))
    return (
        np.asarray(qpos_indices, dtype=np.int32),
        np.asarray(qvel_indices, dtype=np.int32),
        np.asarray(actuator_indices, dtype=np.int32),
    )


class MotionInitialization:
    """Initial state and baked high-level command from a training motion NPZ."""

    def __init__(self, path: str | Path) -> None:
        self.path = resolve_path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Motion file not found: {self.path}")
        with np.load(self.path, allow_pickle=False) as data:
            self.framerate = float(data["framerate"])
            self.joint_names = [str(name) for name in data["joint_names"].tolist()]
            self.initial_joint_position = np.asarray(data["joint_pos"][0], dtype=np.float64)
            self.initial_base_position = np.asarray(data["base_pos_w"][0], dtype=np.float64)
            self.initial_base_quaternion = np.asarray(data["base_quat_w"][0], dtype=np.float64)
            self.velocity_command = np.asarray(data["velocity_command"], dtype=np.float32).copy()
        if self.velocity_command.ndim != 2 or self.velocity_command.shape[1] != 3:
            raise ValueError(f"Invalid velocity command shape in {self.path}: {self.velocity_command.shape}")

    @property
    def duration(self) -> float:
        return self.velocity_command.shape[0] / self.framerate

    def command_at(self, sim_time: float) -> np.ndarray:
        index = int(np.clip(round(sim_time * self.framerate), 0, len(self.velocity_command) - 1))
        return self.velocity_command[index].copy()


class G1Sim2SimRuntime:
    """Own all model, sensor, policy, mapping, and controller state."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        mode: str,
        checkpoint: Path | None,
        randomize_obstacle: bool,
        seed: int,
        device: str,
        command_override: np.ndarray | None,
    ) -> None:
        if mode not in INFERENCE_MODES:
            raise ValueError(f"Inference mode must be one of {INFERENCE_MODES}, got {mode}")
        self.config = config
        self.mode = mode
        self.command_override = command_override
        self.scene_config = ObstacleSceneConfig.from_dict(config["obstacle_scene"])
        self.model = build_obstacle_model(
            config["model_path"],
            self.scene_config,
            mode,
            randomize=randomize_obstacle,
            seed=seed,
        )
        self.model.opt.timestep = float(config["sim_dt"])
        self.data = mujoco.MjData(self.model)
        self.poses = reset_obstacles(
            self.model,
            self.data,
            self.scene_config,
            mode,
            randomize=randomize_obstacle,
            seed=seed,
        )

        reference = json.loads(Path(config["isaac_reference_path"]).read_text(encoding="utf-8"))
        self.joint_names = list(config["policy_joint_names"])
        if self.joint_names != reference["joint_names"]:
            raise ValueError("YAML policy joint order differs from captured Isaac articulation order")
        if len(self.joint_names) != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM} policy joints, got {len(self.joint_names)}")
        self.qpos_indices, self.qvel_indices, self.actuator_indices = setup_joint_mappings(
            self.model, self.joint_names
        )
        joint = reference["joint"]
        self.default_position = np.asarray(joint["default_position"], dtype=np.float32)
        self.kp = np.asarray(joint["stiffness"], dtype=np.float32)
        self.kd = np.asarray(joint["damping"], dtype=np.float32)
        self.effort_limit = np.asarray(joint["effort_limit"], dtype=np.float32)
        for name, array in (
            ("default_position", self.default_position),
            ("kp", self.kp),
            ("kd", self.kd),
            ("effort_limit", self.effort_limit),
        ):
            if array.shape != (ACTION_DIM,):
                raise ValueError(f"{name} shape={array.shape}, expected {(ACTION_DIM,)}")

        source_motion = "climb" if mode == "climb" else "vault"
        self.motion = MotionInitialization(config["motion_files"][source_motion])
        self._reset_robot_from_motion()

        checkpoint_path = checkpoint or Path(config["policy_checkpoint"])
        self.policy = G1PerceptivePolicy(
            checkpoint_path,
            device=device,
            normalizer_epsilon=float(config["policy"]["normalizer_epsilon"]),
        )
        self.camera = MujocoDepthCamera(self.model, DepthCameraConfig.from_dict(config["depth_camera"]))
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, config["base_body"]
        )
        if self.base_body_id < 0:
            raise ValueError(f"Base body not found: {config['base_body']}")
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.target_position = self.default_position.copy()
        self.last_depth = None
        self.last_command = self.command_at(0.0)

    def _reset_robot_from_motion(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.motion.initial_base_position
        self.data.qpos[3:7] = self.motion.initial_base_quaternion
        motion_index = {name: index for index, name in enumerate(self.motion.joint_names)}
        missing = [name for name in self.joint_names if name not in motion_index]
        if missing:
            raise ValueError(f"Motion is missing policy joints: {missing}")
        self.data.qpos[self.qpos_indices] = np.asarray(
            [self.motion.initial_joint_position[motion_index[name]] for name in self.joint_names],
            dtype=np.float64,
        )
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def command_at(self, sim_time: float) -> np.ndarray:
        if self.command_override is not None:
            return self.command_override.copy()
        if self.mode == "sequential":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return self.motion.command_at(sim_time)

    def base_angular_velocity(self) -> np.ndarray:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            velocity,
            1,
        )
        return velocity[:3].astype(np.float32)

    def policy_step(self) -> tuple[np.ndarray, np.ndarray]:
        frame = self.camera.capture(self.data)
        command = self.command_at(float(self.data.time))
        joint_position = self.data.qpos[self.qpos_indices].astype(np.float32)
        joint_velocity = self.data.qvel[self.qvel_indices].astype(np.float32)
        proprio = build_proprioception(
            command,
            projected_gravity(self.data.qpos[3:7]),
            self.base_angular_velocity(),
            joint_position - self.default_position,
            joint_velocity,
            self.last_action,
        )
        action = self.policy.infer(frame.policy_image, proprio)
        self.last_action = action
        self.target_position = self.default_position + float(
            self.config["policy"]["action_scale"]
        ) * action
        self.last_depth = frame
        self.last_command = command
        return action, proprio

    def physics_step(self) -> np.ndarray:
        position = self.data.qpos[self.qpos_indices]
        velocity = self.data.qvel[self.qvel_indices]
        torque = self.kp * (self.target_position - position) - self.kd * velocity
        torque = np.clip(torque, -self.effort_limit, self.effort_limit)
        self.data.ctrl[self.actuator_indices] = torque
        mujoco.mj_step(self.model, self.data)
        return torque.astype(np.float32)

    def close(self) -> None:
        self.camera.close()


def status_row(runtime: G1Sim2SimRuntime, torque: np.ndarray) -> dict[str, float]:
    quaternion = runtime.data.qpos[3:7]
    gravity = projected_gravity(quaternion)
    return {
        "time": float(runtime.data.time),
        "base_x": float(runtime.data.qpos[0]),
        "base_y": float(runtime.data.qpos[1]),
        "base_z": float(runtime.data.qpos[2]),
        "tilt": float(np.arccos(np.clip(-gravity[2], -1.0, 1.0))),
        "command_x": float(runtime.last_command[0]),
        "action_min": float(runtime.last_action.min()),
        "action_max": float(runtime.last_action.max()),
        "torque_max_abs": float(np.max(np.abs(torque))),
        "depth_min": float(runtime.last_depth.depth_m.min()) if runtime.last_depth else np.nan,
        "depth_mean": float(runtime.last_depth.depth_m.mean()) if runtime.last_depth else np.nan,
    }


def update_depth_overlay(viewer: Any, runtime: G1Sim2SimRuntime) -> None:
    """Show the exact policy depth frame in the lower-right of the main viewer."""
    if runtime.last_depth is None:
        return
    viewport_width = int(viewer.viewport.width)
    overlay_width = min(480, max(192, viewport_width // 3))
    overlay_height = int(round(overlay_width * runtime.last_depth.policy_image.shape[1] / runtime.last_depth.policy_image.shape[2]))
    margin = 12
    viewport = mujoco.MjrRect(
        max(margin, viewport_width - overlay_width - margin),
        margin,
        overlay_width,
        overlay_height,
    )
    rgb = colorize_policy_depth(runtime.last_depth.policy_image)
    row_indices = np.linspace(0, rgb.shape[0] - 1, overlay_height).astype(np.int32)
    column_indices = np.linspace(0, rgb.shape[1] - 1, overlay_width).astype(np.int32)
    rgb_resized = rgb[row_indices[:, None], column_indices[None, :]]
    viewer.set_images((viewport, rgb_resized))


def run(
    runtime: G1Sim2SimRuntime,
    *,
    duration: float,
    headless: bool,
    realtime: bool,
    output_dir: Path,
    stop_on_fall: bool,
    show_depth: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict[str, float]] = []
    torque = np.zeros(ACTION_DIM, dtype=np.float32)
    control_decimation = int(runtime.config["control_decimation"])
    runtime.policy_step()
    log_rows.append(status_row(runtime, torque))
    min_base_height = float(runtime.data.qpos[2])
    fell = False
    viewer = None
    if not headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(runtime.model, runtime.data)
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[0] = 1
        viewer.opt.geomgroup[2] = 1
        viewer.cam.distance = 7.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -12.0
        if show_depth:
            update_depth_overlay(viewer, runtime)

    wall_start = time.perf_counter()
    physics_steps = 0
    next_print = 0.0
    try:
        while runtime.data.time < duration and (viewer is None or viewer.is_running()):
            step_start = time.perf_counter()
            torque = runtime.physics_step()
            physics_steps += 1
            if physics_steps % control_decimation == 0:
                runtime.policy_step()
                if viewer is not None and show_depth:
                    update_depth_overlay(viewer, runtime)
                row = status_row(runtime, torque)
                log_rows.append(row)
                min_base_height = min(min_base_height, row["base_z"])
                if runtime.data.time + 1e-9 >= next_print:
                    print(
                        f"t={row['time']:6.2f}s x={row['base_x']:7.3f} z={row['base_z']:.3f} "
                        f"cmd={row['command_x']:.1f} tilt={row['tilt']:.2f} "
                        f"depth=[{row['depth_min']:.2f},{row['depth_mean']:.2f}]"
                    )
                    next_print += 1.0
                fell = runtime.data.time > 0.5 and row["base_z"] < 0.35
                if fell and stop_on_fall:
                    print("Fall threshold reached (torso z < 0.35 m); stopping early.")
                    break
            if viewer is not None:
                viewer.cam.lookat[:] = [float(runtime.data.qpos[0]) + 0.8, 0.0, 0.55]
                viewer.sync()
            if realtime:
                remaining = runtime.model.opt.timestep - (time.perf_counter() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()

    wall_time = time.perf_counter() - wall_start
    trajectory_path = output_dir / "trajectory.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    if runtime.last_depth is not None:
        np.save(output_dir / "final_depth_m.npy", runtime.last_depth.depth_m)
        np.save(output_dir / "final_depth_policy.npy", runtime.last_depth.policy_image)

    active = [pose.position[0] for pose in runtime.poses.values() if pose.active]
    furthest_obstacle = float(max(active)) if active else float("nan")
    summary = {
        "mode": runtime.mode,
        "checkpoint": str(runtime.policy.checkpoint_path),
        "checkpoint_iteration": runtime.policy.iteration,
        "simulated_seconds": float(runtime.data.time),
        "wall_seconds": wall_time,
        "realtime_factor": float(runtime.data.time / wall_time) if wall_time > 0.0 else 0.0,
        "initial_base_x": float(log_rows[0]["base_x"]),
        "final_base_x": float(runtime.data.qpos[0]),
        "final_base_y": float(runtime.data.qpos[1]),
        "final_base_z": float(runtime.data.qpos[2]),
        "minimum_base_z": min_base_height,
        "fell": fell,
        "furthest_active_obstacle_x": furthest_obstacle,
        "passed_furthest_obstacle": bool(runtime.data.qpos[0] > furthest_obstacle),
        "trajectory": str(trajectory_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=INFERENCE_MODES, default="vault")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--cmd", nargs=3, type=float, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--randomize-obstacle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Overlay the exact 32x48 policy depth image in the MuJoCo viewer.",
    )
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--keep-running-after-fall", action="store_true")
    parser.add_argument("--check", action="store_true", help="Load and run one policy tick, then exit.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless and args.show_depth:
        raise ValueError("--show-depth requires the interactive viewer; omit --headless")
    config = load_config(args.config.resolve())
    command_override = np.asarray(args.cmd, dtype=np.float32) if args.cmd is not None else None
    runtime = G1Sim2SimRuntime(
        config,
        mode=args.mode,
        checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
        randomize_obstacle=args.randomize_obstacle,
        seed=args.seed,
        device=args.device,
        command_override=command_override,
    )
    print(
        f"Loaded model nq={runtime.model.nq}, nv={runtime.model.nv}, nu={runtime.model.nu}; "
        f"policy iteration={runtime.policy.iteration}; mode={args.mode}"
    )
    print(
        f"Reset from {runtime.motion.path.name}: base={runtime.data.qpos[:3].round(4).tolist()}, "
        f"motion_duration={runtime.motion.duration:.2f}s"
    )
    try:
        if args.check:
            action, proprio = runtime.policy_step()
            print(
                f"Check passed: depth={runtime.last_depth.policy_image.shape}, proprio={proprio.shape}, "
                f"action={action.shape}, action_range=[{action.min():.4f}, {action.max():.4f}]"
            )
            return
        duration = args.duration
        if duration is None:
            duration = 24.0 if args.mode == "sequential" else runtime.motion.duration
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else (PROJECT_ROOT / "sim2sim_output" / args.mode).resolve()
        )
        summary = run(
            runtime,
            duration=float(duration),
            headless=args.headless,
            realtime=not args.no_realtime,
            output_dir=output_dir,
            stop_on_fall=not args.keep_running_after_fall,
            show_depth=args.show_depth,
        )
        print(json.dumps(summary, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
