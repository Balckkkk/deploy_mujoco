#!/usr/bin/env python3
"""G1 29-DoF AME policy deployment in MuJoCo.

This follows the control structure of ``r2_deploy.py`` while matching the AME
training observation exactly: 96 proprioceptive values followed by 2079 local
terrain-hit coordinates.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

from ame_heightmap import MujocoAmeHeightMap
from ame_terrain import TERRAIN_NAMES, build_mujoco_model_with_terrain, generate_ame_terrain


SCRIPT_DIR = Path(__file__).resolve().parent


def get_projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world gravity direction ``[0, 0, -1]`` into the base frame."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    return np.array(
        [
            2.0 * (w * y - z * x),
            -2.0 * (z * y + w * x),
            1.0 - 2.0 * (w * w + z * z),
        ],
        dtype=np.float32,
    )


def compute_heading_yaw_command(
    quaternion_wxyz: np.ndarray,
    *,
    target_heading: float,
    stiffness: float,
    limits: tuple[float, float] | list[float],
) -> float:
    """Match Isaac Lab's heading-to-yaw-rate command used by G1 PLAY."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    heading = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    heading_error = float(np.arctan2(np.sin(target_heading - heading), np.cos(target_heading - heading)))
    lower, upper = (float(value) for value in limits)
    return float(np.clip(stiffness * heading_error, lower, upper))


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp - dq * kd


def _resolve_project_path(path: str | os.PathLike[str]) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = SCRIPT_DIR / result
    return result.resolve()


def load_config(config_path: str, policy_override: str | None) -> tuple[dict[str, Any], Path]:
    path = _resolve_project_path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["xml_path"] = str(_resolve_project_path(config["xml_path"]))
    policy_path = _resolve_project_path(policy_override or config["policy_path"])
    return config, policy_path


class OnnxPolicy:
    """Single-environment AME ONNX inference wrapper."""

    def __init__(self, policy_path: Path, observation_dim: int) -> None:
        if not policy_path.is_file():
            raise FileNotFoundError(f"AME ONNX policy not found: {policy_path}")
        if policy_path.suffix.lower() != ".onnx":
            raise ValueError(f"AME deployment requires an exported .onnx policy, got: {policy_path}")
        self.session = ort.InferenceSession(
            str(policy_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"AME ONNX expects one input and one output, got {len(inputs)} inputs and {len(outputs)} outputs"
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.observation_dim = int(observation_dim)
        if inputs[0].shape[-1] != self.observation_dim:
            raise ValueError(
                f"AME ONNX input shape={inputs[0].shape}, expected [1, {self.observation_dim}]"
            )
        if outputs[0].shape[-1] != 29:
            raise ValueError(f"AME ONNX output shape={outputs[0].shape}, expected [1, 29]")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observation_batch = np.asarray(observation, dtype=np.float32).reshape(1, self.observation_dim)
        action = self.session.run(
            [self.output_name],
            {self.input_name: observation_batch},
        )[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (29,):
            raise ValueError(f"policy action shape={action.shape}, expected (29,)")
        if not np.all(np.isfinite(action)):
            raise ValueError("policy produced NaN or Inf actions")
        return action


def setup_joint_mappings(
    model: mujoco.MjModel,
    policy_joints: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return qpos, qvel, and actuator indices in policy joint order."""
    qpos_indices: list[int] = []
    dof_indices: list[int] = []
    actuator_indices: list[int] = []
    for joint_name in policy_joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Policy joint not found in MJCF: {joint_name}")
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if actuator_id < 0:
            raise ValueError(f"Actuator not found in MJCF: {joint_name}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        dof_indices.append(int(model.jnt_dofadr[joint_id]))
        actuator_indices.append(int(actuator_id))
    return (
        np.asarray(qpos_indices, dtype=np.int32),
        np.asarray(dof_indices, dtype=np.int32),
        np.asarray(actuator_indices, dtype=np.int32),
    )


def build_policy_observation(
    *,
    angular_velocity: np.ndarray,
    base_quaternion: np.ndarray,
    command: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    last_action: np.ndarray,
    default_angles: np.ndarray,
    height_observation: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Build AME terms in the order declared by ``PolicyCfg``."""
    proprio = np.concatenate(
        [
            angular_velocity * float(config["ang_vel_scale"]),
            get_projected_gravity(base_quaternion),
            command * np.asarray(config["cmd_scale"], dtype=np.float32),
            (joint_position - default_angles) * float(config["dof_pos_scale"]),
            joint_velocity * float(config["dof_vel_scale"]),
            last_action,
        ],
        dtype=np.float32,
    )
    if proprio.shape != (int(config["proprio_obs_dim"]),):
        raise ValueError(
            f"proprio observation shape={proprio.shape}, expected ({int(config['proprio_obs_dim'])},)"
        )
    if height_observation.shape != (int(config["height_obs_dim"]),):
        raise ValueError(
            f"height observation shape={height_observation.shape}, expected ({int(config['height_obs_dim'])},)"
        )
    observation = np.concatenate([proprio, height_observation], dtype=np.float32)
    if observation.shape != (int(config["num_obs"]),):
        raise ValueError(
            f"policy observation shape={observation.shape}, expected ({int(config['num_obs'])},)"
        )
    return observation


def print_heightmap_debug(scanner: MujocoAmeHeightMap, sim_time: float) -> None:
    scan = scanner.last_scan
    if scan is None:
        return
    cfg = scanner.config
    local_hits = scan["local_hits"]
    z_grid = local_hits[:, 2].reshape(cfg.num_y, cfg.num_x)
    valid_ratio = float(np.mean(scan["valid"]))
    raw_z = scan["raw_local_z"]
    clipped_low = int(np.count_nonzero(raw_z < cfg.min_local_z))
    clipped_high = int(np.count_nonzero(raw_z > cfg.max_local_z))
    print(
        f"[AME HEIGHTMAP] t={sim_time:.2f}s obs={scan['observation'].shape} "
        f"xyz={local_hits.shape} valid={valid_ratio * 100.0:.1f}% "
        f"raw_z=[{raw_z.min():.4f}, {raw_z.max():.4f}] "
        f"z=[{z_grid.min():.4f}, {z_grid.max():.4f}] "
        f"clipped(low/high)={clipped_low}/{clipped_high}"
    )
    print("[AME HEIGHTMAP] rows y=-0.5..0.5 m, columns x=-0.8..0.8 m")
    print(np.array2string(z_grid, precision=4, suppress_small=True, max_line_width=240))


def draw_heightmap_debug(
    viewer: Any,
    scanner: MujocoAmeHeightMap,
    show_ray_lines: bool,
) -> None:
    scan = scanner.last_scan
    if viewer.user_scn is None or scan is None:
        return
    rotation = np.eye(3, dtype=np.float64).reshape(-1)
    point_size = np.full(3, 0.014, dtype=np.float64)
    zero_size = np.zeros(3, dtype=np.float64)
    valid_color = np.array([0.0, 0.9, 0.2, 0.9], dtype=np.float32)
    miss_color = np.array([1.0, 0.1, 0.0, 0.9], dtype=np.float32)
    ray_color = np.array([0.0, 0.55, 1.0, 0.22], dtype=np.float32)

    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        for start, hit, valid in zip(
            scan["ray_starts_w"], scan["hit_points_w"], scan["valid"], strict=True
        ):
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                point_size,
                hit,
                rotation,
                valid_color if valid else miss_color,
            )
            scene.ngeom += 1
            if show_ray_lines and scene.ngeom < scene.maxgeom:
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    zero_size,
                    np.zeros(3, dtype=np.float64),
                    rotation,
                    ray_color,
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.0, start, hit)
                scene.ngeom += 1


class JoystickCommand:
    """Optional polling joystick; axes match the existing R2 deployment."""

    def __init__(self, initial: np.ndarray) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            pygame.quit()
            raise RuntimeError("No joystick device found")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.command = initial.astype(np.float32, copy=True)

    def update(self) -> np.ndarray:
        self.pygame.event.pump()
        self.command[0] = -self.joystick.get_axis(1)
        self.command[1] = -self.joystick.get_axis(0)
        self.command[2] = -self.joystick.get_axis(3)
        return self.command

    def close(self) -> None:
        self.pygame.quit()


def _prepare_runtime(
    config: dict[str, Any],
    policy_path: Path,
) -> dict[str, Any]:
    terrain = generate_ame_terrain(
        str(config.get("terrain", "flat")),
        difficulty=float(config.get("terrain_difficulty", 0.5)),
        seed=int(config.get("terrain_seed", 42)),
    )
    model = build_mujoco_model_with_terrain(config["xml_path"], terrain)
    data = mujoco.MjData(model)
    model.opt.timestep = float(config["simulation_dt"])

    joint_names = list(config["policy_joints"])
    num_actions = int(config["num_actions"])
    if len(joint_names) != num_actions:
        raise ValueError(f"policy_joints has {len(joint_names)} entries, expected {num_actions}")
    qpos_indices, dof_indices, actuator_indices = setup_joint_mappings(model, joint_names)

    arrays = {
        name: np.asarray(config[name], dtype=np.float32)
        for name in ("default_angles", "kps", "kds", "effort_limits")
    }
    for name, values in arrays.items():
        if values.shape != (num_actions,):
            raise ValueError(f"{name} shape={values.shape}, expected ({num_actions},)")

    # Isaac Lab config uses 0.01 armature on every actuated joint.
    model.dof_armature[dof_indices] = float(config.get("armature", 0.01))
    data.qpos[:] = model.qpos0
    data.qpos[:3] = np.asarray(config["base_init_pos"], dtype=np.float64)
    data.qpos[3:7] = np.array([0.6, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[qpos_indices] = arrays["default_angles"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    scanner = MujocoAmeHeightMap(model, data)
    height_observation = scanner.scan()
    policy = OnnxPolicy(policy_path, int(config["num_obs"]))
    policy(np.zeros(int(config["num_obs"]), dtype=np.float32))

    return {
        "model": model,
        "data": data,
        "qpos_indices": qpos_indices,
        "dof_indices": dof_indices,
        "actuator_indices": actuator_indices,
        "scanner": scanner,
        "height_observation": height_observation,
        "policy": policy,
        "terrain": terrain,
        **arrays,
    }


def run_check(config: dict[str, Any], policy_path: Path) -> None:
    runtime = _prepare_runtime(config, policy_path)
    action = np.zeros(29, dtype=np.float32)
    data = runtime["data"]
    observation = build_policy_observation(
        angular_velocity=data.qvel[3:6],
        base_quaternion=data.qpos[3:7],
        command=np.asarray(config["command"], dtype=np.float32),
        joint_position=data.qpos[runtime["qpos_indices"]],
        joint_velocity=data.qvel[runtime["dof_indices"]],
        last_action=action,
        default_angles=runtime["default_angles"],
        height_observation=runtime["height_observation"],
        config=config,
    )
    action = runtime["policy"](observation)
    scanner = runtime["scanner"]
    print(f"Model: nq={runtime['model'].nq}, nv={runtime['model'].nv}, nu={runtime['model'].nu}")
    print(f"Policy: {policy_path} (ONNX Runtime)")
    terrain = runtime["terrain"]
    print(
        f"Terrain: {terrain.name}, difficulty={terrain.difficulty:.2f}, seed={terrain.seed}, "
        f"size={terrain.size:.1f}m, z=[{terrain.min_height:.3f}, {terrain.max_height:.3f}]"
    )
    print(f"Observation: {observation.shape}, action: {action.shape}")
    print(
        f"Action stats: min={action.min():.4f}, max={action.max():.4f}, "
        f"mean={action.mean():.4f}"
    )
    print_heightmap_debug(scanner, data.time)
    print("AME deployment check passed.")


def run_simulation(
    config: dict[str, Any],
    policy_path: Path,
    *,
    command_override: list[float] | None,
    use_joystick: bool,
    heightmap_debug: bool,
    heightmap_rays: bool,
    heightmap_print_interval: float,
    duration_override: float | None,
) -> None:
    runtime = _prepare_runtime(config, policy_path)
    model: mujoco.MjModel = runtime["model"]
    data: mujoco.MjData = runtime["data"]
    scanner: MujocoAmeHeightMap = runtime["scanner"]
    qpos_indices = runtime["qpos_indices"]
    dof_indices = runtime["dof_indices"]
    actuator_indices = runtime["actuator_indices"]
    default_angles = runtime["default_angles"]
    action_scale = float(config["action_scale"])
    control_decimation = int(config["control_decimation"])
    duration = float(duration_override or config["simulation_duration"])

    command = np.asarray(command_override or config["command"], dtype=np.float32).copy()
    joystick = JoystickCommand(command) if use_joystick else None
    heading_command_enabled = bool(config.get("heading_command", False)) and command_override is None and not use_joystick
    action = np.zeros(29, dtype=np.float32)
    target_q = default_angles.copy()
    height_observation = runtime["height_observation"]
    counter = 0
    last_heightmap_print = -np.inf

    print(f"Loaded G1 model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"Loaded AME policy: {policy_path} (ONNX Runtime)")
    terrain = runtime["terrain"]
    print(
        f"Loaded AME terrain: {terrain.name}, difficulty={terrain.difficulty:.2f}, "
        f"seed={terrain.seed}, z=[{terrain.min_height:.3f}, {terrain.max_height:.3f}]"
    )
    print(
        f"Observation: {int(config['proprio_obs_dim'])} proprio + "
        f"{int(config['height_obs_dim'])} terrain xyz = {int(config['num_obs'])}; action: 29"
    )
    print(
        f"Height map: {scanner.config.num_y}x{scanner.config.num_x}, "
        f"resolution={scanner.config.resolution:.2f} m, body={scanner.config.body_name}"
    )
    if heading_command_enabled:
        print(
            "Command mode: heading tracking, "
            f"target={float(config.get('heading_target', 0.0)):.3f} rad, "
            f"stiffness={float(config.get('heading_control_stiffness', 0.5)):.3f}"
        )
    else:
        print("Command mode: direct vx, vy, yaw-rate")
    print_heightmap_debug(scanner, data.time)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall_start = time.perf_counter()
            while viewer.is_running() and data.time < duration:
                step_start = time.perf_counter()
                q = data.qpos[qpos_indices].copy()
                dq = data.qvel[dof_indices].copy()
                torque = pd_control(target_q, q, runtime["kps"], dq, runtime["kds"])
                torque = np.clip(torque, -runtime["effort_limits"], runtime["effort_limits"])
                data.ctrl[actuator_indices] = torque
                mujoco.mj_step(model, data)
                counter += 1

                if counter % control_decimation == 0:
                    if joystick is not None:
                        command = joystick.update()
                    elif heading_command_enabled:
                        command[2] = compute_heading_yaw_command(
                            data.qpos[3:7],
                            target_heading=float(config.get("heading_target", 0.0)),
                            stiffness=float(config.get("heading_control_stiffness", 0.5)),
                            limits=config.get("yaw_command_limits", [-1.0, 1.0]),
                        )
                    height_observation = scanner.scan()
                    observation = build_policy_observation(
                        angular_velocity=data.qvel[3:6],
                        base_quaternion=data.qpos[3:7],
                        command=command,
                        joint_position=data.qpos[qpos_indices],
                        joint_velocity=data.qvel[dof_indices],
                        last_action=action,
                        default_angles=default_angles,
                        height_observation=height_observation,
                        config=config,
                    )
                    action = runtime["policy"](observation)
                    target_q = default_angles + action_scale * action

                    if (
                        heightmap_debug
                        and heightmap_print_interval > 0.0
                        and data.time - last_heightmap_print >= heightmap_print_interval
                    ):
                        print_heightmap_debug(scanner, data.time)
                        last_heightmap_print = data.time

                if counter % 2 == 0:
                    if heightmap_debug:
                        draw_heightmap_debug(viewer, scanner, heightmap_rays)
                    viewer.sync()

                elapsed = time.perf_counter() - step_start
                sleep_time = model.opt.timestep - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            wall_elapsed = time.perf_counter() - wall_start
            ratio = data.time / wall_elapsed if wall_elapsed > 0.0 else 0.0
            print(f"Simulation finished: sim={data.time:.2f}s wall={wall_elapsed:.2f}s ratio={ratio:.3f}x")
    finally:
        if joystick is not None:
            joystick.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 29-DoF AME MuJoCo sim2sim deployment")
    parser.add_argument("--config", default="config/ame_g1.yaml", help="AME deployment YAML")
    parser.add_argument("--policy", default=None, help="Exported AME ONNX policy path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate model, scanner, observation, and policy without GUI",
    )
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration in seconds")
    parser.add_argument(
        "--terrain",
        choices=TERRAIN_NAMES,
        default=None,
        help="AME training terrain tile; defaults to the YAML setting",
    )
    parser.add_argument(
        "--terrain-difficulty",
        type=float,
        default=None,
        help="Terrain difficulty in [0, 1]",
    )
    parser.add_argument("--terrain-seed", type=int, default=None, help="Deterministic terrain seed")
    parser.add_argument("--cmd", nargs=3, type=float, metavar=("VX", "VY", "WZ"), help="Override velocity command")
    parser.add_argument("--joystick", action="store_true", help="Use joystick axes for vx, vy, and yaw rate")
    parser.add_argument(
        "--heightmap-debug",
        action="store_true",
        help="Draw hits and periodically print the 21x33 z grid",
    )
    parser.add_argument("--heightmap-rays", action="store_true", help="Also draw the 20 m vertical rays")
    parser.add_argument(
        "--heightmap-print-interval",
        type=float,
        default=1.0,
        help="Simulation-time interval for printing terrain values; <=0 disables periodic printing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, policy_path = load_config(args.config, args.policy)
    if args.terrain is not None:
        config["terrain"] = args.terrain
    if args.terrain_difficulty is not None:
        config["terrain_difficulty"] = args.terrain_difficulty
    if args.terrain_seed is not None:
        config["terrain_seed"] = args.terrain_seed
    if args.check:
        run_check(config, policy_path)
        return
    run_simulation(
        config,
        policy_path,
        command_override=args.cmd,
        use_joystick=args.joystick,
        heightmap_debug=args.heightmap_debug,
        heightmap_rays=args.heightmap_rays,
        heightmap_print_interval=args.heightmap_print_interval,
        duration_override=args.duration,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
