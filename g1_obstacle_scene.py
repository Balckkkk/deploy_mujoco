"""Training-compatible vault/climb scene construction for MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np


SceneMode = Literal["vault", "climb", "both", "sequential"]


@dataclass(frozen=True)
class ObstacleConfig:
    key: str
    body_name: str
    geom_name: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    hidden_position: tuple[float, float, float]
    friction: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    randomization: dict[str, tuple[float, float]]
    sequential_position: tuple[float, float, float] | None = None

    @classmethod
    def from_dict(cls, key: str, values: dict[str, Any]) -> "ObstacleConfig":
        return cls(
            key=key,
            body_name=str(values["body_name"]),
            geom_name=str(values["geom_name"]),
            size=tuple(float(value) for value in values["size"]),
            position=tuple(float(value) for value in values["position"]),
            hidden_position=tuple(float(value) for value in values["hidden_position"]),
            friction=tuple(float(value) for value in values["friction"]),
            rgba=tuple(float(value) for value in values["rgba"]),
            randomization={
                axis: tuple(float(value) for value in bounds)
                for axis, bounds in values["randomization"].items()
            },
            sequential_position=(
                tuple(float(value) for value in values["sequential_position"])
                if "sequential_position" in values
                else None
            ),
        )


@dataclass(frozen=True)
class SceneLightConfig:
    name: str
    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    ambient: tuple[float, float, float]
    diffuse: tuple[float, float, float]
    specular: tuple[float, float, float]
    cast_shadow: bool

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SceneLightConfig":
        return cls(
            name=str(values["name"]),
            position=tuple(float(value) for value in values["position"]),
            direction=tuple(float(value) for value in values["direction"]),
            ambient=tuple(float(value) for value in values["ambient"]),
            diffuse=tuple(float(value) for value in values["diffuse"]),
            specular=tuple(float(value) for value in values["specular"]),
            cast_shadow=bool(values.get("cast_shadow", True)),
        )


@dataclass(frozen=True)
class ObstacleSceneConfig:
    ground_name: str
    ground_size: tuple[float, float, float]
    ground_friction: tuple[float, float, float]
    ground_rgba: tuple[float, float, float, float]
    headlight_ambient: tuple[float, float, float]
    headlight_diffuse: tuple[float, float, float]
    headlight_specular: tuple[float, float, float]
    lights: tuple[SceneLightConfig, ...]
    obstacles: dict[str, ObstacleConfig]

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ObstacleSceneConfig":
        ground = values["ground"]
        lighting = values.get("lighting", {})
        headlight = lighting.get("headlight", {})
        return cls(
            ground_name=str(ground["name"]),
            ground_size=tuple(float(value) for value in ground.get("size", (10.0, 10.0, 0.1))),
            ground_friction=tuple(float(value) for value in ground["friction"]),
            ground_rgba=tuple(float(value) for value in ground["rgba"]),
            headlight_ambient=tuple(
                float(value) for value in headlight.get("ambient", (0.1, 0.1, 0.1))
            ),
            headlight_diffuse=tuple(
                float(value) for value in headlight.get("diffuse", (0.4, 0.4, 0.4))
            ),
            headlight_specular=tuple(
                float(value) for value in headlight.get("specular", (0.5, 0.5, 0.5))
            ),
            lights=tuple(
                SceneLightConfig.from_dict(light) for light in lighting.get("lights", ())
            ),
            obstacles={
                key: ObstacleConfig.from_dict(key, obstacle)
                for key, obstacle in values["obstacles"].items()
            },
        )


@dataclass(frozen=True)
class ObstaclePose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    active: bool


def _active_obstacles(mode: SceneMode) -> set[str]:
    if mode == "vault":
        return {"vault"}
    if mode == "climb":
        return {"climb"}
    if mode in {"both", "sequential"}:
        return {"vault", "climb"}
    raise ValueError(f"Unsupported scene mode: {mode}")


def sample_obstacle_poses(
    config: ObstacleSceneConfig,
    mode: SceneMode,
    *,
    randomize: bool = False,
    seed: int = 42,
) -> dict[str, ObstaclePose]:
    active = _active_obstacles(mode)
    generator = np.random.default_rng(seed)
    result: dict[str, ObstaclePose] = {}
    for key, obstacle in config.obstacles.items():
        is_active = key in active
        if not is_active:
            result[key] = ObstaclePose(
                position=np.asarray(obstacle.hidden_position, dtype=np.float64),
                quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                active=False,
            )
            continue
        nominal = obstacle.position
        if mode == "sequential" and obstacle.sequential_position is not None:
            nominal = obstacle.sequential_position
        offset = np.zeros(6, dtype=np.float64)
        if randomize:
            for index, axis in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
                bounds = obstacle.randomization[axis]
                offset[index] = generator.uniform(bounds[0], bounds[1])
        position = np.asarray(nominal, dtype=np.float64) + offset[:3]
        roll, pitch, yaw = offset[3:]
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        quaternion = np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
        result[key] = ObstaclePose(position=position, quaternion_wxyz=quaternion, active=True)
    return result


def build_obstacle_model(
    robot_model_path: str | Path,
    config: ObstacleSceneConfig,
    mode: SceneMode,
    *,
    randomize: bool = False,
    seed: int = 42,
) -> mujoco.MjModel:
    """Compile the robot, plane, and two kinematic obstacle bodies."""
    poses = sample_obstacle_poses(config, mode, randomize=randomize, seed=seed)
    spec = mujoco.MjSpec.from_file(str(Path(robot_model_path).resolve()))
    spec.visual.headlight.ambient = config.headlight_ambient
    spec.visual.headlight.diffuse = config.headlight_diffuse
    spec.visual.headlight.specular = config.headlight_specular
    spec.worldbody.add_geom(
        name=config.ground_name,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=config.ground_size,
        friction=config.ground_friction,
        rgba=config.ground_rgba,
        group=0,
        condim=3,
    )
    for light in config.lights:
        spec.worldbody.add_light(
            name=light.name,
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            pos=light.position,
            dir=light.direction,
            ambient=light.ambient,
            diffuse=light.diffuse,
            specular=light.specular,
            castshadow=light.cast_shadow,
        )
    for key, obstacle in config.obstacles.items():
        pose = poses[key]
        body = spec.worldbody.add_body(
            name=obstacle.body_name,
            pos=pose.position,
            quat=pose.quaternion_wxyz,
            mocap=True,
        )
        body.add_geom(
            name=obstacle.geom_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=np.asarray(obstacle.size) / 2.0,
            friction=obstacle.friction,
            rgba=obstacle.rgba,
            group=0,
            condim=3,
        )
    model = spec.compile()
    # Isaac's ray caster is unaffected by the inactive obstacle parked at
    # (1000, 1000, -100). MuJoCo otherwise includes that body in auto-computed
    # camera clipping planes, which would move the near plane beyond the active
    # obstacle. Use explicit scene-scale clipping while preserving the exact
    # training hidden pose.
    model.stat.center[:] = [3.5, 0.0, 0.5]
    model.stat.extent = 10.0
    model.vis.map.znear = 0.005  # 0.05 m because znear is relative to extent.
    model.vis.map.zfar = 10.0  # 100 m; policy preprocessing clips at 3 m.
    return model


def reset_obstacles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: ObstacleSceneConfig,
    mode: SceneMode,
    *,
    randomize: bool = False,
    seed: int = 42,
) -> dict[str, ObstaclePose]:
    """Reset mocap obstacle poses without rebuilding the MuJoCo model."""
    poses = sample_obstacle_poses(config, mode, randomize=randomize, seed=seed)
    for key, obstacle in config.obstacles.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obstacle.body_name)
        if body_id < 0:
            raise ValueError(f"Obstacle body not found: {obstacle.body_name}")
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"Obstacle body is not a mocap body: {obstacle.body_name}")
        data.mocap_pos[mocap_id] = poses[key].position
        data.mocap_quat[mocap_id] = poses[key].quaternion_wxyz
    mujoco.mj_forward(model, data)
    return poses
