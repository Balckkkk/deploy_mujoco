"""MuJoCo height scanner matching the G1 AME Isaac Lab training setup.

The training ray caster is attached to ``torso_link``, aligned with torso yaw,
and casts a 21 x 33 grid of vertical rays against the ground only.  Each
sample is represented by its local ``(x, y, z)`` hit coordinate, so the policy
receives 693 * 3 = 2079 terrain values.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AmeHeightMapConfig:
    """Geometry and observation semantics copied from AME training."""

    body_name: str = "torso_link"
    size_x: float = 1.6
    size_y: float = 1.0
    resolution: float = 0.05
    ray_start_height: float = 20.0
    min_local_z: float = -1.2
    max_local_z: float = 0.0
    terrain_group: int = 0
    robot_group: int = 1

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.size_x <= 0.0 or self.size_y <= 0.0:
            raise ValueError("grid sizes must be positive")
        if self.ray_start_height <= 0.0:
            raise ValueError("ray_start_height must be positive")
        if self.min_local_z > self.max_local_z:
            raise ValueError("min_local_z must not exceed max_local_z")

    @staticmethod
    def _point_count(size: float, resolution: float) -> int:
        intervals = size / resolution
        rounded = round(intervals)
        if not np.isclose(intervals, rounded, atol=1.0e-8):
            raise ValueError("grid size must be an integer multiple of resolution")
        return int(rounded) + 1

    @property
    def num_x(self) -> int:
        return self._point_count(self.size_x, self.resolution)

    @property
    def num_y(self) -> int:
        return self._point_count(self.size_y, self.resolution)

    @property
    def num_rays(self) -> int:
        return self.num_x * self.num_y

    @property
    def observation_dim(self) -> int:
        return self.num_rays * 3


class MujocoAmeHeightMap:
    """Cast the terrain rays and return the AME 2079-D terrain observation.

    MuJoCo ray groups are prepared so world-body geometry is in the terrain
    group and robot geometry is in another group.  This is necessary because
    ``mj_ray`` can otherwise hit the G1 itself, while the Isaac Lab training
    sensor explicitly casts only against ``/World/ground``.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        config: AmeHeightMapConfig | None = None,
        *,
        mujoco_module: Any | None = None,
        prepare_geom_groups: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or AmeHeightMapConfig()
        self.mujoco = mujoco_module or importlib.import_module("mujoco")

        self.body_id = int(
            self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_BODY,
                self.config.body_name,
            )
        )
        if self.body_id < 0:
            raise ValueError(f"Body '{self.config.body_name}' not found in MuJoCo model")

        group_count = int(getattr(self.mujoco, "mjNGROUP", 6))
        for group in (self.config.terrain_group, self.config.robot_group):
            if not 0 <= int(group) < group_count:
                raise ValueError(f"geom group {group} is outside [0, {group_count - 1}]")
        if self.config.terrain_group == self.config.robot_group:
            raise ValueError("terrain_group and robot_group must be different")

        if prepare_geom_groups:
            self._separate_robot_and_terrain_groups()
        self._geomgroup = np.zeros(group_count, dtype=np.uint8)
        self._geomgroup[self.config.terrain_group] = 1
        self._grid_local_xy = self._build_local_grid()
        self.last_scan: dict[str, np.ndarray] | None = None

    def _separate_robot_and_terrain_groups(self) -> None:
        """Make rays see world geometry while ignoring all articulated bodies."""
        if not hasattr(self.model, "geom_bodyid") or not hasattr(self.model, "geom_group"):
            return
        body_ids = np.asarray(self.model.geom_bodyid)
        self.model.geom_group[body_ids == 0] = self.config.terrain_group
        self.model.geom_group[body_ids != 0] = self.config.robot_group

    def _build_local_grid(self) -> np.ndarray:
        cfg = self.config
        x = np.linspace(-cfg.size_x / 2.0, cfg.size_x / 2.0, cfg.num_x, dtype=np.float64)
        y = np.linspace(-cfg.size_y / 2.0, cfg.size_y / 2.0, cfg.num_y, dtype=np.float64)
        # Equivalent to torch.meshgrid(x, y, indexing="xy") in Isaac Lab:
        # shape (num_y, num_x), with x changing fastest after flattening.
        grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
        return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)

    @property
    def grid_local_xy(self) -> np.ndarray:
        return self._grid_local_xy.copy()

    @staticmethod
    def _yaw_rotation(xmat: np.ndarray) -> np.ndarray:
        matrix = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def scan(self) -> np.ndarray:
        """Return flattened ``[x, y, z]`` hits with shape ``(2079,)``."""
        cfg = self.config
        self.mujoco.mj_kinematics(self.model, self.data)

        body_pos_w = np.asarray(self.data.xpos[self.body_id], dtype=np.float64).copy()
        yaw_rotation = self._yaw_rotation(self.data.xmat[self.body_id])
        grid_world_xy = self._grid_local_xy @ yaw_rotation.T + body_pos_w[:2]

        ray_starts_w = np.empty((cfg.num_rays, 3), dtype=np.float64)
        ray_starts_w[:, :2] = grid_world_xy
        ray_starts_w[:, 2] = body_pos_w[2] + cfg.ray_start_height
        ray_direction_w = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        distances = np.full(cfg.num_rays, -1.0, dtype=np.float64)
        geom_ids = np.full(cfg.num_rays, -1, dtype=np.int32)
        for index, ray_start_w in enumerate(ray_starts_w):
            geom_id = np.array([-1], dtype=np.int32)
            distances[index] = self.mujoco.mj_ray(
                self.model,
                self.data,
                ray_start_w,
                ray_direction_w,
                self._geomgroup,
                1,
                self.body_id,
                geom_id,
                None,
            )
            geom_ids[index] = int(geom_id[0])

        valid = (distances >= 0.0) & np.isfinite(distances)
        hit_points_w = ray_starts_w.copy()
        hit_points_w[:, 2] = body_pos_w[2] + cfg.min_local_z
        hit_points_w[valid] = ray_starts_w[valid] + distances[valid, None] * ray_direction_w

        local_hits = np.empty((cfg.num_rays, 3), dtype=np.float64)
        local_hits[:, :2] = (hit_points_w[:, :2] - body_pos_w[:2]) @ yaw_rotation
        local_hits[:, 2] = hit_points_w[:, 2] - body_pos_w[2]
        raw_local_z = local_hits[:, 2].copy()
        # Training clips z only. There is no abs() and no 0.93 m compensation.
        local_hits[:, 2] = np.clip(local_hits[:, 2], cfg.min_local_z, cfg.max_local_z)
        observation = local_hits.reshape(-1).astype(np.float32)

        self.last_scan = {
            "body_pos_w": body_pos_w,
            "grid_local_xy": self.grid_local_xy,
            "ray_starts_w": ray_starts_w,
            "distances": distances,
            "geom_ids": geom_ids,
            "valid": valid,
            "hit_points_w": hit_points_w,
            "raw_local_z": raw_local_z,
            "local_hits": local_hits.astype(np.float32),
            "observation": observation.copy(),
        }
        return observation
