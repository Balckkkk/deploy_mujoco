"""MuJoCo ray-cast height map matching ``LeggedLabDeploy/mid360.py``.

The output contains 187 heights sampled from a robot-yaw-aligned rectangle:

* forward x: 0.1 .. 1.1 m, 11 points
* lateral y: -0.8 .. 0.8 m, 17 points
* resolution: 0.1 m

The flattened order also matches ``mid360.py`` after its Fortran-order
flatten: forward x changes fastest and lateral y changes slowest.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Mid360HeightMapConfig:
    """Geometry and output semantics of the real Mid360 height map."""

    body_name: str = "base_link"
    x_min: float = 0.1
    x_max: float = 1.1
    y_min: float = -0.8
    y_max: float = 0.8
    resolution: float = 0.1
    ray_start_height: float = 0.5
    z_compensation: float = 0.93
    miss_local_z: float = 0.0
    round_decimals: int = 4
    training_height_offset: float = 0.0
    training_height_scale: float = 1.0
    training_observation_clip: float = 100.0
    terrain_geom_groups: tuple[int, ...] = (0,)
    static_only: bool = True

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("height-map axis maximum must not be below its minimum")
        # A ray must start above the body reference height. This changes only
        # the ray origin; reported heights remain relative to the body.
        if self.ray_start_height <= 0.0:
            raise ValueError("ray_start_height must be positive")
        if self.training_observation_clip <= 0.0:
            raise ValueError("training_observation_clip must be positive")

    @staticmethod
    def _point_count(axis_min: float, axis_max: float, resolution: float) -> int:
        intervals = (axis_max - axis_min) / resolution
        rounded = round(intervals)
        if not np.isclose(intervals, rounded, atol=1.0e-8):
            raise ValueError("axis range must be an integer multiple of resolution")
        return int(rounded) + 1

    @property
    def num_x(self) -> int:
        return self._point_count(self.x_min, self.x_max, self.resolution)

    @property
    def num_y(self) -> int:
        return self._point_count(self.y_min, self.y_max, self.resolution)

    @property
    def num_heights(self) -> int:
        return self.num_x * self.num_y


class MujocoMid360HeightMap:
    """Acquire a Mid360-compatible height vector using vertical MuJoCo rays.

    ``scan()`` returns a ``float32`` vector with shape ``(187,)``. Heights use
    the same transformation as ``mid360.py``::

        height = abs((terrain_world_z - body_world_z) + 0.93)

    Only yaw is used to rotate the sampling rectangle. Roll and pitch do not
    tilt the scan plane or the downward ray direction.

    ``scan_training_observation()`` uses the R2 LeggedLab training semantics::

        height = (body_world_z - terrain_world_z - height_offset) * height_scale

    The training vector uses the same 187-point grid and Isaac Lab's default
    ``GridPatternCfg(ordering="xy")`` order: forward x changes fastest.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        config: Mid360HeightMapConfig | None = None,
        *,
        mujoco_module: Any | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or Mid360HeightMapConfig()
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
        self._geomgroup = np.zeros(group_count, dtype=np.uint8)
        for group in self.config.terrain_geom_groups:
            if not 0 <= int(group) < group_count:
                raise ValueError(f"terrain geom group {group} is outside [0, {group_count - 1}]")
            self._geomgroup[int(group)] = 1

        self._grid_local_ray_order = self._build_local_grid()
        if self._grid_local_ray_order.shape != (self.config.num_heights, 2):
            raise RuntimeError("unexpected Mid360 grid shape")

        self.last_scan: dict[str, np.ndarray] | None = None

    def _build_local_grid(self) -> np.ndarray:
        """Return the same pre-flatten point order used by ``mid360.py``."""
        cfg = self.config
        x_values = np.linspace(cfg.x_min, cfg.x_max, cfg.num_x, dtype=np.float64)
        y_values = np.linspace(cfg.y_min, cfg.y_max, cfg.num_y, dtype=np.float64)

        # Shape is (num_x, num_y). C-order flatten therefore makes y change
        # fastest, matching np.array(np.meshgrid(x, y)).T.reshape(-1, 2).
        grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="ij")
        return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)

    @staticmethod
    def _yaw_rotation(xmat: np.ndarray) -> np.ndarray:
        matrix = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        return np.array(
            [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
            dtype=np.float64,
        )

    def _to_mid360_output_order(self, values: np.ndarray) -> np.ndarray:
        """Apply the reshape(...).flatten(order='F') used by ``mid360.py``."""
        cfg = self.config
        return np.asarray(values).reshape(cfg.num_x, cfg.num_y).flatten(order="F")

    @property
    def output_grid_xy(self) -> np.ndarray:
        """Local XY coordinates aligned one-to-one with the 187 outputs."""
        cfg = self.config
        grid = self._grid_local_ray_order.reshape(cfg.num_x, cfg.num_y, 2)
        return grid.transpose(1, 0, 2).reshape(-1, 2).copy()

    def scan(self) -> np.ndarray:
        """Cast the 187 rays and return Mid360-compatible height observations."""
        cfg = self.config
        self.mujoco.mj_kinematics(self.model, self.data)

        body_pos_w = np.asarray(self.data.xpos[self.body_id], dtype=np.float64).copy()
        yaw_rotation = self._yaw_rotation(self.data.xmat[self.body_id])
        grid_world_xy = self._grid_local_ray_order @ yaw_rotation.T
        grid_world_xy += body_pos_w[:2]

        ray_starts_w = np.empty((cfg.num_heights, 3), dtype=np.float64)
        ray_starts_w[:, :2] = grid_world_xy
        ray_starts_w[:, 2] = body_pos_w[2] + cfg.ray_start_height
        ray_direction_w = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        distances = np.full(cfg.num_heights, -1.0, dtype=np.float64)
        geom_ids = np.full(cfg.num_heights, -1, dtype=np.int32)
        for index, ray_start_w in enumerate(ray_starts_w):
            geom_id = np.array([-1], dtype=np.int32)
            distances[index] = self.mujoco.mj_ray(
                self.model,
                self.data,
                ray_start_w,
                ray_direction_w,
                self._geomgroup,
                int(cfg.static_only),
                self.body_id,
                geom_id,
                None,
            )
            geom_ids[index] = int(geom_id[0])

        valid_ray_order = (distances >= 0.0) & np.isfinite(distances)
        local_z_ray_order = np.full(
            cfg.num_heights,
            cfg.miss_local_z,
            dtype=np.float64,
        )
        local_z_ray_order[valid_ray_order] = (
            ray_starts_w[valid_ray_order, 2]
            - distances[valid_ray_order]
            - body_pos_w[2]
        )
        hit_points_w_ray_order = ray_starts_w.copy()
        hit_points_w_ray_order[:, 2] = body_pos_w[2] + cfg.miss_local_z
        hit_points_w_ray_order[valid_ray_order] = (
            ray_starts_w[valid_ray_order]
            + distances[valid_ray_order, np.newaxis] * ray_direction_w
        )

        # mid360.py adds 0.93, takes abs(), rounds to four decimals, then
        # publishes its Fortran-flattened 11x17 grid.
        heights_ray_order = np.abs(local_z_ray_order + cfg.z_compensation)
        heights = self._to_mid360_output_order(heights_ray_order)
        heights = np.round(heights, cfg.round_decimals).astype(np.float32)

        # Match legged_lab/envs/base/base_env.py. In the Isaac Lab version used
        # by the training project, RayCasterData.pos_w is the tracked parent
        # body position; the (0.6, 0.0, 20.0) ray-pattern offset only moves ray
        # origins and does not enter this observation formula.
        training_heights_ray_order = np.full(
            cfg.num_heights,
            -cfg.training_observation_clip,
            dtype=np.float64,
        )
        training_heights_ray_order[valid_ray_order] = (
            body_pos_w[2]
            - hit_points_w_ray_order[valid_ray_order, 2]
            - cfg.training_height_offset
        ) * cfg.training_height_scale
        training_heights = self._to_mid360_output_order(training_heights_ray_order)
        training_heights = np.clip(
            training_heights,
            -cfg.training_observation_clip,
            cfg.training_observation_clip,
        ).astype(np.float32)

        self.last_scan = {
            "body_pos_w": body_pos_w,
            "ray_starts_w": ray_starts_w,
            "distances_ray_order": distances,
            "geom_ids_ray_order": geom_ids,
            "valid_ray_order": valid_ray_order,
            "hit_points_w_ray_order": hit_points_w_ray_order,
            "valid": self._to_mid360_output_order(valid_ray_order),
            "local_z": self._to_mid360_output_order(local_z_ray_order),
            "heights": heights.copy(),
            "training_heights": training_heights,
            "grid_local_xy": self.output_grid_xy,
        }
        return heights

    def scan_training_observation(self) -> np.ndarray:
        """Cast rays and return the observation exactly as defined in training."""
        self.scan()
        return self.last_scan["training_heights"].copy()
