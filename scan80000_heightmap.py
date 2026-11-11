"""187-point MuJoCo height scan used by the early scan80000_deploy.py replay.

Grid: 17 x points over -0.8..0.8 m and 11 y points over -0.5..0.5 m (17x11 = 187), centered on the torso and rotated
by torso yaw only (roll/pitch do not tilt the scan plane). Output order is ``meshgrid(x, y, indexing="xy")`` flattened,
i.e. x changes fastest. Each value is ``clamp(body_z - terrain_z, max=5.0)``.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Scan187Config:
    body_name: str = "torso_link"
    size_x: float = 1.6
    size_y: float = 1.0
    resolution: float = 0.1
    max_distance: float = 5.0
    ray_start_height: float = 20.0
    terrain_geom_groups: tuple[int, ...] = (0,)
    static_only: int = 1

    @property
    def num_x(self) -> int:
        return int(round(self.size_x / self.resolution)) + 1

    @property
    def num_y(self) -> int:
        return int(round(self.size_y / self.resolution)) + 1

    @property
    def num_heights(self) -> int:
        return self.num_x * self.num_y


class Scan187HeightMap:
    def __init__(self, model: Any, data: Any, config: Scan187Config | None = None, *, mujoco_module: Any | None = None):
        self.model, self.data = model, data
        self.config = config or Scan187Config()
        self.mujoco = mujoco_module or importlib.import_module("mujoco")
        cfg = self.config
        self.body_id = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, cfg.body_name))
        if self.body_id < 0:
            raise ValueError(f"Body '{cfg.body_name}' not found in MuJoCo model")
        self._geomgroup = np.zeros(int(getattr(self.mujoco, "mjNGROUP", 6)), dtype=np.uint8)
        for g in cfg.terrain_geom_groups:
            self._geomgroup[int(g)] = 1
        x = np.linspace(-cfg.size_x / 2, cfg.size_x / 2, cfg.num_x, dtype=np.float64)
        y = np.linspace(-cfg.size_y / 2, cfg.size_y / 2, cfg.num_y, dtype=np.float64)
        gx, gy = np.meshgrid(x, y, indexing="xy")  # (num_y, num_x), x changes fastest
        self._offsets = np.stack((gx.reshape(-1), gy.reshape(-1)), axis=1)  # (187, 2)

    @staticmethod
    def _yaw_rotation(xmat: np.ndarray) -> np.ndarray:
        m = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
        yaw = float(np.arctan2(m[1, 0], m[0, 0]))
        return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float64)

    def scan(self) -> np.ndarray:
        cfg = self.config
        self.mujoco.mj_kinematics(self.model, self.data)
        body_pos = np.asarray(self.data.xpos[self.body_id], dtype=np.float64).copy()
        yaw_rot = self._yaw_rotation(self.data.xmat[self.body_id])
        xy = self._offsets @ yaw_rot.T + body_pos[:2]  # (187, 2)
        starts = np.empty((cfg.num_heights, 3), dtype=np.float64)
        starts[:, :2], starts[:, 2] = xy, body_pos[2] + cfg.ray_start_height
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        dist = np.full(cfg.num_heights, -1.0, dtype=np.float64)
        for i, s in enumerate(starts):
            dist[i] = self.mujoco.mj_ray(self.model, self.data, s, direction, self._geomgroup,
                                         int(cfg.static_only), self.body_id, np.array([-1], dtype=np.int32), None)
        terrain_z = np.where(dist >= 0, body_pos[2] + cfg.ray_start_height - dist, 0.0)
        return np.clip(body_pos[2] - terrain_z, a_min=None, a_max=cfg.max_distance).astype(np.float32)
