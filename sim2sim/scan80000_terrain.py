"""Scan80000 single-tile terrain and yaw-aligned, bilinearly sampled height map."""

import mujoco
import numpy as np


def add_terrain(spec, boxes):
    # Reserve geom group 0 for terrain rays. Robot visuals and collisions use 2 and 3.
    terrain = spec.worldbody.add_body(name="terrain")
    if len(boxes) == 0:
        terrain.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                         size=(0, 0, 0.05), group=0, rgba=(0.65, 0.67, 0.69, 1))
    for i, box in enumerate(boxes):
        color = (0.65, 0.67, 0.69, 1) if i == 0 else (0.69, 0.51, 0.31, 1)
        terrain.add_geom(name=f"terrain_box_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                         pos=box[:3], quat=box[3:7], size=box[7:] / 2, group=0, rgba=color)


class TrainingHeightMap:
    """Rasterize the static MuJoCo geometry once on the training world's lattice."""

    def __init__(self, model, data, sensor, resolution):
        self.resolution = resolution
        self.max_distance = sensor["max_distance"]
        pattern = sensor["pattern"]
        dx = pattern["resolution"]
        axes = [np.arange(-size / 2, size / 2 + dx / 2, dx) for size in pattern["size"]]
        self.grid_shape = (len(axes[1]), len(axes[0]))
        self.offsets = np.stack(np.meshgrid(*axes, indexing="xy"), axis=-1).reshape(-1, 2)
        terrain = model.geom_group == 0
        if np.any(model.geom_type[terrain] == mujoco.mjtGeom.mjGEOM_PLANE):
            # Flat-only motions have a single level plane.
            self.origin = np.zeros(2)
            self.heights = np.zeros((2, 2))
            return

        centers = data.geom_xpos[terrain]
        rotations = data.geom_xmat[terrain].reshape(-1, 3, 3)
        extents = (np.abs(rotations) @ model.geom_size[terrain, :, None])[..., 0]
        lo = np.floor((centers - extents).min(axis=0)[:2] / resolution).astype(int) - 2
        hi = np.ceil((centers + extents).max(axis=0)[:2] / resolution).astype(int) + 2
        self.origin = lo * resolution
        nx, ny = hi - lo + 1
        self.heights = np.full((ny, nx), -100.0)  # InstinctMJ's MISS_HEIGHT
        groups = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
        groups[0] = 1
        z = (centers + extents)[:, 2].max() + 1.0
        origin = np.array([0.0, 0.0, z])
        direction = np.array([0.0, 0.0, -1.0])
        hit = np.empty(1, dtype=np.int32)
        for j in range(ny):
            origin[1] = (lo[1] + j) * resolution
            for i in range(nx):
                origin[0] = (lo[0] + i) * resolution
                d = mujoco.mj_ray(model, data, origin, direction, groups, 1, -1, hit)
                if d >= 0:
                    self.heights[j, i] = z - d

    def world_xy(self, pos, rotation):
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        return pos[:2] + self.offsets @ np.array([[c, s], [-s, c]])

    def scan(self, pos, rotation):
        xy = self.world_xy(pos, rotation)
        ny, nx = self.heights.shape
        grid = np.clip((xy - self.origin) / self.resolution, 0, [nx - 1, ny - 1])
        ij = np.minimum(grid.astype(int), [nx - 2, ny - 2])
        i, j = ij.T
        fx, fy = (grid - ij).T
        h = self.heights
        low = (1 - fx) * h[j, i] + fx * h[j, i + 1]
        high = (1 - fx) * h[j + 1, i] + fx * h[j + 1, i + 1]
        return np.minimum(pos[2] - ((1 - fy) * low + fy * high), self.max_distance).astype(np.float32)
