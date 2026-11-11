"""Scan80000 single-tile terrain and yaw-aligned, bilinearly sampled height map."""

import mujoco
import numpy as np


def peak_box_height(boxes):
    """Highest obstacle top above the ground slab for an upright BOX scene."""
    if len(boxes) <= 1:
        return 0.
    floor_z = boxes[0, 2] + boxes[0, 9] / 2
    return float(np.max(boxes[1:, 2] + boxes[1:, 9] / 2) - floor_z)


def vary_boxes(boxes, height=None, offset_y=0.0):
    """Set peak obstacle height above the floor and shift all obstacles along world +Y (metres)."""
    if not np.isfinite(offset_y) or (height is not None and (not np.isfinite(height) or height < 0)):
        raise ValueError("Box height must be finite and nonnegative, and box offset must be finite")
    if height is None and offset_y == 0:
        return boxes
    if height == 0:
        return boxes[:1].copy()
    if len(boxes) <= 1:
        raise ValueError("The scene needs an obstacle footprint to change box height or position")
    result = np.array(boxes, dtype=np.float64, copy=True)
    obstacles = result[1:]
    obstacles[:, 1] += offset_y
    if height is not None:
        if np.any(np.abs(result[:, 4:6]) > 1e-9):
            raise ValueError("Changing box height requires upright boxes")
        floor_z = result[0, 2] + result[0, 9] / 2
        peak = peak_box_height(boxes)
        if peak <= 0:
            raise ValueError("Expected an obstacle above the floor")
        scale = height / peak
        obstacles[:, 2] = floor_z + (obstacles[:, 2] - floor_z) * scale
        obstacles[:, 9] *= scale
    return result


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


class TerrainContacts:
    """Detect both feet on the obstacle and their subsequent landings beyond its far edge."""

    def __init__(self, rt, target_xy):
        self.feet = {rt.model.body(f"{side}_ankle_roll_link").id: i for i, side in enumerate(("left", "right"))}
        self.top_times = [None, None]
        self.landing_times = [None, None]
        boxes = rt.motion.boxes
        self.floor = rt.model.geom("terrain_box_0" if len(boxes) else "ground").id
        self.tops = {rt.model.geom(f"terrain_box_{i}").id: b[2] + b[9] / 2 for i, b in enumerate(boxes) if i}
        unit_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        self.corners = [b[:2] + (unit_corners * b[7:9] / 2) @ rt.data.geom_xmat[g].reshape(3, 3)[:2, :2].T
                        for b, g in zip(boxes[1:], self.tops)]
        direction = target_xy - rt.motion.pos[0, :2]
        self.direction = direction / max(np.linalg.norm(direction), 1e-12)
        self.far_edge = max((np.max(polygon @ self.direction) for polygon in self.corners), default=np.inf)
        self.force = np.empty(6)

    def measure_contacts(self, model, data):
        # After mj_step, contact forces correspond to the beginning of that physics step.
        now = max(0., data.time - model.opt.timestep)
        for index, contact in enumerate(data.contact):
            terrain, robot = map(int, contact.geom)
            if robot == self.floor or robot in self.tops:
                terrain, robot = robot, terrain
            foot = self.feet.get(model.geom_bodyid[robot])
            if foot is None or (terrain != self.floor and terrain not in self.tops) or abs(contact.frame[2]) < .8:
                continue
            mujoco.mj_contactForce(model, data, index, self.force)
            if self.force[0] <= 1.:
                continue
            if terrain in self.tops and abs(contact.pos[2] - self.tops[terrain]) < .025:
                if self.top_times[foot] is None:
                    self.top_times[foot] = now
            elif terrain == self.floor and None not in self.top_times and contact.pos[:2] @ self.direction > self.far_edge:
                if self.landing_times[foot] is None:
                    self.landing_times[foot] = now

    @property
    def events(self):
        landing = None if None in self.landing_times else max(self.landing_times)
        return [*self.top_times, landing]
