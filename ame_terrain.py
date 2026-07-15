"""MuJoCo terrain generation matching AME finetune terrain parameters.

The training setup uses 8 x 8 m sub-terrains, a 5 cm horizontal grid and a
5 mm vertical quantization.  This module ports the CPU height-field algorithms
used by ``FINETUNE_ROUGH_TERRAINS_CFG``.  Training mesh terrains (pyramid
stairs and rails) are installed as the same box primitives instead of a
smoothed MuJoCo hfield.  Terrain heights are shifted so the central spawn
platform is at z=0, matching Isaac Lab's per-tile terrain origin behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


TERRAIN_NAMES = (
    "flat",
    "play_stakes",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "stakes1",
    "stakes2",
    "stakes3",
    "hf_gaps",
    "stonebridge",
    "rails",
)


@dataclass(frozen=True)
class AmeTerrain:
    name: str
    difficulty: float
    seed: int
    heights: np.ndarray
    horizontal_scale: float = 0.05

    @property
    def size(self) -> float:
        return (self.heights.shape[0] - 1) * self.horizontal_scale

    @property
    def min_height(self) -> float:
        return float(np.min(self.heights))

    @property
    def max_height(self) -> float:
        return float(np.max(self.heights))


def _paint_square(field: np.ndarray, cx: int, cy: int, side: int, value: float) -> None:
    lower = side // 2
    upper = side - lower
    x1, x2 = max(0, cx - lower), min(field.shape[0], cx + upper)
    y1, y2 = max(0, cy - lower), min(field.shape[1], cy + upper)
    if x1 < x2 and y1 < y2:
        field[x1:x2, y1:y2] = value


def _with_hf_border(
    generator,
    *,
    border_width: float,
    horizontal_scale: float,
    total_points: int,
) -> np.ndarray:
    # This reproduces Isaac Lab's height_field_to_mesh allocation exactly.
    border_pixels = int(border_width / horizontal_scale) + 1
    interior_points = total_points - 2 * border_pixels
    heights = np.zeros((total_points, total_points), dtype=np.float32)
    heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = generator(interior_points)
    return heights


def _pyramid_stairs(difficulty: float, *, inverted: bool, points: int, scale: float) -> np.ndarray:
    size = (points - 1) * scale
    border_width = 1.0
    platform_width = 3.0
    step_width = 0.3
    step_height = 0.05 + difficulty * (0.25 - 0.05)
    num_steps = int((size - 2.0 * border_width - platform_width) // (2.0 * step_width) + 1)

    coordinates = np.linspace(-size / 2.0, size / 2.0, points)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    edge_distance = np.maximum(np.abs(x), np.abs(y))
    terrain_half = size / 2.0 - border_width
    heights = np.zeros((points, points), dtype=np.float32)
    inside = edge_distance <= terrain_half + 1.0e-9
    ring_from_edge = np.floor((terrain_half - edge_distance) / step_width).astype(np.int32)
    level = np.clip(ring_from_edge + 1, 1, num_steps + 1)
    heights[inside] = level[inside] * step_height
    if inverted:
        heights[inside] *= -1.0

    # Isaac Lab spawns at the reported center origin. Shift that origin to z=0.
    center_height = float(heights[points // 2, points // 2])
    return heights - center_height


def _alternate_stakes_raw(
    points: int,
    difficulty: float,
    rng: np.random.Generator,
    *,
    stake_height_max: float,
    stake_side_range: tuple[float, float],
    stake_gap_range: tuple[float, float],
    column_gap_range: tuple[float, float],
    platform_width: float,
    horizontal_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    stake_side = stake_side_range[1] - difficulty * (stake_side_range[1] - stake_side_range[0])
    stake_gap = stake_gap_range[0] + difficulty * (stake_gap_range[1] - stake_gap_range[0])
    column_gap = column_gap_range[1] - difficulty * (column_gap_range[1] - column_gap_range[0])
    side_px = max(1, int(stake_side / horizontal_scale))
    gap_px = max(0, int(stake_gap / horizontal_scale))
    column_gap_px = max(0, int(column_gap / horizontal_scale))
    platform_px = max(1, int(platform_width / horizontal_scale))
    height_px = max(0, int(stake_height_max / vertical_scale))
    values = np.arange(-height_px, height_px + 1) if height_px else np.array([0])
    field = np.full((points, points), -2.0, dtype=np.float32)

    def place(start: int, along_x: bool) -> None:
        offset = column_gap_px // 2
        step = max(1, gap_px + side_px)
        while start < points:
            value = float(rng.choice(values)) * vertical_scale
            if along_x:
                _paint_square(field, start, points // 2 + offset, side_px, value)
            else:
                _paint_square(field, points // 2 + offset, start, side_px, value)
            offset = -offset
            start += step

    place(0, True)
    place(0, False)
    x1, x2 = (points - platform_px) // 2, (points + platform_px) // 2
    field[x1:x2, x1:x2] = 0.0
    return field


def _double_stakes_raw(
    points: int,
    difficulty: float,
    rng: np.random.Generator,
    *,
    horizontal_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    stake_side = 0.40 - difficulty * (0.40 - 0.20)
    stake_gap = 0.10 + difficulty * (0.30 - 0.10)
    column_gap = 0.10
    side_px = max(1, int(stake_side / horizontal_scale))
    gap_px = max(0, int(stake_gap / horizontal_scale))
    column_gap_px = max(0, int(column_gap / horizontal_scale))
    platform_px = int(2.0 / horizontal_scale)
    height_px = int(0.03 / vertical_scale)
    values = np.arange(-height_px, height_px + 1)
    field = np.full((points, points), -2.0, dtype=np.float32)
    center = points // 2
    center_offset = side_px + column_gap_px

    def place_pair(primary: int, along_x: bool) -> None:
        base_offset = max(center_offset // 2, side_px // 2)
        for sign in (-1, 1):
            value = float(rng.choice(values)) * vertical_scale
            if along_x:
                _paint_square(field, primary, center + sign * base_offset, side_px, value)
            else:
                _paint_square(field, center + sign * base_offset, primary, side_px, value)

    step = max(1, gap_px + side_px)
    for primary in range(0, points, step):
        place_pair(primary, True)
        place_pair(primary, False)
    x1, x2 = (points - platform_px) // 2, (points + platform_px) // 2
    field[x1:x2, x1:x2] = 0.0
    return field


def _concentric_gaps_raw(
    points: int,
    difficulty: float,
    rng: np.random.Generator,
    *,
    horizontal_scale: float,
) -> np.ndarray:
    gap_px = int((0.2 + difficulty * (0.6 - 0.2)) / horizontal_scale)
    ground_px = int(0.5 / horizontal_scale)
    platform_px = int(2.0 / horizontal_scale)
    field = np.zeros((points, points), dtype=np.float32)
    start, stop, gap = 0, points, True
    while stop - start > platform_px:
        if gap:
            field[start:stop, start:stop] = -2.0
            start += max(1, gap_px)
            stop -= max(1, gap_px)
        else:
            field[start:stop, start:stop] = rng.integers(-6, 7) * 0.005
            start += ground_px
            stop -= ground_px
        gap = not gap
    x1, x2 = (points - platform_px) // 2, (points + platform_px) // 2
    field[x1:x2, x1:x2] = 0.0
    return field


def _stonebridge_raw(
    points: int,
    difficulty: float,
    rng: np.random.Generator,
    *,
    horizontal_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    width = int((0.35 - difficulty * (0.35 - 0.25)) / horizontal_scale)
    length = int((1.0 - difficulty * (1.0 - 0.6)) / horizontal_scale)
    distance = int((0.3 + difficulty * (0.5 - 0.3)) / horizontal_scale)
    platform = int(2.0 / horizontal_scale)
    height_px = int(0.03 / vertical_scale)
    values = np.arange(-height_px - 1, height_px)
    field = np.full((points, points), -2.0, dtype=np.float32)

    start = distance
    while start < points:
        stop = min(points, start + width)
        cross_start = (points - length) // 2
        field[start:stop, cross_start : cross_start + length] = float(rng.choice(values)) * vertical_scale
        start = stop + distance
    start = distance
    while start < points:
        stop = min(points, start + width)
        cross_start = (points - length) // 2
        field[cross_start : cross_start + length, start:stop] = float(rng.choice(values)) * vertical_scale
        start = stop + distance

    x1, x2 = (points - platform) // 2, (points + platform) // 2
    field[x1:x2, x1:x2] = 0.0
    return field


def _rails(difficulty: float, *, points: int, scale: float) -> np.ndarray:
    size = (points - 1) * scale
    coords = np.linspace(-size / 2.0, size / 2.0, points)
    x, y = np.meshgrid(coords, coords, indexing="ij")
    radius = np.maximum(np.abs(x), np.abs(y))
    rail_height = 0.05 + difficulty * (0.25 - 0.05)
    inner_1, outer_1 = 1.0, 1.0 + 0.1
    inner_2 = (2.0 + (size - 2.0) * 0.6) / 2.0
    outer_2 = inner_2 + 0.3
    field = np.zeros((points, points), dtype=np.float32)
    field[(radius >= inner_1) & (radius <= outer_1)] = rail_height
    field[(radius >= inner_2) & (radius <= outer_2)] = rail_height
    return field


def generate_ame_terrain(name: str, difficulty: float = 0.5, seed: int = 42) -> AmeTerrain:
    """Generate one representative tile from the AME training distribution."""
    if name not in TERRAIN_NAMES:
        raise ValueError(f"Unknown AME terrain '{name}'. Choices: {', '.join(TERRAIN_NAMES)}")
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    scale, vertical_scale, points = 0.05, 0.005, 161
    rng = np.random.default_rng(seed)

    if name == "flat":
        heights = np.zeros((points, points), dtype=np.float32)
    elif name == "pyramid_stairs":
        heights = _pyramid_stairs(difficulty, inverted=False, points=points, scale=scale)
    elif name == "pyramid_stairs_inv":
        heights = _pyramid_stairs(difficulty, inverted=True, points=points, scale=scale)
    elif name == "rails":
        heights = _rails(difficulty, points=points, scale=scale)
    else:
        if name == "play_stakes":
            raw_generator = lambda n: _alternate_stakes_raw(
                n,
                difficulty,
                rng,
                stake_height_max=0.0,
                stake_side_range=(0.2, 0.2),
                stake_gap_range=(0.3, 0.3),
                column_gap_range=(0.3, 0.3),
                platform_width=2.0,
                horizontal_scale=scale,
                vertical_scale=vertical_scale,
            )
        elif name == "stakes1":
            raw_generator = lambda n: _double_stakes_raw(
                n, difficulty, rng, horizontal_scale=scale, vertical_scale=vertical_scale
            )
        elif name in ("stakes2", "stakes3"):
            raw_generator = lambda n: _alternate_stakes_raw(
                n,
                difficulty,
                rng,
                stake_height_max=0.03,
                stake_side_range=(0.20, 0.40),
                stake_gap_range=(0.05, 0.15) if name == "stakes2" else (0.05, 0.25),
                column_gap_range=(0.0, 0.2) if name == "stakes2" else (0.3, 0.2),
                platform_width=2.0,
                horizontal_scale=scale,
                vertical_scale=vertical_scale,
            )
        elif name == "hf_gaps":
            raw_generator = lambda n: _concentric_gaps_raw(
                n, difficulty, rng, horizontal_scale=scale
            )
        else:
            raw_generator = lambda n: _stonebridge_raw(
                n, difficulty, rng, horizontal_scale=scale, vertical_scale=vertical_scale
            )
        heights = _with_hf_border(
            raw_generator,
            border_width=0.25,
            horizontal_scale=scale,
            total_points=points,
        )

    # Match terrain-origin spawning: central 2 x 2 m patch becomes local z=0.
    center = points // 2
    center_height = float(heights[center, center])
    heights = np.asarray(heights - center_height, dtype=np.float32)
    return AmeTerrain(name=name, difficulty=difficulty, seed=int(seed), heights=heights)


def _add_terrain_box(
    spec: mujoco.MjSpec,
    name: str,
    dimensions: tuple[float, float, float],
    position: tuple[float, float, float],
) -> None:
    """Add one static box using Isaac Lab's full dimensions and center pose."""
    if min(dimensions) <= 0.0:
        raise ValueError(f"Terrain box '{name}' has non-positive dimensions: {dimensions}")
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5 * value for value in dimensions],
        pos=position,
        material="groundplane",
        friction=[1.0, 0.005, 0.0001],
        group=0,
    )


def _add_box_border(
    spec: mujoco.MjSpec,
    name: str,
    size: tuple[float, float],
    inner_size: tuple[float, float],
    height: float,
    position: tuple[float, float, float],
) -> None:
    """Port Isaac Lab ``make_border`` as four MuJoCo box geoms."""
    thickness_x = 0.5 * (size[0] - inner_size[0])
    thickness_y = 0.5 * (size[1] - inner_size[1])
    px, py, pz = position
    _add_terrain_box(
        spec,
        f"{name}_left",
        (thickness_x, inner_size[1], height),
        (px - 0.5 * inner_size[0] - 0.5 * thickness_x, py, pz),
    )
    _add_terrain_box(
        spec,
        f"{name}_right",
        (thickness_x, inner_size[1], height),
        (px + 0.5 * inner_size[0] + 0.5 * thickness_x, py, pz),
    )
    _add_terrain_box(
        spec,
        f"{name}_top",
        (size[0], thickness_y, height),
        (px, py + 0.5 * inner_size[1] + 0.5 * thickness_y, pz),
    )
    _add_terrain_box(
        spec,
        f"{name}_bottom",
        (size[0], thickness_y, height),
        (px, py - 0.5 * inner_size[1] - 0.5 * thickness_y, pz),
    )


def _add_pyramid_stairs_boxes(spec: mujoco.MjSpec, terrain: AmeTerrain, *, inverted: bool) -> None:
    """Port Isaac Lab pyramid stair mesh generators without hfield smoothing."""
    size = terrain.size
    border_width = 1.0
    platform_width = 3.0
    step_width = 0.3
    step_height = 0.05 + terrain.difficulty * (0.25 - 0.05)
    terrain_size = size - 2.0 * border_width
    num_steps = int((size - 2.0 * border_width - platform_width) // (2.0 * step_width) + 1)
    total_height = (num_steps + 1) * step_height
    origin_z = -total_height if inverted else total_height

    # Isaac meshes are generated around (size/2, size/2); subtracting their
    # returned origin centers the tile at world x=y=0 and its spawn at z=0.
    border_center_z = -0.5 * step_height - origin_z
    _add_box_border(
        spec,
        "ame_stairs_border",
        (size, size),
        (terrain_size, terrain_size),
        step_height,
        (0.0, 0.0, border_center_z),
    )

    for level in range(num_steps):
        ring_size = terrain_size - 2.0 * level * step_width
        offset = 0.5 * terrain_size - (level + 0.5) * step_width
        if inverted:
            box_height = total_height - (level + 1) * step_height
            box_z = -0.5 * total_height - 0.5 * (level + 1) * step_height - origin_z
        else:
            box_height = (level + 2) * step_height
            box_z = 0.5 * level * step_height - origin_z

        _add_terrain_box(
            spec,
            f"ame_stairs_{level}_top",
            (ring_size, step_width, box_height),
            (0.0, offset, box_z),
        )
        _add_terrain_box(
            spec,
            f"ame_stairs_{level}_bottom",
            (ring_size, step_width, box_height),
            (0.0, -offset, box_z),
        )
        side_length = ring_size - 2.0 * step_width
        _add_terrain_box(
            spec,
            f"ame_stairs_{level}_right",
            (step_width, side_length, box_height),
            (offset, 0.0, box_z),
        )
        _add_terrain_box(
            spec,
            f"ame_stairs_{level}_left",
            (step_width, side_length, box_height),
            (-offset, 0.0, box_z),
        )

    middle_size = terrain_size - 2.0 * num_steps * step_width
    if inverted:
        middle_height = step_height
        middle_z = -total_height - 0.5 * step_height - origin_z
    else:
        middle_height = (num_steps + 2) * step_height
        middle_z = 0.5 * num_steps * step_height - origin_z
    _add_terrain_box(
        spec,
        "ame_stairs_middle",
        (middle_size, middle_size, middle_height),
        (0.0, 0.0, middle_z),
    )


def _add_rails_boxes(spec: mujoco.MjSpec, terrain: AmeTerrain) -> None:
    """Port Isaac Lab ``rails_terrain`` as two box borders and a ground box."""
    size = terrain.size
    platform_width = 2.0
    rail_height = 0.05 + terrain.difficulty * (0.25 - 0.05)
    rail_1_thickness, rail_2_thickness = 0.1, 0.3
    rail_center = (0.0, 0.0, 0.5 * rail_height)

    inner_1 = (platform_width, platform_width)
    outer_1 = tuple(value + 2.0 * rail_1_thickness for value in inner_1)
    _add_box_border(spec, "ame_rail_inner", outer_1, inner_1, rail_height, rail_center)

    rail_2_ratio = 0.6
    inner_2_value = platform_width + (size - platform_width) * rail_2_ratio
    inner_2 = (inner_2_value, inner_2_value)
    outer_2 = tuple(value + 2.0 * rail_2_thickness for value in inner_2)
    _add_box_border(spec, "ame_rail_outer", outer_2, inner_2, rail_height, rail_center)
    _add_terrain_box(spec, "ame_rails_ground", (size, size, 1.0), (0.0, 0.0, -0.5))


def _add_heightfield(spec: mujoco.MjSpec, terrain: AmeTerrain) -> None:
    """Install custom AME height-field terrain types in MuJoCo."""
    h_min, h_max = terrain.min_height, terrain.max_height
    elevation_range = max(h_max - h_min, 1.0e-4)
    normalized = (terrain.heights - h_min) / elevation_range
    # MuJoCo rows run along world y and columns along world x; AME arrays are x,y.
    hfield_data = normalized.T.reshape(-1).astype(np.float32)
    half_size = terrain.size / 2.0
    spec.add_hfield(
        name="ame_terrain",
        size=[half_size, half_size, elevation_range, 0.1],
        nrow=terrain.heights.shape[1],
        ncol=terrain.heights.shape[0],
        userdata=hfield_data,
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname="ame_terrain",
        pos=[0.0, 0.0, h_min],
        material="groundplane",
        friction=[1.0, 0.005, 0.0001],
        group=0,
    )


def _iter_height_rectangles(mask: np.ndarray):
    """Yield greedy non-overlapping rectangles of equal-height grid vertices."""
    remaining = np.asarray(mask, dtype=bool).copy()
    rows, cols = remaining.shape
    for row in range(rows):
        for col in range(cols):
            if not remaining[row, col]:
                continue
            col_stop = col
            while col_stop < cols and remaining[row, col_stop]:
                col_stop += 1
            row_stop = row + 1
            while row_stop < rows and np.all(remaining[row_stop, col:col_stop]):
                row_stop += 1
            remaining[row:row_stop, col:col_stop] = False
            yield row, row_stop, col, col_stop


def _add_discrete_height_boxes(spec: mujoco.MjSpec, terrain: AmeTerrain) -> None:
    """Represent AME height-field plateaus with vertical box edges.

    Isaac Lab converts these grids to triangle meshes with ``slope_threshold``
    correction, which moves vertices at deep discontinuities to make the sides
    vertical.  A MuJoCo hfield cannot express that geometry.  A common bottom
    surface plus merged plateau boxes preserves the ray/contact top profile and
    the vertical hole/stake boundaries without thousands of per-cell geoms.
    """
    heights = terrain.heights
    base_height = terrain.min_height
    _add_terrain_box(
        spec,
        "ame_hf_bottom",
        (terrain.size, terrain.size, 1.0),
        (0.0, 0.0, base_height - 0.5),
    )

    scale = terrain.horizontal_scale
    coordinate_min = -0.5 * terrain.size
    rectangle_index = 0
    for height in np.unique(heights):
        height = float(height)
        if height <= base_height + 1.0e-7:
            continue
        vertex_mask = np.isclose(heights, height, atol=1.0e-7, rtol=0.0)
        # A horizontal source-mesh cell exists only when all four vertices
        # share the plateau height. Mixed cells belong to the corrected edge.
        mask = (
            vertex_mask[:-1, :-1]
            & vertex_mask[1:, :-1]
            & vertex_mask[:-1, 1:]
            & vertex_mask[1:, 1:]
        )
        for row, row_stop, col, col_stop in _iter_height_rectangles(mask):
            size_x = (row_stop - row) * scale
            size_y = (col_stop - col) * scale
            x0 = coordinate_min + row * scale
            x1 = coordinate_min + row_stop * scale
            y0 = coordinate_min + col * scale
            y1 = coordinate_min + col_stop * scale
            box_height = height - base_height
            _add_terrain_box(
                spec,
                f"ame_hf_plateau_{rectangle_index}",
                (size_x, size_y, box_height),
                (0.5 * (x0 + x1), 0.5 * (y0 + y1), base_height + 0.5 * box_height),
            )
            rectangle_index += 1


def build_mujoco_model_with_terrain(xml_path: str | Path, terrain: AmeTerrain) -> mujoco.MjModel:
    """Replace the MJCF floor with the matching AME collision/render terrain."""
    spec = mujoco.MjSpec.from_file(str(Path(xml_path).resolve()))
    # Some robot-only MJCF assets intentionally omit actuators.  Add unit-gear
    # motors so the deployment's explicit PD torques retain their semantics.
    if len(spec.actuators) == 0:
        for joint in spec.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            actuator = spec.add_actuator()
            actuator.set_to_motor()
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.name = joint.name
            actuator.target = joint.name
    if spec.material("groundplane") is None:
        spec.add_material(name="groundplane", rgba=[0.75, 0.75, 0.75, 1.0])
    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor":
            spec.delete(geom)

    if terrain.name == "pyramid_stairs":
        _add_pyramid_stairs_boxes(spec, terrain, inverted=False)
    elif terrain.name == "pyramid_stairs_inv":
        _add_pyramid_stairs_boxes(spec, terrain, inverted=True)
    elif terrain.name == "rails":
        _add_rails_boxes(spec, terrain)
    elif terrain.name != "flat":
        _add_discrete_height_boxes(spec, terrain)
    else:
        _add_heightfield(spec, terrain)

    # FINETUNE_ROUGH_TERRAINS_CFG surrounds its tiles with a 50 m border.
    # Express its world z in the tile-origin frame used by this deployment.
    if terrain.name == "pyramid_stairs":
        border_top = terrain.min_height
    elif terrain.name == "pyramid_stairs_inv":
        border_top = terrain.max_height
    else:
        border_top = 0.0
    border_width = 50.0
    outer_size = terrain.size + 2.0 * border_width
    _add_box_border(
        spec,
        "ame_world_border",
        (outer_size, outer_size),
        (terrain.size, terrain.size),
        1.0,
        (0.0, 0.0, border_top - 0.5),
    )
    return spec.compile()
