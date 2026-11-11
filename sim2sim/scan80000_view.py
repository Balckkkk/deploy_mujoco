"""Height scan and reference-motion overlays for the SuperMimic passive viewer."""

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from scan80000_policy import GOAL_DIM, SCAN_DIM, quat_to_mat


# Fixed root-to-terrain distance in metres, shared by the markers and map.
DISTANCES = (0.4, 0.65, 0.85, 1.2)
PALETTE = np.array([(255, 220, 83), (99, 207, 142), (38, 148, 171), (63, 73, 146)])


class ReferenceGhost:
    """Forward kinematics of the recorded G1 pose in the same world frame as the replay."""

    def __init__(self, model, joint_qpos_indices):
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_qpos_indices = joint_qpos_indices
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = 0
        self.options.geomgroup[2] = 1  # Visual meshes in the training MJCF.
        self.options.sitegroup[:] = 0
        self.perturb = mujoco.MjvPerturb()

    def append(self, scene, motion, frame):
        self.append_pose(scene, motion.pos[frame], motion.quat[frame], motion.joints[frame])

    def append_pose(self, scene, position, quaternion, joints):
        self.data.qpos[:3] = position
        self.data.qpos[3:7] = quaternion
        self.data.qpos[self.joint_qpos_indices] = joints
        mujoco.mj_kinematics(self.model, self.data)

        first = scene.ngeom
        mujoco.mjv_addGeoms(self.model, self.data, self.options, self.perturb,
                            mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        for geom in scene.geoms[first:scene.ngeom]:
            geom.matid = -1
            geom.rgba[:] = (1.0, 0.32, 0.62, 0.42)
            geom.emission = 0.45
            geom.castshadow = 0


def draw_planner_goal(scene, position, yaw):
    """Add a floor arrow at the final target, pointing along the requested heading."""
    if scene.ngeom >= scene.maxgeom:
        return
    start = np.asarray(position, dtype=np.float64) + [0, 0, 0.035]
    end = start + 0.32 * np.array([np.cos(yaw), np.sin(yaw), 0])
    arrow = scene.geoms[scene.ngeom]
    kind = mujoco.mjtGeom.mjGEOM_ARROW
    mujoco.mjv_initGeom(arrow, kind, np.zeros(3), start, np.eye(3).ravel(), (0.15, 1, 0.4, 1))
    mujoco.mjv_connector(arrow, kind, 0.025, start, end)
    arrow.emission = 0.5
    scene.ngeom += 1


def scan_colors(values):
    return np.stack([np.interp(values, DISTANCES, channel) for channel in PALETTE.T], axis=-1).astype(np.uint8)


def heightmap_panel(values, colors):
    # Policy layout is [leftward y, forward x]. Screen top is forward, left is +y.
    pixels = colors.transpose(1, 0, 2)[::-1, ::-1]
    rows, cols = pixels.shape[:2]
    cell = 14
    width = max(184, cols * cell + 20)
    x0, y0 = (width - cols * cell) // 2, 45
    bottom = y0 + rows * cell
    panel = Image.new("RGB", (width, bottom + 52), (23, 30, 39))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 6), "Root-to-terrain distance (m)", fill="white")
    draw.text((10, 25), "+X forward (up)   +Y left", fill=(179, 198, 209))
    for r, c in np.ndindex(rows, cols):
        x, y = x0 + c * cell, y0 + r * cell
        draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=tuple(pixels[r, c]))
    cx, cy = x0 + (cols // 2) * cell, y0 + (rows // 2) * cell
    draw.rectangle((cx, cy, cx + cell - 2, cy + cell - 2), outline="white", width=2)
    bar = scan_colors(np.linspace(DISTANCES[0], DISTANCES[-1], cols * cell))[None, :, :]
    panel.paste(Image.fromarray(bar).resize((cols * cell, 6)), (x0, bottom + 7))
    draw.text((x0, bottom + 15), "<=0.4", fill="white")
    draw.text((width - x0 - 32, bottom + 15), ">=1.2", fill="white")
    draw.text((x0, bottom + 33), f"Range  {values.min():.2f} - {values.max():.2f} m", fill="white")
    return panel


def append_scan_markers(scene, scanner, root_pose, values, radius=0.017):
    colors = scan_colors(values)
    valid = values < scanner.max_distance
    colors[~valid] = (102, 103, 110)
    xy = scanner.world_xy(root_pose[:3], quat_to_mat(root_pose[3:7]))
    points = np.column_stack((xy, root_pose[2] - values))
    rgba = np.column_stack((colors / 255.0, np.ones(len(values)))).astype(np.float32)
    size = np.full(3, radius)
    orientation = np.eye(3).ravel()
    for point, color in zip(points[valid], rgba[valid]):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, point, orientation, color)
        geom.emission = 0.8
        geom.specular = 0.0
        scene.ngeom += 1
    return colors


def draw_heightmap(viewer, scanner, root_pose, observation):
    values = observation[GOAL_DIM:GOAL_DIM + SCAN_DIM]
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        colors = append_scan_markers(scene, scanner, root_pose, values)

    viewport = viewer.viewport
    if viewport is None or viewport.width < 32 or viewport.height < 32:
        return
    panel = heightmap_panel(values, colors.reshape(*scanner.grid_shape, 3))
    scale = min(1.5, .27 * viewport.width / panel.width, .60 * viewport.height / panel.height)
    w, h = max(1, round(panel.width * scale)), max(1, round(panel.height * scale))
    rect = mujoco.MjrRect(viewport.width - w - 8, 8, w, h)
    viewer.set_images((rect, np.asarray(panel.resize((w, h), Image.NEAREST))))
