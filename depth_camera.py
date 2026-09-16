"""Reusable training-compatible MuJoCo depth-camera interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import mujoco
import numpy as np


# Local OpenGL camera axes expressed in the robotics/world camera convention:
# OpenGL +X=right, +Y=up, -Z=forward; world +X=forward, +Y=left, +Z=up.
WORLD_TO_OPENGL_CAMERA = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


@dataclass(frozen=True)
class DepthCameraConfig:
    name: str
    width: int
    height: int
    vertical_fov_degrees: float
    update_period: float
    minimum_valid_distance: float
    min_depth: float
    max_depth: float
    normalize: bool = True
    output_min: float = 0.0
    output_max: float = 1.0
    visible_geom_groups: tuple[int, ...] = (0, 2)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DepthCameraConfig":
        depth_range = values["depth_range"]
        output_range = values.get("output_range", (0.0, 1.0))
        if values.get("data_type", "distance_to_image_plane") != "distance_to_image_plane":
            raise ValueError("Only distance_to_image_plane matches the target training policy")
        return cls(
            name=str(values["name"]),
            width=int(values["width"]),
            height=int(values["height"]),
            vertical_fov_degrees=float(values["vertical_fov_degrees"]),
            update_period=float(values["update_period"]),
            minimum_valid_distance=float(values.get("min_distance", 0.0)),
            min_depth=float(depth_range[0]),
            max_depth=float(depth_range[1]),
            normalize=bool(values.get("normalize", True)),
            output_min=float(output_range[0]),
            output_max=float(output_range[1]),
            visible_geom_groups=tuple(int(group) for group in values.get("visible_geom_groups", (0, 2))),
        )

    @property
    def horizontal_fov_degrees(self) -> float:
        fovy = math.radians(self.vertical_fov_degrees)
        return math.degrees(2.0 * math.atan(math.tan(fovy / 2.0) * self.width / self.height))

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        focal = self.height / (2.0 * math.tan(math.radians(self.vertical_fov_degrees) / 2.0))
        return np.array(
            [[focal, 0.0, self.width / 2.0], [0.0, focal, self.height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class DepthFrame:
    timestamp: float
    depth_m: np.ndarray
    policy_image: np.ndarray


def colorize_policy_depth(policy_image: np.ndarray) -> np.ndarray:
    """Convert normalized policy depth to an RGB image (near=warm, far=dark)."""
    image = np.asarray(policy_image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Expected depth image with shape (H, W) or (1, H, W), got {image.shape}")
    near = 1.0 - np.clip(image, 0.0, 1.0)
    return np.uint8(
        np.stack(
            (
                np.clip(1.7 * near, 0.0, 1.0),
                np.clip(1.7 * near - 0.35, 0.0, 1.0),
                np.clip(1.5 - 2.2 * near, 0.0, 1.0),
            ),
            axis=-1,
        )
        * 255.0
    )


class MujocoDepthCamera:
    """Render, preprocess, and deproject one named MuJoCo camera.

    ``policy_image`` uses channel-first shape ``(1, H, W)``. Captures are held
    between camera update ticks, matching the 50 Hz training sensor.
    """

    def __init__(self, model: mujoco.MjModel, config: DepthCameraConfig) -> None:
        self.model = model
        self.config = config
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, config.name)
        if self.camera_id < 0:
            raise ValueError(f"Camera not found in model: {config.name}")
        model_fovy = float(model.cam_fovy[self.camera_id])
        if not math.isclose(model_fovy, config.vertical_fov_degrees, abs_tol=1e-8):
            raise ValueError(
                f"Camera fovy mismatch: model={model_fovy}, config={config.vertical_fov_degrees}"
            )
        self._renderer = mujoco.Renderer(model, height=config.height, width=config.width)
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = 0
        for group in config.visible_geom_groups:
            self._scene_option.geomgroup[group] = 1
        self._last_frame: DepthFrame | None = None
        self._renderer.enable_depth_rendering()

    @property
    def last_frame(self) -> DepthFrame | None:
        return self._last_frame

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "MujocoDepthCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def capture(self, data: mujoco.MjData, *, force: bool = False) -> DepthFrame:
        timestamp = float(data.time)
        if (
            not force
            and self._last_frame is not None
            and timestamp + 1e-12 < self._last_frame.timestamp + self.config.update_period
        ):
            return self._last_frame
        self._renderer.update_scene(data, camera=self.camera_id, scene_option=self._scene_option)
        raw_depth = np.asarray(self._renderer.render(), dtype=np.float32).copy()
        depth_m = self.preprocess_metric(raw_depth)
        policy_image = self.to_policy_image(depth_m)
        depth_m.setflags(write=False)
        policy_image.setflags(write=False)
        self._last_frame = DepthFrame(timestamp=timestamp, depth_m=depth_m, policy_image=policy_image)
        return self._last_frame

    def preprocess_metric(self, depth: np.ndarray) -> np.ndarray:
        result = np.asarray(depth, dtype=np.float32).copy()
        result[~np.isfinite(result)] = self.config.max_depth
        result[result < self.config.minimum_valid_distance] = self.config.max_depth
        return np.clip(result, self.config.min_depth, self.config.max_depth)

    def to_policy_image(self, depth_m: np.ndarray) -> np.ndarray:
        result = self.preprocess_metric(depth_m)
        if self.config.normalize:
            span = self.config.max_depth - self.config.min_depth
            result = (result - self.config.min_depth) / span
            result = result * (self.config.output_max - self.config.output_min) + self.config.output_min
        return result[np.newaxis, ...].astype(np.float32, copy=False)

    def capture_rgb(self, data: mujoco.MjData) -> np.ndarray:
        self._renderer.disable_depth_rendering()
        try:
            self._renderer.update_scene(data, camera=self.camera_id, scene_option=self._scene_option)
            return np.asarray(self._renderer.render(), dtype=np.uint8).copy()
        finally:
            self._renderer.enable_depth_rendering()

    def deproject_camera(self, depth_m: np.ndarray) -> np.ndarray:
        """Return points in Isaac's camera convention (+X forward, +Y left, +Z up)."""
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.shape != (self.config.height, self.config.width):
            raise ValueError(
                f"Expected depth shape {(self.config.height, self.config.width)}, got {depth.shape}"
            )
        intrinsic = self.config.intrinsic_matrix
        rows, columns = np.indices(depth.shape, dtype=np.float64)
        x = depth
        y = -(columns + 0.5 - intrinsic[0, 2]) * depth / intrinsic[0, 0]
        z = -(rows + 0.5 - intrinsic[1, 2]) * depth / intrinsic[1, 1]
        return np.stack((x, y, z), axis=-1)

    def camera_pose_world(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Return position and rotation of the +X-forward camera frame in world."""
        position = np.asarray(data.cam_xpos[self.camera_id], dtype=np.float64).copy()
        rotation_opengl = np.asarray(data.cam_xmat[self.camera_id], dtype=np.float64).reshape(3, 3)
        rotation_world_camera = rotation_opengl @ WORLD_TO_OPENGL_CAMERA.T
        return position, rotation_world_camera

    def deproject_world(self, data: mujoco.MjData, depth_m: np.ndarray) -> np.ndarray:
        points_camera = self.deproject_camera(depth_m)
        position, rotation = self.camera_pose_world(data)
        return points_camera @ rotation.T + position
