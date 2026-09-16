#!/usr/bin/env python3
"""Numerically validate and visualize the phase-2 G1 depth camera."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# EGL makes the default check work without an X server. It can still feed a Tk
# window for --visualize; users may override MUJOCO_GL explicitly.
os.environ.setdefault("MUJOCO_GL", "egl")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import yaml

from depth_camera import DepthCameraConfig, MujocoDepthCamera


DEFAULT_CONFIG = PROJECT_ROOT / "config/g1_perceptive.yaml"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_calibration_model(model_path: Path) -> mujoco.MjModel:
    """Add simple known geometry; this is not the phase-3 obstacle scene."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    spec.worldbody.add_geom(
        name="depth_test_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5.0, 5.0, 0.1],
        rgba=[0.18, 0.18, 0.18, 1.0],
        group=0,
    )
    spec.worldbody.add_geom(
        name="depth_test_center_panel",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[1.5, 0.0, 0.75],
        size=[0.05, 0.6, 0.75],
        rgba=[0.9, 0.2, 0.08, 1.0],
        group=0,
    )
    spec.worldbody.add_geom(
        name="depth_test_side_panel",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[2.2, -1.0, 0.55],
        size=[0.05, 0.35, 0.55],
        rgba=[0.1, 0.45, 0.95, 1.0],
        group=0,
    )
    return spec.compile()


def colorize_depth(policy_image: np.ndarray) -> np.ndarray:
    """Near=warm, far=dark visualization; policy data itself is unchanged."""
    value = 1.0 - np.clip(policy_image, 0.0, 1.0)
    red = np.clip(1.7 * value, 0.0, 1.0)
    green = np.clip(1.7 * value - 0.35, 0.0, 1.0)
    blue = np.clip(1.5 - 2.2 * value, 0.0, 1.0)
    return np.uint8(np.stack((red, green, blue), axis=-1) * 255.0)


def make_comparison(rgb: np.ndarray, depth_rgb: np.ndarray, scale: int = 8) -> Image.Image:
    rgb_image = Image.fromarray(rgb).resize(
        (rgb.shape[1] * scale, rgb.shape[0] * scale), Image.Resampling.NEAREST
    )
    depth_image = Image.fromarray(depth_rgb).resize(
        (depth_rgb.shape[1] * scale, depth_rgb.shape[0] * scale), Image.Resampling.NEAREST
    )
    canvas = Image.new("RGB", (rgb_image.width * 2, rgb_image.height + 28), "black")
    canvas.paste(rgb_image, (0, 28))
    canvas.paste(depth_image, (rgb_image.width, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "RGB (visible groups 0,2)", fill="white")
    draw.text((rgb_image.width + 8, 7), "Depth: near warm / 3m dark", fill="white")
    return canvas


def validate(config: dict, camera_cfg: DepthCameraConfig, model: mujoco.MjModel, data: mujoco.MjData):
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, config["home_keyframe"])
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    with MujocoDepthCamera(model, camera_cfg) as camera:
        frame = camera.capture(data, force=True)
        rgb = camera.capture_rgb(data)
        points_world = camera.deproject_world(data, frame.depth_m)
        initial_frame = frame
        data.time = camera_cfg.update_period * 0.25
        held_frame = camera.capture(data)
        data.time = camera_cfg.update_period
        refreshed_frame = camera.capture(data)

        expected_shape = (camera_cfg.height, camera_cfg.width)
        assert frame.depth_m.shape == expected_shape
        assert frame.policy_image.shape == (1, *expected_shape)
        assert rgb.shape == (*expected_shape, 3)
        assert np.all(np.isfinite(frame.depth_m))
        assert camera_cfg.min_depth <= float(frame.depth_m.min())
        assert float(frame.depth_m.max()) <= camera_cfg.max_depth
        assert 0.0 <= float(frame.policy_image.min())
        assert float(frame.policy_image.max()) <= 1.0
        np.testing.assert_allclose(frame.policy_image[0], frame.depth_m / 3.0, atol=2e-7)
        assert held_frame is initial_frame
        assert refreshed_frame is not initial_frame

        center = (camera_cfg.height // 2, camera_cfg.width // 2)
        center_depth = float(frame.depth_m[center])
        assert 1.5 < center_depth < 2.0, f"Center panel depth is wrong: {center_depth}"
        # The center panel's front face is x=1.45. Raster/intrinsic convention
        # differs slightly at even-sized center pixels, so use a 3 cm bound.
        center_world = points_world[center]
        assert abs(float(center_world[0]) - 1.45) < 0.03

        position, rotation = camera.camera_pose_world(data)
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["base_body"])
        expected_position = data.xpos[torso_id] + data.xmat[torso_id].reshape(3, 3) @ np.asarray(
            config["depth_camera"]["position"]
        )
        np.testing.assert_allclose(position, expected_position, atol=1e-9)
        expected_forward = np.array(
            [
                np.cos(np.deg2rad(config["depth_camera"]["pitch_degrees"])),
                0.0,
                -np.sin(np.deg2rad(config["depth_camera"]["pitch_degrees"])),
            ]
        )
        np.testing.assert_allclose(rotation[:, 0], expected_forward, atol=1e-8)

        report = {
            "status": "PASS",
            "shape_depth": list(frame.depth_m.shape),
            "shape_policy": list(frame.policy_image.shape),
            "vertical_fov_degrees": camera_cfg.vertical_fov_degrees,
            "horizontal_fov_degrees": camera_cfg.horizontal_fov_degrees,
            "update_period_seconds": camera_cfg.update_period,
            "depth_min_m": float(frame.depth_m.min()),
            "depth_max_m": float(frame.depth_m.max()),
            "center_depth_m": center_depth,
            "center_point_world": center_world.tolist(),
            "camera_position_world": position.tolist(),
            "camera_forward_world": rotation[:, 0].tolist(),
            "intrinsic_matrix": camera_cfg.intrinsic_matrix.tolist(),
        }
        return report, frame, rgb


def save_outputs(output_dir: Path, report: dict, frame, rgb: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_rgb = colorize_depth(frame.policy_image[0])
    Image.fromarray(rgb).save(output_dir / "rgb.png")
    Image.fromarray(depth_rgb).save(output_dir / "depth.png")
    comparison = make_comparison(rgb, depth_rgb)
    comparison.save(output_dir / "comparison.png")
    np.save(output_dir / "depth_m.npy", frame.depth_m)
    np.save(output_dir / "policy_image.npy", frame.policy_image)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_dir / "comparison.png"


def show_image(image_path: Path) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.title("G1 perceptive depth-camera phase-2 check")
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(root, image=photo)
    label.image = photo
    label.pack()
    tk.Label(root, text="关闭窗口结束；右侧为送入策略前的 32×48 深度图").pack(pady=5)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "depth_camera_debug")
    parser.add_argument("--visualize", action="store_true", help="Open an RGB/depth comparison window.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    camera_cfg = DepthCameraConfig.from_dict(config["depth_camera"])
    model = build_calibration_model(project_path(config["model_path"]))
    data = mujoco.MjData(model)
    report, frame, rgb = validate(config, camera_cfg, model, data)
    image_path = save_outputs(args.output_dir.resolve(), report, frame, rgb)
    print(json.dumps(report, indent=2))
    print(f"Saved debug outputs to: {args.output_dir.resolve()}")
    if args.visualize:
        show_image(image_path)


if __name__ == "__main__":
    main()
