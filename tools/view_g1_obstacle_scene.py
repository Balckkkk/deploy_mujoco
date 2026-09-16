#!/usr/bin/env python3
"""Build and inspect the phase-3 G1 vault/climb MuJoCo scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import yaml

from depth_camera import DepthCameraConfig, MujocoDepthCamera, colorize_policy_depth
from g1_obstacle_scene import (
    ObstacleSceneConfig,
    build_obstacle_model,
    reset_obstacles,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config/g1_perceptive.yaml"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_scene(args):
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scene_cfg = ObstacleSceneConfig.from_dict(config["obstacle_scene"])
    model = build_obstacle_model(
        project_path(config["model_path"]),
        scene_cfg,
        args.mode,
        randomize=args.randomize,
        seed=args.seed,
    )
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, config["home_keyframe"])
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    poses = reset_obstacles(
        model,
        data,
        scene_cfg,
        args.mode,
        randomize=args.randomize,
        seed=args.seed,
    )
    return config, scene_cfg, model, data, poses


def scene_report(scene_cfg, model, data, poses, mode, randomize, seed):
    obstacles = {}
    for key, obstacle in scene_cfg.obstacles.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obstacle.geom_name)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obstacle.body_name)
        half_size = model.geom_size[geom_id].copy()
        obstacles[key] = {
            "active": poses[key].active,
            "position": data.xpos[body_id].tolist(),
            "quaternion_wxyz": data.xquat[body_id].tolist(),
            "full_size": (2.0 * half_size).tolist(),
            "nominal_top_height": float(obstacle.position[2] + obstacle.size[2] / 2.0),
            "nominal_bottom_height": float(obstacle.position[2] - obstacle.size[2] / 2.0),
            "mocap_id": int(model.body_mocapid[body_id]),
            "friction": model.geom_friction[geom_id].tolist(),
        }
    return {
        "status": "PASS",
        "mode": mode,
        "randomized": randomize,
        "seed": seed,
        "ground_friction": list(scene_cfg.ground_friction),
        "obstacles": obstacles,
    }


def render_overview(model: mujoco.MjModel, data: mujoco.MjData, mode: str) -> np.ndarray:
    width, height = 960, 540
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [4.0 if mode == "sequential" else 3.2, 0.0, 0.35]
    camera.distance = 12.5 if mode == "sequential" else 11.0
    camera.azimuth = 90.0
    camera.elevation = -16.0
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.geomgroup[2] = 1
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        return renderer.render().copy()


def save_outputs(output_dir: Path, config, model, data, report) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = render_overview(model, data, report["mode"])
    Image.fromarray(overview).save(output_dir / "scene_overview.png")
    camera_cfg = DepthCameraConfig.from_dict(config["depth_camera"])
    with MujocoDepthCamera(model, camera_cfg) as camera:
        frame = camera.capture(data, force=True)
        depth_image = Image.fromarray(colorize_policy_depth(frame.policy_image)).resize(
            (camera_cfg.width * 10, camera_cfg.height * 10), Image.Resampling.NEAREST
        )
        canvas = Image.new("RGB", (depth_image.width, depth_image.height + 28), "black")
        canvas.paste(depth_image, (0, 28))
        ImageDraw.Draw(canvas).text((8, 7), "Robot depth camera: near warm / 3m dark", fill="white")
        canvas.save(output_dir / "robot_depth.png")
        np.save(output_dir / "robot_depth_m.npy", frame.depth_m)
        report["depth"] = {
            "shape": list(frame.depth_m.shape),
            "minimum_m": float(frame.depth_m.min()),
            "maximum_m": float(frame.depth_m.max()),
            "center_m": float(frame.depth_m[camera_cfg.height // 2, camera_cfg.width // 2]),
        }
    (output_dir / "scene_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def launch_viewer(model: mujoco.MjModel, data: mujoco.MjData, mode: str) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[0] = 1
        viewer.opt.geomgroup[2] = 1
        viewer.cam.lookat[:] = [4.0 if mode == "sequential" else 3.2, 0.0, 0.35]
        viewer.cam.distance = 12.5 if mode == "sequential" else 11.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -16.0
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("vault", "climb", "both", "sequential"), default="both")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", help="Save images and exit instead of opening a viewer.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "obstacle_scene_debug")
    args = parser.parse_args()
    config, scene_cfg, model, data, poses = load_scene(args)
    report = scene_report(scene_cfg, model, data, poses, args.mode, args.randomize, args.seed)
    print(json.dumps(report, indent=2))
    print("Color key: vault=orange, climb=cyan")
    if args.headless:
        save_outputs(args.output_dir.resolve(), config, model, data, report)
        print(f"Saved scene outputs to: {args.output_dir.resolve()}")
    else:
        launch_viewer(model, data, args.mode)


if __name__ == "__main__":
    main()
