import os

os.environ.setdefault("MUJOCO_GL", "egl")

import unittest
from pathlib import Path

import mujoco
import numpy as np
import yaml

from depth_camera import DepthCameraConfig, MujocoDepthCamera
from g1_obstacle_scene import (
    ObstacleSceneConfig,
    build_obstacle_model,
    reset_obstacles,
    sample_obstacle_poses,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/g1_perceptive.yaml"


class G1ObstacleSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.scene_cfg = ObstacleSceneConfig.from_dict(cls.config["obstacle_scene"])
        cls.camera_cfg = DepthCameraConfig.from_dict(cls.config["depth_camera"])
        cls.model_path = ROOT / cls.config["model_path"]

    def test_nominal_modes_and_geometry(self):
        expected_active = {
            "vault": {"vault"},
            "climb": {"climb"},
            "both": {"vault", "climb"},
            "sequential": {"vault", "climb"},
        }
        for mode, active_keys in expected_active.items():
            with self.subTest(mode=mode):
                model = build_obstacle_model(self.model_path, self.scene_cfg, mode)
                data = mujoco.MjData(model)
                poses = reset_obstacles(model, data, self.scene_cfg, mode)
                self.assertEqual(model.nmocap, 2)
                ground_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, self.scene_cfg.ground_name
                )
                self.assertEqual(model.geom_type[ground_id], mujoco.mjtGeom.mjGEOM_PLANE)
                np.testing.assert_allclose(model.geom_size[ground_id], self.scene_cfg.ground_size)
                np.testing.assert_allclose(model.geom_friction[ground_id], self.scene_cfg.ground_friction)
                self.assertEqual(model.nlight, len(self.scene_cfg.lights))
                np.testing.assert_allclose(
                    model.vis.headlight.ambient, self.scene_cfg.headlight_ambient
                )
                np.testing.assert_allclose(
                    model.vis.headlight.diffuse, self.scene_cfg.headlight_diffuse
                )
                for key, obstacle in self.scene_cfg.obstacles.items():
                    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obstacle.geom_name)
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obstacle.body_name)
                    self.assertEqual(model.geom_type[geom_id], mujoco.mjtGeom.mjGEOM_BOX)
                    np.testing.assert_allclose(2.0 * model.geom_size[geom_id], obstacle.size)
                    np.testing.assert_allclose(model.geom_friction[geom_id], obstacle.friction)
                    self.assertNotEqual(model.geom_contype[geom_id], 0)
                    self.assertNotEqual(model.geom_conaffinity[geom_id], 0)
                    self.assertEqual(poses[key].active, key in active_keys)
                    self.assertGreaterEqual(model.body_mocapid[body_id], 0)
                if mode == "sequential":
                    np.testing.assert_allclose(
                        poses["climb"].position,
                        self.scene_cfg.obstacles["climb"].sequential_position,
                    )

    def test_randomization_is_deterministic_and_bounded(self):
        first = sample_obstacle_poses(self.scene_cfg, "both", randomize=True, seed=123)
        second = sample_obstacle_poses(self.scene_cfg, "both", randomize=True, seed=123)
        for key, obstacle in self.scene_cfg.obstacles.items():
            np.testing.assert_allclose(first[key].position, second[key].position)
            np.testing.assert_allclose(first[key].quaternion_wxyz, second[key].quaternion_wxyz)
            delta = first[key].position - np.asarray(obstacle.position)
            for index, axis in enumerate(("x", "y", "z")):
                lower, upper = obstacle.randomization[axis]
                self.assertGreaterEqual(delta[index], lower)
                self.assertLessEqual(delta[index], upper)
            yaw = 2.0 * np.arctan2(first[key].quaternion_wxyz[3], first[key].quaternion_wxyz[0])
            lower, upper = obstacle.randomization["yaw"]
            self.assertGreaterEqual(yaw, lower)
            self.assertLessEqual(yaw, upper)

    def test_active_obstacles_are_visible_to_depth_camera(self):
        cases = (("vault", 0.0), ("climb", 3.5))
        for mode, robot_x in cases:
            with self.subTest(mode=mode):
                model = build_obstacle_model(self.model_path, self.scene_cfg, mode)
                data = mujoco.MjData(model)
                home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
                mujoco.mj_resetDataKeyframe(model, data, home_id)
                data.qpos[0] = robot_x
                reset_obstacles(model, data, self.scene_cfg, mode)
                with MujocoDepthCamera(model, self.camera_cfg) as camera:
                    frame = camera.capture(data, force=True)
                center_depth = frame.depth_m[
                    self.camera_cfg.height // 2, self.camera_cfg.width // 2
                ]
                self.assertLess(center_depth, self.camera_cfg.max_depth)
                self.assertGreater(np.mean(frame.depth_m < self.camera_cfg.max_depth), 0.5)


if __name__ == "__main__":
    unittest.main()
