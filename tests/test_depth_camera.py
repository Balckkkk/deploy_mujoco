import os

os.environ.setdefault("MUJOCO_GL", "egl")

import unittest
from pathlib import Path

import mujoco
import numpy as np
import yaml

from depth_camera import DepthCameraConfig, MujocoDepthCamera, colorize_policy_depth
from tools.test_g1_depth_camera import build_calibration_model, project_path, validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/g1_perceptive.yaml"


class DepthCameraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.camera_cfg = DepthCameraConfig.from_dict(cls.config["depth_camera"])

    def test_training_intrinsics(self):
        self.assertEqual((self.camera_cfg.height, self.camera_cfg.width), (32, 48))
        self.assertAlmostEqual(self.camera_cfg.vertical_fov_degrees, 70.0)
        self.assertAlmostEqual(self.camera_cfg.horizontal_fov_degrees, 92.81132645176451)
        np.testing.assert_allclose(
            self.camera_cfg.intrinsic_matrix,
            [[22.85036811, 0.0, 24.0], [0.0, 22.85036811, 16.0], [0.0, 0.0, 1.0]],
            atol=1e-8,
        )

    def test_preprocessing_semantics(self):
        model = mujoco.MjModel.from_xml_path(str(project_path(self.config["model_path"])))
        with MujocoDepthCamera(model, self.camera_cfg) as camera:
            values = np.array([[np.nan, np.inf, 0.01, 0.05, 1.5, 5.0]], dtype=np.float32)
            metric = camera.preprocess_metric(values)
            np.testing.assert_allclose(metric, [[3.0, 3.0, 3.0, 0.05, 1.5, 3.0]])
            policy = camera.to_policy_image(metric)
            np.testing.assert_allclose(policy, metric[np.newaxis] / 3.0, atol=1e-7)

    def test_policy_depth_colorization(self):
        image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        rgb = colorize_policy_depth(image)
        self.assertEqual(rgb.shape, (1, 3, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreater(int(rgb[0, 0, 0]), int(rgb[0, 2, 0]))

    def test_render_geometry_pose_and_update_rate(self):
        model = build_calibration_model(project_path(self.config["model_path"]))
        data = mujoco.MjData(model)
        report, frame, _ = validate(self.config, self.camera_cfg, model, data)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(frame.policy_image.shape, (1, 32, 48))


if __name__ == "__main__":
    unittest.main()
