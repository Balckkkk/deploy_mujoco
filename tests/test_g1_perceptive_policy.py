import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch
import torch.nn.functional as functional
import yaml

from g1_perceptive_policy import (
    ACTION_DIM,
    DEPTH_DIM,
    G1PerceptivePolicy,
    PROPRIO_DIM,
    build_proprioception,
)
from g1_perceptive_sim2sim import G1Sim2SimRuntime, load_config, update_depth_overlay


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/g1_perceptive.yaml"


class G1PerceptivePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.config = load_config(CONFIG_PATH)
        cls.checkpoint_path = Path(cls.config["policy_checkpoint"])

    def test_checkpoint_network_matches_direct_tensor_evaluation(self):
        policy = G1PerceptivePolicy(self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        normalizer = checkpoint["policy_normalizer_state_dict"]
        generator = np.random.default_rng(123)
        depth = generator.uniform(0.0, 1.0, (1, 32, 48)).astype(np.float32)
        proprio = generator.normal(0.0, 0.3, PROPRIO_DIM).astype(np.float32)

        actual = policy.infer(depth, proprio)
        image = torch.from_numpy(depth).unsqueeze(0)
        prefix = "encoders._parallel_blocks.depth_image.conv.conv."
        for index in (0, 2, 4):
            image = functional.conv2d(
                image,
                state[f"{prefix}{index}.weight"],
                state[f"{prefix}{index}.bias"],
                stride=2,
                padding=1,
            )
            image = functional.elu(image)
        latent = image.mean(dim=(-2, -1))
        normalized = (torch.from_numpy(proprio).unsqueeze(0) - normalizer["emp_norm._mean"]) / (
            normalizer["emp_norm._std"] + 0.01
        )
        reference = torch.cat((normalized, latent), dim=-1)
        for index in (0, 2, 4, 6):
            reference = functional.linear(
                reference, state[f"actor.{index}.weight"], state[f"actor.{index}.bias"]
            )
            if index != 6:
                reference = functional.elu(reference)
        np.testing.assert_allclose(actual, reference.squeeze(0).numpy(), rtol=1e-6, atol=1e-6)

    def test_proprioception_term_order(self):
        terms = [
            np.full(3, 1.0),
            np.full(3, 2.0),
            np.full(3, 3.0),
            np.full(ACTION_DIM, 4.0),
            np.full(ACTION_DIM, 5.0),
            np.full(ACTION_DIM, 6.0),
        ]
        packed = build_proprioception(*terms)
        self.assertEqual(packed.shape, (PROPRIO_DIM,))
        np.testing.assert_array_equal(
            packed,
            np.concatenate(terms, dtype=np.float32),
        )

    def test_runtime_reset_and_single_policy_step(self):
        runtime = G1Sim2SimRuntime(
            self.config,
            mode="climb",
            checkpoint=None,
            randomize_obstacle=False,
            seed=42,
            device="cpu",
            command_override=None,
        )
        try:
            np.testing.assert_allclose(runtime.data.qpos[:3], [0.0, 0.0, 0.80252707], atol=1e-7)
            np.testing.assert_array_equal(runtime.command_at(0.0), [0.0, 0.0, 0.0])
            np.testing.assert_array_equal(runtime.command_at(3.0), [1.0, 0.0, 0.0])
            action, proprio = runtime.policy_step()
            self.assertEqual(runtime.last_depth.policy_image.shape, (1, 32, 48))
            self.assertEqual(proprio.shape, (PROPRIO_DIM,))
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
        finally:
            runtime.close()

    def test_config_matches_training_vault_height(self):
        vault = self.raw_config["obstacle_scene"]["obstacles"]["vault"]
        self.assertAlmostEqual(vault["position"][2], 0.054999999701976776)
        self.assertAlmostEqual(vault["position"][2] + vault["size"][2] / 2.0, 0.35500001162290573)
        reference = json.loads(
            (ROOT / self.raw_config["isaac_reference_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(self.raw_config["policy_joint_names"], reference["joint_names"])
        self.assertEqual(DEPTH_DIM + PROPRIO_DIM, self.raw_config["policy"]["observation_dim"])

    def test_depth_overlay_resizes_to_viewer_viewport(self):
        class FakeViewer:
            viewport = mujoco.MjrRect(0, 0, 1200, 800)
            image = None

            def set_images(self, viewport_image):
                self.image = viewport_image

        viewer = FakeViewer()
        runtime = SimpleNamespace(
            last_depth=SimpleNamespace(
                policy_image=np.linspace(0.0, 1.0, 32 * 48, dtype=np.float32).reshape(1, 32, 48)
            )
        )
        update_depth_overlay(viewer, runtime)
        viewport, image = viewer.image
        self.assertEqual(image.shape, (viewport.height, viewport.width, 3))
        self.assertEqual((viewport.width, viewport.height), (400, 267))


if __name__ == "__main__":
    unittest.main()
