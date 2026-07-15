import unittest

import mujoco
import numpy as np

from ame_terrain import TERRAIN_NAMES, build_mujoco_model_with_terrain, generate_ame_terrain


XML_PATH = "assets/g1/mjcf/g1_29dof_ame.xml"


class AmeTerrainTest(unittest.TestCase):
    @staticmethod
    def _ray_height(model: mujoco.MjModel, x: float, y: float) -> float:
        data = mujoco.MjData(model)
        model.geom_group[np.asarray(model.geom_bodyid) != 0] = 1
        mujoco.mj_forward(model, data)
        geom_id = np.array([-1], dtype=np.int32)
        ray_z = 3.0
        distance = mujoco.mj_ray(
            model,
            data,
            np.array([x, y, ray_z]),
            np.array([0.0, 0.0, -1.0]),
            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
            1,
            -1,
            geom_id,
            None,
        )
        if geom_id[0] < 0:
            raise AssertionError(f"Ray at ({x}, {y}) missed terrain")
        return ray_z - distance

    def test_all_training_terrains_have_expected_grid_and_center_origin(self):
        for name in TERRAIN_NAMES:
            with self.subTest(name=name):
                terrain = generate_ame_terrain(name, difficulty=0.5, seed=42)
                self.assertEqual(terrain.heights.shape, (161, 161))
                self.assertEqual(terrain.heights.dtype, np.float32)
                self.assertAlmostEqual(terrain.size, 8.0)
                self.assertAlmostEqual(float(terrain.heights[80, 80]), 0.0)
                self.assertTrue(np.all(np.isfinite(terrain.heights)))

    def test_seeded_stakes_are_deterministic(self):
        first = generate_ame_terrain("stakes3", difficulty=0.7, seed=123).heights
        second = generate_ame_terrain("stakes3", difficulty=0.7, seed=123).heights
        np.testing.assert_array_equal(first, second)

    def test_finetune_height_ranges_are_preserved(self):
        stakes = generate_ame_terrain("stakes1", difficulty=0.5, seed=42)
        self.assertAlmostEqual(stakes.min_height, -2.0)
        self.assertLessEqual(stakes.max_height, 0.03 + 1.0e-6)
        stairs = generate_ame_terrain("pyramid_stairs_inv", difficulty=0.5, seed=42)
        self.assertAlmostEqual(stairs.max_height, 1.05, places=5)

    def test_stairs_use_exact_box_mesh_profiles(self):
        sample_x = [0.0, 1.25, 1.55, 1.85, 2.15, 2.45, 2.75, 3.5]
        for name, expected in (
            ("pyramid_stairs", [0.0, -0.15, -0.30, -0.45, -0.60, -0.75, -0.90, -1.05]),
            ("pyramid_stairs_inv", [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05]),
        ):
            with self.subTest(name=name):
                terrain = generate_ame_terrain(name, difficulty=0.5, seed=42)
                model = build_mujoco_model_with_terrain(XML_PATH, terrain)
                self.assertEqual(model.nhfield, 0)
                self.assertEqual(int(np.count_nonzero(model.geom_bodyid == 0)), 33)
                actual = [self._ray_height(model, x, 0.0) for x in sample_x]
                np.testing.assert_allclose(actual, expected, atol=1.0e-7)

    def test_rails_use_exact_box_mesh_profile(self):
        terrain = generate_ame_terrain("rails", difficulty=0.5, seed=42)
        model = build_mujoco_model_with_terrain(XML_PATH, terrain)
        self.assertEqual(model.nhfield, 0)
        self.assertEqual(int(np.count_nonzero(model.geom_bodyid == 0)), 13)
        actual = [self._ray_height(model, x, 0.0) for x in (0.0, 1.05, 1.2, 2.9, 3.2)]
        np.testing.assert_allclose(actual, [0.0, 0.15, 0.0, 0.15, 0.0], atol=1.0e-7)

    def test_custom_heightfield_uses_vertical_plateau_boxes(self):
        terrain = generate_ame_terrain("play_stakes", difficulty=0.5, seed=42)
        model = build_mujoco_model_with_terrain(XML_PATH, terrain)
        self.assertEqual(model.nhfield, 0)
        self.assertAlmostEqual(self._ray_height(model, 0.0, 0.0), 0.0, places=7)
        self.assertAlmostEqual(self._ray_height(model, 1.2, -0.1), 0.0, places=7)
        self.assertAlmostEqual(self._ray_height(model, 1.2, 0.1), -2.0, places=7)


if __name__ == "__main__":
    unittest.main()
