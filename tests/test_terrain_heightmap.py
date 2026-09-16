import unittest

import numpy as np

from terrain_heightmap import Mid360HeightMapConfig, MujocoMid360HeightMap


class _FakeMjtObj:
    mjOBJ_BODY = 1


class _FakeMujoco:
    mjNGROUP = 6
    mjtObj = _FakeMjtObj()

    def __init__(self, terrain_height):
        self.terrain_height = terrain_height

    @staticmethod
    def mj_name2id(model, object_type, name):
        del model, object_type
        return 1 if name == "base_link" else -1

    @staticmethod
    def mj_kinematics(model, data):
        del model, data

    def mj_ray(
        self,
        model,
        data,
        ray_start,
        ray_direction,
        geomgroup,
        static_only,
        body_exclude,
        geom_id,
        cutoff,
    ):
        del model, data, ray_direction, static_only, body_exclude, cutoff
        if geomgroup[0] != 1:
            return -1.0
        hit_z = self.terrain_height(float(ray_start[0]), float(ray_start[1]))
        if hit_z is None:
            return -1.0
        geom_id[0] = 0
        return float(ray_start[2] - hit_z)


class _FakeData:
    def __init__(self, yaw=0.0):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        self.xpos = np.zeros((2, 3), dtype=np.float64)
        self.xpos[1] = [10.0, 20.0, 0.8]
        self.xmat = np.zeros((2, 9), dtype=np.float64)
        self.xmat[1] = np.array(
            [cos_yaw, -sin_yaw, 0.0, sin_yaw, cos_yaw, 0.0, 0.0, 0.0, 1.0]
        )


class Mid360HeightMapTest(unittest.TestCase):
    def test_default_grid_matches_mid360_shape_and_order(self):
        scanner = MujocoMid360HeightMap(
            object(),
            _FakeData(),
            mujoco_module=_FakeMujoco(lambda x, y: 0.0),
        )

        grid = scanner.output_grid_xy
        self.assertEqual(grid.shape, (187, 2))
        np.testing.assert_allclose(grid[:11, 0], np.arange(0.1, 1.11, 0.1))
        np.testing.assert_allclose(grid[:11, 1], -0.8)
        np.testing.assert_allclose(grid[11:22, 0], np.arange(0.1, 1.11, 0.1))
        np.testing.assert_allclose(grid[11:22, 1], -0.7)

    def test_flat_ground_matches_mid360_height_semantics(self):
        scanner = MujocoMid360HeightMap(
            object(),
            _FakeData(),
            mujoco_module=_FakeMujoco(lambda x, y: 0.0),
        )

        heights = scanner.scan()
        self.assertEqual(heights.dtype, np.float32)
        np.testing.assert_allclose(heights, 0.13, atol=1.0e-6)
        self.assertTrue(np.all(scanner.last_scan["valid"]))

    def test_training_observation_is_body_height_minus_hit_height(self):
        scanner = MujocoMid360HeightMap(
            object(),
            _FakeData(),
            mujoco_module=_FakeMujoco(lambda x, y: 0.3),
        )

        heights = scanner.scan_training_observation()
        self.assertEqual(heights.dtype, np.float32)
        np.testing.assert_allclose(heights, 0.5, atol=1.0e-6)

    def test_training_observation_applies_policy_height_offset(self):
        scanner = MujocoMid360HeightMap(
            object(),
            _FakeData(),
            config=Mid360HeightMapConfig(training_height_offset=0.5),
            mujoco_module=_FakeMujoco(lambda x, y: 0.0),
        )

        heights = scanner.scan_training_observation()
        np.testing.assert_allclose(heights, 0.3, atol=1.0e-6)

    def test_training_observation_uses_isaac_xy_order_without_rounding(self):
        scanner = MujocoMid360HeightMap(
            object(),
            _FakeData(),
            mujoco_module=_FakeMujoco(lambda x, y: (x - 10.0) / 7.0),
        )

        heights = scanner.scan_training_observation()
        expected_first_row = 0.8 - np.arange(0.1, 1.11, 0.1) / 7.0
        np.testing.assert_allclose(heights[:11], expected_first_row, atol=1.0e-6)

    def test_training_observation_clips_missed_rays(self):
        scanner = MujocoMid360HeightMap(
            object(),
            _FakeData(),
            mujoco_module=_FakeMujoco(lambda x, y: None),
        )

        heights = scanner.scan_training_observation()
        np.testing.assert_array_equal(heights, np.full(187, -100.0, dtype=np.float32))

    def test_grid_follows_body_yaw_but_rays_stay_vertical(self):
        scanner = MujocoMid360HeightMap(
            object(),
            _FakeData(yaw=np.pi / 2.0),
            mujoco_module=_FakeMujoco(lambda x, y: 0.0),
        )

        scanner.scan()
        # First output is local (x=0.1, y=-0.8). At +90 deg yaw it becomes
        # world delta (+0.8, +0.1).
        first_ray_index = 0
        first_start = scanner.last_scan["ray_starts_w"][first_ray_index]
        np.testing.assert_allclose(first_start[:2], [10.8, 20.1], atol=1.0e-12)

    def test_missing_ray_uses_mid360_zero_z_before_compensation(self):
        scanner = MujocoMid360HeightMap(
            object(),
            _FakeData(),
            mujoco_module=_FakeMujoco(lambda x, y: None),
        )

        heights = scanner.scan()
        np.testing.assert_allclose(heights, 0.93, atol=1.0e-6)
        self.assertFalse(np.any(scanner.last_scan["valid"]))

    def test_config_has_expected_dimensions(self):
        config = Mid360HeightMapConfig()
        self.assertEqual(config.num_x, 11)
        self.assertEqual(config.num_y, 17)
        self.assertEqual(config.num_heights, 187)


if __name__ == "__main__":
    unittest.main()
