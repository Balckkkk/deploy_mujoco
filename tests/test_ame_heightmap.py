import unittest

import numpy as np

from ame_heightmap import AmeHeightMapConfig, MujocoAmeHeightMap


class _FakeMjtObj:
    mjOBJ_BODY = 1


class _FakeModel:
    def __init__(self):
        self.geom_bodyid = np.array([0, 1, 2], dtype=np.int32)
        self.geom_group = np.zeros(3, dtype=np.int32)


class _FakeData:
    def __init__(self, yaw=0.0):
        c, s = np.cos(yaw), np.sin(yaw)
        self.xpos = np.zeros((2, 3), dtype=np.float64)
        self.xpos[1] = [10.0, 20.0, 0.84]
        self.xmat = np.zeros((2, 9), dtype=np.float64)
        self.xmat[1] = [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]


class _FakeMujoco:
    mjNGROUP = 6
    mjtObj = _FakeMjtObj()

    def __init__(self, terrain_height):
        self.terrain_height = terrain_height

    @staticmethod
    def mj_name2id(model, object_type, name):
        del model, object_type
        return 1 if name == "torso_link" else -1

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
        if geomgroup[0] != 1 or np.any(geomgroup[1:]):
            return -1.0
        hit_z = self.terrain_height(float(ray_start[0]), float(ray_start[1]))
        if hit_z is None:
            return -1.0
        geom_id[0] = 0
        return float(ray_start[2] - hit_z)


class AmeHeightMapTest(unittest.TestCase):
    def test_grid_shape_and_isaac_xy_order(self):
        scanner = MujocoAmeHeightMap(
            _FakeModel(),
            _FakeData(),
            mujoco_module=_FakeMujoco(lambda x, y: 0.0),
        )
        grid = scanner.grid_local_xy
        self.assertEqual(grid.shape, (693, 2))
        np.testing.assert_allclose(grid[:33, 0], np.arange(-0.8, 0.801, 0.05), atol=1.0e-12)
        np.testing.assert_allclose(grid[:33, 1], -0.5, atol=1.0e-12)
        np.testing.assert_allclose(grid[33:66, 1], -0.45, atol=1.0e-12)

    def test_flat_ground_returns_local_xyz_without_compensation(self):
        scanner = MujocoAmeHeightMap(
            _FakeModel(),
            _FakeData(),
            mujoco_module=_FakeMujoco(lambda x, y: 0.0),
        )
        observation = scanner.scan()
        hits = observation.reshape(693, 3)
        self.assertEqual(observation.shape, (2079,))
        np.testing.assert_allclose(hits[:, :2], scanner.grid_local_xy, atol=1.0e-6)
        np.testing.assert_allclose(hits[:, 2], -0.84, atol=1.0e-6)

    def test_grid_follows_torso_yaw(self):
        scanner = MujocoAmeHeightMap(
            _FakeModel(),
            _FakeData(yaw=np.pi / 2.0),
            mujoco_module=_FakeMujoco(lambda x, y: 0.0),
        )
        scanner.scan()
        # Local (-0.8, -0.5) rotated +90 degrees gives world (+0.5, -0.8).
        np.testing.assert_allclose(
            scanner.last_scan["ray_starts_w"][0, :2],
            [10.5, 19.2],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            scanner.last_scan["local_hits"][0, :2],
            [-0.8, -0.5],
            atol=1.0e-6,
        )

    def test_robot_geometries_are_excluded_from_terrain_rays(self):
        model = _FakeModel()
        MujocoAmeHeightMap(
            model,
            _FakeData(),
            mujoco_module=_FakeMujoco(lambda x, y: 0.0),
        )
        np.testing.assert_array_equal(model.geom_group, [0, 1, 1])

    def test_miss_and_deep_hit_are_clipped_to_negative_1p2(self):
        for terrain in (lambda x, y: None, lambda x, y: -5.0):
            scanner = MujocoAmeHeightMap(
                _FakeModel(),
                _FakeData(),
                mujoco_module=_FakeMujoco(terrain),
            )
            hits = scanner.scan().reshape(693, 3)
            np.testing.assert_allclose(hits[:, 2], -1.2, atol=1.0e-6)

    def test_raw_height_is_retained_when_positive_hits_are_clipped(self):
        scanner = MujocoAmeHeightMap(
            _FakeModel(),
            _FakeData(),
            mujoco_module=_FakeMujoco(lambda x, y: 1.0),
        )
        hits = scanner.scan().reshape(693, 3)
        np.testing.assert_allclose(scanner.last_scan["raw_local_z"], 0.16, atol=1.0e-6)
        np.testing.assert_allclose(hits[:, 2], 0.0, atol=1.0e-6)

    def test_default_dimensions(self):
        config = AmeHeightMapConfig()
        self.assertEqual((config.num_y, config.num_x), (21, 33))
        self.assertEqual(config.num_rays, 693)
        self.assertEqual(config.observation_dim, 2079)


if __name__ == "__main__":
    unittest.main()
