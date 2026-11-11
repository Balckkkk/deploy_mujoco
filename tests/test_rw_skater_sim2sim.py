import sys
import unittest
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim2sim"))

from rw_skater_sim2sim import (  # noqa: E402
    DEFAULT_CONFIG,
    OBS_TERMS,
    RwSkaterRuntime,
    VelocityCommand,
    load_config,
    pack_history_term_major,
)


class RwSkaterSim2SimTest(unittest.TestCase):
    def test_history_is_packed_term_major(self):
        sizes = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3, "joint_pos": 2, "joint_vel": 2, "last_action": 2}
        history = deque(
            {term: np.full(size, 10 * t + k, dtype=np.float32) for k, (term, size) in enumerate(sizes.items())}
            for t in range(3)
        )
        obs = pack_history_term_major(history)
        # First term: all 3 frames of base_ang_vel, oldest first.
        np.testing.assert_array_equal(obs[:9], [0, 0, 0, 10, 10, 10, 20, 20, 20])
        np.testing.assert_array_equal(obs[9:12], [1, 1, 1])
        self.assertEqual(obs.shape[0], 3 * sum(sizes.values()))
        self.assertEqual(tuple(sizes), OBS_TERMS)

    def test_command_clips_to_training_range(self):
        command = VelocityCommand([1.0, 0.3, -2.0], {"lin_vel_x": [-0.1, 0.5], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [-0.5, 0.5]}, [0.1, 0.1, 0.1])
        np.testing.assert_allclose(command.value, [0.5, 0.0, -0.5])
        command.from_axes(np.array([-1.0, 0.0, 0.5], dtype=np.float32))
        np.testing.assert_allclose(command.value, [-0.1, 0.0, 0.25])

    def test_config_matches_policy_and_model(self):
        for name in ("push_gait", "swizzle"):
            runtime = RwSkaterRuntime(load_config(DEFAULT_CONFIG, name))
            runtime.reset()
            self.assertEqual(runtime.num_obs, 420)
            self.assertAlmostEqual(runtime.policy_dt, 0.02)
            np.testing.assert_allclose(runtime.data.qpos[runtime.qpos_indices], runtime.default_joint_pos, atol=1e-6)
            self.assertEqual(runtime.policy_step().shape, (25,))

    def test_push_gait_tracks_forward_command(self):
        runtime = RwSkaterRuntime(load_config(DEFAULT_CONFIG, "push_gait"))
        ok = runtime.run_headless(segments=[(np.array([0.4, 0.0, 0.0], dtype=np.float32), 6.0)], log=False)
        self.assertTrue(ok)
        self.assertGreater(runtime.base_state()["x"], 1.0)


if __name__ == "__main__":
    unittest.main()
