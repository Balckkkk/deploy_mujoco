import unittest

import numpy as np

from ame_deploy import build_policy_observation, compute_heading_yaw_command, load_config


ISAAC_POLICY_JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


class AmeDeployTest(unittest.TestCase):
    def test_config_uses_training_physx_joint_order(self):
        config, _ = load_config("config/ame_g1.yaml", None)
        self.assertEqual(config["policy_joints"], ISAAC_POLICY_JOINTS)
        self.assertTrue(config["heading_command"])
        self.assertEqual(config["heading_target"], 0.0)
        self.assertEqual(config["heading_control_stiffness"], 0.5)
        for key in ("default_angles", "kps", "kds", "effort_limits"):
            self.assertEqual(len(config[key]), 29)

    def test_heading_command_matches_training_controller(self):
        yaw = 0.4
        quaternion = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        command = compute_heading_yaw_command(
            quaternion,
            target_heading=0.0,
            stiffness=0.5,
            limits=(-1.0, 1.0),
        )
        self.assertAlmostEqual(command, -0.2)

        command = compute_heading_yaw_command(
            np.array([0.0, 0.0, 0.0, 1.0]),
            target_heading=0.0,
            stiffness=1.0,
            limits=(-1.0, 1.0),
        )
        self.assertAlmostEqual(command, -1.0)

    def test_policy_observation_term_order_and_scale(self):
        config, _ = load_config("config/ame_g1.yaml", None)
        default = np.asarray(config["default_angles"], dtype=np.float32)
        q_offset = np.arange(29, dtype=np.float32) / 100.0
        velocity = np.arange(29, dtype=np.float32)
        action = -np.arange(29, dtype=np.float32)
        height = np.arange(2079, dtype=np.float32)
        observation = build_policy_observation(
            angular_velocity=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            base_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            command=np.array([0.4, -0.2, 0.1], dtype=np.float32),
            joint_position=default + q_offset,
            joint_velocity=velocity,
            last_action=action,
            default_angles=default,
            height_observation=height,
            config=config,
        )
        self.assertEqual(observation.shape, (2175,))
        np.testing.assert_allclose(observation[0:3], [0.2, 0.4, 0.6])
        np.testing.assert_allclose(observation[3:6], [0.0, 0.0, -1.0])
        np.testing.assert_allclose(observation[6:9], [0.4, -0.2, 0.1])
        np.testing.assert_allclose(observation[9:38], q_offset, atol=1.0e-6)
        np.testing.assert_allclose(observation[38:67], velocity * 0.05)
        np.testing.assert_allclose(observation[67:96], action)
        np.testing.assert_array_equal(observation[96:], height)


if __name__ == "__main__":
    unittest.main()
