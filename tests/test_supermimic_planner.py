import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from sim2sim.supermimic_sim2sim import DEFAULT_PLANNER_CHECKPOINT, POLICY, ReferenceMotion, ScanReplay


@unittest.skipUnless(importlib.util.find_spec("perspmimic") and DEFAULT_PLANNER_CHECKPOINT.is_file(),
                     "S1 integration needs the SuperMimic checkout on PYTHONPATH and the S1 weights")
class PlannerIntegrationTest(unittest.TestCase):
    def setUp(self):
        from sim2sim.supermimic_planner import OnlinePlanner

        self.rt = ScanReplay(POLICY / "motions/low_step.npz", initial_pose_only=True)
        self.planner = OnlinePlanner(self.rt, DEFAULT_PLANNER_CHECKPOINT, (2, 0, 0), 0)

    def test_conditions_match_training_window(self):
        from perspmimic.planning import features as F

        rt, planner = self.rt, self.planner
        motion = ReferenceMotion(POLICY / "motions/low_step.npz", 1 / rt.dt)
        t, g = 48, 55
        pos, quat = motion.pos, motion.quat
        joints = motion.joints[:, planner.to_planner]
        for n in range(t - F.HISTORY + 1, t + 1):
            planner.history.append((np.r_[pos[n], quat[n]], joints[n]))
        planner.target_xy = pos[g, :2]
        planner.target_yaw = F.yaw(quat[g])
        planner.clearance = pos[g, 2] - planner.ground(pos[g, :2])
        cond, *_ = planner.conditions()
        expected = F.window(pos, quat, joints, motion.boxes, t, g)
        for key in ("history", "heightmap", "goal"):
            raw = planner.norm.denorm(key, cond[key])[0].numpy()
            np.testing.assert_allclose(raw, expected[key], atol=2e-6)

    def test_r3_platform_scene_uses_bounded_goal_and_real_start_pose(self):
        from perspmimic.planning import features as F
        from sim2sim.supermimic_planner import OnlinePlanner
        from sim2sim.supermimic_sim2sim import TOP_SCENES

        checkpoint = TOP_SCENES.parent / "ckpt_070000.pt"
        if not checkpoint.is_file():
            self.skipTest("R3 integration requires its local S1 weights")
        for name, height in (("step_top", .35235), ("climb_top", .76382)):
            path = TOP_SCENES / (name + ".npz")
            rt = ScanReplay(path, initial_pose_only=True)
            with np.load(path) as clip:
                planner = OnlinePlanner(rt, checkpoint, clip["goal"], 0)
            self.assertEqual(planner.checkpoint_step, 70000)
            self.assertAlmostEqual(planner.target_ground[2], height, places=4)
            start = rt.data.qpos[:7].copy()
            rot = F.yaw_mat(F.yaw(start[3:]))
            planner.target_xy = start[:2] + rot[:2, 0] * .4
            planner.history.extend([(start, rt.data.qpos[rt.q][planner.to_planner])] * F.HISTORY)
            cond, *_ = planner.conditions()
            raw = planner.norm.denorm("goal", cond["goal"])[0].numpy()
            np.testing.assert_allclose(raw[:2], [.8, 0.], atol=2e-6)

    def test_replan_preserves_world_prefix_and_uses_physics_history(self):
        rt, planner = self.rt, self.planner
        planner.reanchor = "world"
        np.testing.assert_array_equal(rt.data.qvel, 0)
        self.assertEqual(len(rt.motion.pos), 1)
        planner.update(rt)
        first = rt.ref
        self.assertEqual((first.start_frame, len(first.pos), planner.replans), (0, 65, 1))
        np.testing.assert_array_equal(first.pos[0], rt.motion.pos[0])
        for tick in range(1, 11):
            rt.step()
            planner.update(rt)
            self.assertEqual(planner.replans, 1 if tick < 10 else 2)
        i = rt.reference_frame
        self.assertEqual((rt.ref.start_frame, i), (9, 1))
        np.testing.assert_array_equal(planner.history[-1][0], rt.data.qpos[:7])
        np.testing.assert_array_equal(planner.history[-1][1], rt.data.qpos[rt.q[planner.to_planner]])
        for attr in ("pos", "quat", "joints"):
            np.testing.assert_array_equal(getattr(rt.ref, attr)[:22], getattr(first, attr)[9:31])
        np.testing.assert_allclose(rt.ref.vel[i], (first.joints[11] - first.joints[9]) / (2 * rt.dt))
        self.assertEqual(rt.observation().shape, (1601,))
        self.assertTrue(np.isfinite(rt.observation()).all())

    def test_reanchor_uses_same_tick_for_model_prefix_and_tracker_reference(self):
        from perspmimic.planning import features as F
        from sim2sim.supermimic_planner import REPLAN

        rt, planner = self.rt, self.planner
        self.assertEqual(planner.reanchor, "xy")
        for mode in (planner.reanchor, "xy-yaw"):
            with self.subTest(mode=mode):
                rt.reset()
                planner.reanchor = mode
                for frame in range(REPLAN):
                    planner.update(rt)
                    rt.step()
                old, i = rt.ref, rt.reference_frame
                k = planner.prefix_len
                rt.data.qpos[:3] = old.pos[i] + [.24, -.12, .06]
                yaw = F.yaw(old.quat[i]) + .4
                rt.data.qpos[3:7] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
                with patch.object(planner.model, "sample", wraps=planner.model.sample) as sample:
                    self.assertTrue(planner.update(rt))
                kept = slice(i - 1, i + k + 1)
                R = F.yaw_mat(.4 if mode == "xy-yaw" else 0.)
                expected = (old.pos[kept] - old.pos[i]) @ R.T + np.r_[rt.data.qpos[:2], old.pos[i, 2]]
                expected_rot = R @ F.quat_mat(old.quat[kept])
                np.testing.assert_allclose(rt.ref.pos[:k+2], expected, atol=1e-12)
                np.testing.assert_allclose(F.quat_mat(rt.ref.quat[:k+2]), expected_rot, atol=1e-12)
                np.testing.assert_array_equal(rt.ref.joints[:k+2], old.joints[kept])

                _, prefix, mask = sample.call_args.args
                self.assertEqual(int(mask.sum()), k)
                local = planner.norm.denorm("future", prefix)[0, :k].numpy().astype(float)
                pos, rot = F.to_world(local, np.r_[rt.data.qpos[:3], yaw])
                np.testing.assert_allclose(pos, expected[2:], atol=2e-6)
                np.testing.assert_allclose(rot, expected_rot[2:], atol=2e-6)

    def test_missing_terrain_holds_exhausted_plan_and_resumes_with_only_real_prefix(self):
        from perspmimic.planning import features as F
        from sim2sim.supermimic_planner import TerrainCoverageError

        rt, planner = self.rt, self.planner
        available_at = {10: 0, 70: 4, 140: 0}  # replan frame -> genuine frames still available
        last_plan = rt.ref
        conditions = planner.conditions

        def terrain_query():
            if rt.frame in available_at:
                return conditions()
            raise TerrainCoverageError("tile unavailable")

        with patch.object(planner, "conditions", side_effect=terrain_query), \
                patch.object(planner.model, "sample", wraps=planner.model.sample) as sample:
            for frame in range(141):
                try:
                    replanned = planner.update(rt)
                except TerrainCoverageError:
                    planner.extend_last_reference(rt)
                    replanned = False
                self.assertEqual(replanned, frame in available_at)
                if replanned:
                    mask = sample.call_args.args[2]
                    self.assertEqual(int(mask.sum()), available_at[frame])
                    last_plan = rt.ref
                i = rt.reference_frame
                source = min(frame - last_plan.start_frame, len(last_plan.pos) - 1)
                for field in ("pos", "quat", "joints"):
                    np.testing.assert_array_equal(getattr(rt.ref, field)[i], getattr(last_plan, field)[source])
                if frame > planner.last_planned_frame:
                    np.testing.assert_array_equal(rt.ref.vel[i], 0)
                self.assertLessEqual(len(rt.ref.pos), F.FUTURE + 2)
                rt.step()

    def test_changed_height_and_offset_reach_physics_and_both_scanners(self):
        import mujoco
        from perspmimic.planning import features as F
        from sim2sim.supermimic_planner import OnlinePlanner

        rt = ScanReplay(POLICY / "motions/low_step.npz", initial_pose_only=True, box_height=.55, box_offset_y=1.5)
        planner = OnlinePlanner(rt, DEFAULT_PLANNER_CHECKPOINT, (6, 0, 0), 0)
        original_xy = self.rt.motion.boxes[1, :2]
        groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        for xy, height in ((original_xy, 0.), (original_xy + [0., 1.5], .55)):
            pose = np.r_[xy, 1.4, 1., 0., 0., 0.]
            ray = mujoco.mj_ray(rt.model, rt.data, pose[:3], [0, 0, -1], groups, 1, -1, np.empty(1, np.int32))
            s0 = rt.scanner.scan(pose[:3], np.eye(3))
            planner.history.clear()
            planner.history.extend([(pose, rt.data.qpos[rt.q])] * F.HISTORY)
            cond, *_ = planner.conditions()
            s1 = planner.norm.denorm("heightmap", cond["heightmap"])[0]
            np.testing.assert_allclose([pose[2] - ray, pose[2] - s0[len(s0)//2],
                                        pose[2] + float(s1[15, 20])], height, atol=1e-5)
