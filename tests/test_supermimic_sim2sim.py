from contextlib import nullcontext
import unittest

import mujoco
import numpy as np

from scan80000_policy import GOAL_DIM, SCAN_DIM, JOINT_NAMES, build_goal
from sim2sim.scan80000_terrain import TrainingHeightMap, add_terrain
from sim2sim.supermimic_sim2sim import POLICY, ScanReplay


class Scan80000ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = ScanReplay(POLICY / "motions/walk2_subject3.npz")

    def test_frozen_control_and_joint_order(self):
        rt = self.rt
        self.assertEqual((rt.model.nu, rt.model.nexclude), (58, 4))
        self.assertAlmostEqual(rt.dt, 0.02)
        self.assertAlmostEqual(rt.model.opt.timestep, 0.005)
        model_order = [rt.model.joint(i).name for i in range(rt.model.njnt)
                       if rt.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        self.assertEqual(model_order, JOINT_NAMES)
        for name, kp, kd, armature, limit, scale in [
            ("left_hip_pitch_joint", 40.179238, 2.557890, .01017752, 88, .5475465),
            ("left_knee_joint", 99.098428, 6.308802, .025101925, 139, .35066147),
            ("left_ankle_roll_joint", 28.501246, 1.814446, .00721945, 50, .4385773),
            ("right_wrist_yaw_joint", 16.778327, 1.068142, .00425, 5, .07450087),
        ]:
            with self.subTest(joint=name):
                i = JOINT_NAMES.index(name)
                pos = rt.model.actuator(name + "_pd_pos")
                vel = rt.model.actuator(name + "_pd_vel")
                self.assertEqual(pos.id, rt.ctrl[i])
                np.testing.assert_allclose([pos.gainprm[0], pos.biasprm[1], vel.biasprm[2]],
                                           [kp, -kp, -kd], atol=1e-6)
                np.testing.assert_allclose(rt.model.jnt_actfrcrange[rt.model.joint(name).id], [-limit, limit])
                self.assertAlmostEqual(rt.model.dof_armature[rt.v[i]], armature)
                np.testing.assert_allclose(rt.scale[i], scale, atol=1e-7)

    def test_reset_observation_and_action(self):
        rt = self.rt
        rt.reset()
        obs = rt.observation()
        self.assertEqual(obs.shape, (1601,))
        first_velocity = (rt.ref.joints[1] - rt.ref.joints[0]) / (2 * rt.dt)
        np.testing.assert_allclose(obs[1137:1369].reshape(8, 29), np.tile(first_velocity, (8, 1)), atol=1e-6)
        np.testing.assert_allclose(obs[881:905].reshape(8, 3), np.tile(rt.data.qvel[3:6], (8, 1)), atol=1e-6)
        action = rt.policy.infer(obs)
        rt.step()
        np.testing.assert_allclose(rt.data.ctrl[rt.ctrl], rt.default_q + rt.scale * action)
        np.testing.assert_array_equal(rt.observation()[-29:], action)
        rt.reset()
        np.testing.assert_array_equal(rt.observation(), obs)

    def test_end_of_clip_masks_and_stop(self):
        rt = self.rt
        rt.reset()
        rt.frame = len(rt.ref.pos) - 2
        goal = build_goal(rt.ref.pos, rt.ref.quat, rt.ref.joints, np.eye(3), rt.frame,
                          default_q=rt.default_q, ref_joint_vel=rt.ref.vel)
        for term, width in [(goal[:290], 29), (goal[290:580], 29), (goal[610:], 6)]:
            np.testing.assert_array_equal(term.reshape(10, width)[1:], 0)
        # The position command holds the last displacement; its invalid slots are not masked.
        expected = (rt.ref.pos[-1] - rt.ref.pos[-2])
        np.testing.assert_allclose(np.linalg.norm(goal[580:610].reshape(10, 3)[1:], axis=1),
                                   np.linalg.norm(expected), atol=1e-6)
        rt.frame += 1
        with self.assertRaises(StopIteration):
            rt.step()
        rt.reset()

    def test_heightmap_view_uses_policy_frame_and_yaw(self):
        from sim2sim.scan80000_view import draw_heightmap, heightmap_panel, scan_colors

        rt = self.rt
        rt.reset()
        obs = rt.observation()
        vals = obs[GOAL_DIM:GOAL_DIM + SCAN_DIM]
        vals[:] = np.linspace(.45, 1.15, SCAN_DIM)
        vals[0] = rt.scanner.max_distance
        # Local +x points along world +y. Make stale-pose and axis-order errors visible.
        pose = np.array([2., -3., 1.5, 2**-.5, 0., 0., 2**-.5])

        class Viewer:
            viewport = mujoco.MjrRect(0, 0, 1200, 800)
            user_scn = mujoco.MjvScene(rt.model, maxgeom=SCAN_DIM)
            lock = staticmethod(nullcontext)

            def set_images(self, pair):
                self.rect, self.image = pair

        viewer = Viewer()
        draw_heightmap(viewer, rt.scanner, pose, obs)
        self.assertEqual(viewer.user_scn.ngeom, SCAN_DIM - 1)
        xyz = np.stack([g.pos for g in viewer.user_scn.geoms[:viewer.user_scn.ngeom]])
        x, y = rt.scanner.offsets[1:].T
        np.testing.assert_allclose(xyz, np.column_stack((2-y, -3+x, 1.5-vals[1:])), atol=1e-6)
        self.assertEqual(viewer.image.shape, (viewer.rect.height, viewer.rect.width, 3))
        # The top-left cell is forward-left (+x,+y), the last value in policy order.
        colors = scan_colors(vals)
        panel = np.asarray(heightmap_panel(vals, colors.reshape(*rt.scanner.grid_shape, 3)))
        np.testing.assert_array_equal(panel[50, 20], colors[-1])

    def test_reference_overlay_appends_current_world_pose(self):
        from sim2sim.scan80000_view import ReferenceGhost

        rt = self.rt
        rt.reset()
        live_qpos = rt.data.qpos.copy()
        ghost = ReferenceGhost(rt.model, rt.q)
        scene = mujoco.MjvScene(rt.model, maxgeom=256)
        expected = mujoco.MjData(rt.model)
        visual_ids = np.flatnonzero(rt.model.geom_group == 2)
        for frame in (0, 140):
            scene.ngeom = 1  # Existing heightmap marker.
            scene.geoms[0].type = mujoco.mjtGeom.mjGEOM_SPHERE
            ghost.append(scene, rt.ref, frame)
            self.assertEqual(scene.geoms[0].type, mujoco.mjtGeom.mjGEOM_SPHERE)
            visuals = scene.geoms[1:scene.ngeom]
            self.assertCountEqual([geom.objid for geom in visuals], visual_ids.tolist())
            expected.qpos[:3] = rt.ref.pos[frame]
            expected.qpos[3:7] = rt.ref.quat[frame]
            expected.qpos[rt.q] = rt.ref.joints[frame]
            mujoco.mj_forward(rt.model, expected)
            for geom in visuals:
                np.testing.assert_allclose(geom.pos, expected.geom_xpos[geom.objid], atol=1e-6)
                np.testing.assert_allclose(geom.mat.ravel(), expected.geom_xmat[geom.objid], atol=1e-6)
                self.assertTrue(0 < geom.rgba[3] < 1)
        np.testing.assert_array_equal(rt.data.qpos, live_qpos)

    def test_scan_interpolates_box_edge(self):
        spec = mujoco.MjSpec()
        boxes = np.array([[0, 0, -.05, 1, 0, 0, 0, 4, 4, .1],
                          [.255, 0, .2, 1, 0, 0, 0, .48, 1, .4]])
        add_terrain(spec, boxes)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        sensor = {"max_distance": 5, "pattern": {"resolution": .1, "size": [1.6, 1.0]}}
        scanner = TrainingHeightMap(model, data, sensor, .05)
        scan = scanner.scan(np.array([.025, 0, 1.0]), np.eye(3)).reshape(11, 17)
        # Lattice samples x=0 and x=.05 see ground and box respectively.
        np.testing.assert_allclose(scan[5, 7:10], [1, .8, .6], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
