"""Run SuperMimic's S1 EMA planner on measured MuJoCo state and feed its future to S0."""

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from perspmimic.evaluation.evaluate_planner_kinematic import CARROT_M, REPLAN, load
from perspmimic.planning import features as F
from scan80000_policy import JOINT_NAMES, frontbackward_velocity


@dataclass
class ReferenceWindow:
    start_frame: int
    pos: np.ndarray
    quat: np.ndarray
    joints: np.ndarray
    vel: np.ndarray


class TerrainCoverageError(ValueError):
    """The S1 terrain query has left the finite box tile."""


def local_frames(pos, rot, joints, origin, yaw_rotation):
    return np.concatenate(((pos - origin) @ yaw_rotation, F.rot6d(yaw_rotation.T @ rot), joints), axis=-1)


def reanchor_reference(pos, quat, i, p, yaw, mode):
    """Align the old reference at this tick with the measured base in the horizontal plane."""
    if mode == "world":
        return pos, quat
    offset_xy = pos[:, :2] - pos[i, :2]
    if mode == "xy-yaw":
        angle = yaw - F.yaw(quat[i])
        offset_xy = offset_xy @ F.yaw_mat(angle)[:2, :2].T
        # World yaw left-multiplies the original wxyz base quaternion.
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        w, x, y, z = quat.T
        quat = np.stack((c*w - s*z, c*x - s*y, c*y + s*x, c*z + s*w), axis=1)
    pos = pos.copy()
    pos[:, :2] = p[:2] + offset_xy
    return pos, quat


class OnlinePlanner:
    def __init__(self, rt, checkpoint, goal, seed, *, reanchor=None):
        self.reanchor = "xy" if reanchor is None else reanchor
        if self.reanchor not in ("xy", "xy-yaw", "world"):
            raise ValueError(f"Unknown planner reanchor mode: {reanchor}")
        self.model, self.norm, ck = load(checkpoint, "cpu")
        self.checkpoint_step = ck["step"]
        radius = ck["meta"]["config"].get("goal_radius_m", 0.)
        if getattr(self.norm, "goal_radius", 0.) != radius:
            raise ValueError(f"S1 requires goal radius {radius} m; load the matching SuperMimic planner on PYTHONPATH")
        names = list(ck["meta"]["joint_names"])
        if sorted(names) != sorted(JOINT_NAMES):
            raise ValueError("S1 and S0 must control the same named joints")
        self.to_planner = [JOINT_NAMES.index(n) for n in names]
        self.to_tracker = [names.index(n) for n in JOINT_NAMES]
        self.prefix_len = int(ck["meta"]["config"].get("prefix", F.PREFIX))
        if not 0 <= self.prefix_len <= F.FUTURE - REPLAN:
            raise ValueError("S1 committed prefix extends past the previous plan")
        if not np.isclose(rt.dt * F.FPS, 1) or (rt.horizon - 1) * rt.stride + REPLAN >= F.FUTURE:
            raise ValueError("S1 reference rate or lookahead is incompatible with S0")
        self.seed = seed
        self.rng = torch.Generator(device="cpu")
        self.boxes = rt.motion.boxes
        if np.any(np.abs(self.boxes[:, 4:6]) > 1e-9):
            raise ValueError("S1 was trained on boxes with yaw rotation only")
        self.history = deque(maxlen=F.HISTORY)
        self.replans = 0
        self.last_planned_frame = 0
        origin = rt.motion.pos[0]
        yaw = F.yaw(rt.motion.quat[0])
        if not np.isfinite(goal).all():
            raise ValueError("S1 target must be finite")
        self.target_xy = origin[:2] + F.yaw_mat(yaw)[:2, :2] @ goal[:2]
        self.target_yaw = yaw + np.deg2rad(goal[2])
        self.target_ground = np.r_[self.target_xy, self.ground(self.target_xy)]
        if not np.isfinite(self.target_ground).all():
            raise ValueError("The planner target is outside the scene terrain")
        self.clearance = origin[2] - self.ground(origin[:2])

    def ground(self, xy):
        if len(self.boxes):
            return F.top_height(xy, self.boxes)
        return np.zeros(np.shape(xy)[:-1])

    def goal_error(self, qpos):
        yaw_delta = self.target_yaw - F.yaw(qpos[3:7])
        yaw_error = abs(np.rad2deg(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))))
        return np.linalg.norm(self.target_xy - qpos[:2]), yaw_error

    def conditions(self):
        poses, joints = map(np.stack, zip(*self.history))
        pos, rot = poses[:, :3], F.quat_mat(poses[:, 3:7])
        p, yaw = pos[-1], F.yaw(poses[-1, 3:7])
        Ry = F.yaw_mat(yaw)
        frames = local_frames(pos, rot, joints, p, Ry)
        clearance = pos[:, 2] - self.ground(pos[:, :2])
        history = np.column_stack((frames, -rot[:, 2, :], clearance))
        heightmap = (F.heightmap(p, yaw, self.boxes) if len(self.boxes)
                     else np.full((len(F.GRID_Y), len(F.GRID_X)), -p[2]))

        delta = self.target_xy - p[:2]
        distance = np.linalg.norm(delta)
        waypoint = self.target_xy.copy()
        heading = self.target_yaw
        if distance > CARROT_M:
            waypoint = p[:2] + delta * (CARROT_M / distance)
            heading = np.arctan2(delta[1], delta[0])
        xyz = np.r_[waypoint, self.ground(waypoint) + self.clearance]
        if not (np.isfinite(heightmap).all() and np.isfinite(clearance).all() and np.isfinite(xyz[2])):
            raise TerrainCoverageError("S1 terrain query extends beyond the scene tile")
        goal = np.r_[(xyz - p) @ Ry, np.sin(heading - yaw), np.cos(heading - yaw)]
        cond = {}
        for key, values in (("history", history), ("heightmap", heightmap), ("goal", goal)):
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite S1 {key} at replan {self.replans}")
            cond[key] = self.norm.norm(key, torch.as_tensor(values, dtype=torch.float32)[None])
        cond.update(history_valid=torch.tensor([True]), goal_valid=torch.tensor([True]))
        return cond, p, yaw, Ry

    def extend_last_reference(self, rt):
        """Keep the remaining plan and hold its final pose while the terrain query is unavailable."""
        reference, i = rt.ref, rt.reference_frame
        start = max(i - 1, 0)
        indices = np.minimum(np.arange(start, i + F.FUTURE + 1), len(reference.pos) - 1)
        joints = reference.joints[indices]
        rt.ref = ReferenceWindow(reference.start_frame + start, reference.pos[indices], reference.quat[indices],
                                 joints, frontbackward_velocity(joints, rt.dt))

    @torch.inference_mode()
    def update(self, rt):
        state = (rt.data.qpos[:7].copy(), rt.data.qpos[rt.q[self.to_planner]].copy())
        if rt.frame == 0:
            self.replans = 0
            self.last_planned_frame = 0
            self.rng.manual_seed(self.seed)
            self.history.clear()
            self.history.extend([state] * F.HISTORY)
        else:
            self.history.append(state)
        if rt.frame % REPLAN:
            return False

        cond, p, yaw, Ry = self.conditions()
        reference, i = rt.ref, rt.reference_frame
        # Frames repeated during a terrain outage are not a committed S1 prefix.
        k = min(self.prefix_len, max(0, self.last_planned_frame - rt.frame))
        # Keep t-1 (when available), t, and the k committed frames in one reference window.
        previous = min(i, 1)
        kept = slice(i - previous, i + 1 + k)
        ref_pos, ref_quat = reanchor_reference(reference.pos[kept], reference.quat[kept], previous,
                                               p, yaw, self.reanchor if rt.frame else "world")
        ref_joints = reference.joints[kept]
        prefix = torch.zeros(1, F.FUTURE, 38)
        mask = torch.zeros(1, F.FUTURE, dtype=torch.bool)
        if k:
            local = local_frames(ref_pos[-k:], F.quat_mat(ref_quat[-k:]), ref_joints[-k:, self.to_planner], p, Ry)
            prefix[0, :k] = self.norm.norm("future", torch.as_tensor(local, dtype=torch.float32))
            mask[0, :k] = True
        samples = self.model.sample(cond, prefix, mask, generator=self.rng)
        future = self.norm.denorm("future", samples)[0].numpy().astype(np.float64)
        pos, rot = F.to_world(future[k:], np.r_[p, yaw])
        if not all(np.isfinite(x).all() for x in (future, pos, rot)):
            raise FloatingPointError(f"Invalid S1 output at simulation frame {rt.frame}")
        quat = np.empty((len(rot), 4))
        for q, matrix in zip(quat, rot):
            mujoco.mju_mat2Quat(q, matrix.ravel())
        joints = np.concatenate((ref_joints, future[k:, 9:][:, self.to_tracker]))
        rt.ref = ReferenceWindow(rt.frame - previous, np.concatenate((ref_pos, pos)), np.concatenate((ref_quat, quat)),
                                 joints, frontbackward_velocity(joints, rt.dt))
        self.last_planned_frame = rt.frame + F.FUTURE
        self.replans += 1
        return True
