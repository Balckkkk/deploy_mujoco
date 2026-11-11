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


class OnlinePlanner:
    def __init__(self, rt, checkpoint, goal, seed):
        self.model, self.norm, ck = load(checkpoint, "cpu")
        self.checkpoint_step = ck["step"]
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
        prefix = torch.zeros(1, F.FUTURE, 38)
        mask = torch.zeros(1, F.FUTURE, dtype=torch.bool)
        committed = slice(i + 1, i + 1 + k)
        if k:
            local = local_frames(reference.pos[committed], F.quat_mat(reference.quat[committed]),
                                 reference.joints[committed][:, self.to_planner], p, Ry)
            prefix[0, :k] = self.norm.norm("future", torch.as_tensor(local, dtype=torch.float32))
            mask[0, :k] = True
        samples = self.model.sample(cond, prefix, mask, generator=self.rng)
        future = self.norm.denorm("future", samples)[0].numpy().astype(np.float64)
        pos, rot = F.to_world(future, np.r_[p, yaw])
        if not all(np.isfinite(x).all() for x in (future, pos, rot)):
            raise FloatingPointError(f"Invalid S1 output at simulation frame {rt.frame}")
        quat = np.empty((len(future), 4))
        for q, matrix in zip(quat, rot):
            mujoco.mju_mat2Quat(q, matrix.ravel())
        joints = future[:, 9:][:, self.to_tracker]
        if k:
            # Keep the committed world frames bit-identical across the change of yaw frame.
            pos[:k], quat[:k], joints[:k] = reference.pos[committed], reference.quat[committed], reference.joints[committed]

        # Future[0] is t+1. Retain t and, after startup, t-1 for S0's 50 Hz central difference.
        past = slice(max(i - 1, 0), i + 1)
        pos = np.concatenate((reference.pos[past], pos))
        quat = np.concatenate((reference.quat[past], quat))
        joints = np.concatenate((reference.joints[past], joints))
        rt.ref = ReferenceWindow(rt.frame - min(i, 1), pos, quat, joints, frontbackward_velocity(joints, rt.dt))
        self.last_planned_frame = rt.frame + F.FUTURE
        self.replans += 1
        return True
