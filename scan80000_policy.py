"""Standalone Scan80000 (perspmimic V5 NoEE scan) inference for the MuJoCo sim2sim deployment.

Mirrors ``perspmimic/models/motion_encoder_actor.py``: goal(670) is encoded into a 32-D latent, then decoded together
with scan(187) + 8-step proprioceptive history(744). The checkpoint is the frozen ``model_80000.pt``; observation
normalisation is ``(obs - mean) / (std + 1e-2)`` over the full 1601-D vector. Joint order is InstinctMJ's hinge order
(waist, hips, knees, ankles, then arms).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

GOAL_DIM = 670
SCAN_DIM = 187
PROPRIO_DIM = 744
OBS_DIM = GOAL_DIM + SCAN_DIM + PROPRIO_DIM
ACTION_DIM = 29

JOINT_NAMES = [
    "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# The nominal joint pose the policy was trained with (``robot.data.default_joint_pos_nominal`` at iteration 80000):
# a standing pose, not all zeros. proprio joint_pos = q - DEFAULT_Q, goal joint_pos_ref = q_ref - DEFAULT_Q,
# and the PD target = DEFAULT_Q + action_scale * action.
DEFAULT_Q = np.array([
    0.0, 0.0, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float32)


def _mlp(input_dim, hidden, output_dim):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.SiLU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class MotionEncoderActor(nn.Module):
    def __init__(self, hidden=(2048, 1024, 512, 256)):
        super().__init__()
        self.latent_dim = 32
        self.encoder = _mlp(GOAL_DIM, hidden, self.latent_dim)
        self.decoder = _mlp(self.latent_dim + SCAN_DIM + PROPRIO_DIM, hidden, ACTION_DIM)

    def encode(self, obs):
        return torch.tanh(self.encoder(obs[..., :GOAL_DIM]))

    def forward(self, obs):
        return self.decoder(torch.cat((self.encode(obs), obs[..., GOAL_DIM:]), dim=-1))


def quat_to_mat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def tan_norm(R):
    """[R x_axis, R z_axis] (6-D), matching ``instinct_mj.utils.math.quat_to_tan_norm``."""
    return np.concatenate((R[..., :, 0], R[..., :, 2]), -1)


class Scan80000Policy:
    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "cpu", eps: float = 1e-2):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = torch.device(device)
        ck = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.iteration = int(ck.get("iter", -1))
        self.network = MotionEncoderActor().to(self.device)
        state = {k[len("actor."):]: v for k, v in ck["model_state_dict"].items()
                 if k.startswith("actor.")}
        self.network.load_state_dict(state, strict=True)
        self.network.eval()
        norm = ck["policy_normalizer_state_dict"]
        self.mean = norm["_mean"].to(self.device, dtype=torch.float32)  # (1, 1601)
        self.std = norm["_std"].to(self.device, dtype=torch.float32)
        if self.mean.shape != (1, OBS_DIM):
            raise ValueError(f"Normalizer mean shape {tuple(self.mean.shape)}, expected (1, {OBS_DIM})")
        self.eps = eps

    @torch.inference_mode()
    def infer(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if x.shape != (OBS_DIM,):
            raise ValueError(f"obs shape {x.shape}, expected ({OBS_DIM},)")
        if not np.all(np.isfinite(x)):
            raise ValueError("policy input contains NaN/Inf")
        t = torch.from_numpy(x.copy()).unsqueeze(0).to(self.device)
        t = (t - self.mean) / (self.std + self.eps)
        action = self.network(t).squeeze(0).cpu().numpy().astype(np.float32)
        return action


def frontbackward_velocity(x, dt=1.0 / 50.0):
    """Center-difference velocity with half-forward/backward edges, matching ``estimate_velocity`` (frontbackward)."""
    v = np.zeros_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    v[0] = (x[1] - x[0]) / (2 * dt)
    v[-1] = (x[-1] - x[-2]) / (2 * dt)
    return v


def build_goal(ref_pos, ref_quat, ref_joint, robot_R, frame, horizon=10, stride=5,
               *, default_q=DEFAULT_Q, ref_joint_vel=None):
    """670-D goal from reference frames. ref_* are (T, ...); robot_R is the current robot base rotation (3,3).

    frame is the current reference frame index; future frames are frame, frame+stride, ..., frame+(horizon-1)*stride.
    """
    indices = frame + stride * np.arange(horizon)
    valid = indices < len(ref_pos)
    fut = np.minimum(indices, len(ref_pos) - 1)
    R_ref = quat_to_mat(ref_quat[frame])  # (3,3)
    jp = (ref_joint[fut] - default_q) * valid[:, None]
    if ref_joint_vel is None:
        ref_joint_vel = frontbackward_velocity(ref_joint)
    jv = ref_joint_vel[fut] * valid[:, None]
    pos_ref = (ref_pos[fut] - ref_pos[frame]) @ R_ref  # = R_ref^T @ (p_fut - p_f)
    rel = robot_R.T @ quat_to_mat(ref_quat[fut])
    rot_ref = tan_norm(rel) * valid[:, None]
    return np.concatenate((jp.flatten(), jv.flatten(), pos_ref.flatten(), rot_ref.flatten())).astype(np.float32)


def build_proprio(history):
    """Pack eight frames into 744 values, oldest first within each observation term."""
    terms = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "last_action")
    blocks = [np.stack([frame[term] for frame in history]).ravel() for term in terms]
    return np.concatenate(blocks).astype(np.float32)
