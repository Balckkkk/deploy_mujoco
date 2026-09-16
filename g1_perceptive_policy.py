"""Standalone PyTorch inference for the G1 perceptive vault/climb policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


DEPTH_SHAPE = (1, 32, 48)
DEPTH_DIM = int(np.prod(DEPTH_SHAPE))
PROPRIO_DIM = 96
OBSERVATION_DIM = DEPTH_DIM + PROPRIO_DIM
ACTION_DIM = 29


class G1PerceptiveNetwork(nn.Module):
    """Checkpoint-compatible CNN+MLP actor, without training dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # ParallelLayer removes the depth pixels, retains the 96 proprio values,
        # and appends the 32-D depth latent in this order.
        self.actor = nn.Sequential(
            nn.Linear(PROPRIO_DIM + 32, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, ACTION_DIM),
        )

    def forward(self, depth_image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        latent = self.depth_encoder(depth_image)
        return self.actor(torch.cat((proprio, latent), dim=-1))


class G1PerceptivePolicy:
    """Load ``model_*.pt`` and run deterministic actor-mean inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        normalizer_epsilon: float = 1e-2,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.iteration = int(checkpoint.get("iter", -1))
        state = checkpoint["model_state_dict"]
        normalizer = checkpoint["policy_normalizer_state_dict"]

        self.network = G1PerceptiveNetwork().to(self.device)
        self._load_network_state(state)
        self.network.eval()

        mask = normalizer["_normalize_mask"].to(device=self.device, dtype=torch.bool)
        if mask.shape != (OBSERVATION_DIM,):
            raise ValueError(f"Normalizer mask shape={tuple(mask.shape)}, expected {(OBSERVATION_DIM,)}")
        expected_mask = torch.cat(
            (
                torch.zeros(DEPTH_DIM, dtype=torch.bool, device=self.device),
                torch.ones(PROPRIO_DIM, dtype=torch.bool, device=self.device),
            )
        )
        if not torch.equal(mask, expected_mask):
            raise ValueError("Checkpoint normalizer does not pass through exactly the depth-image slice")
        self.proprio_mean = normalizer["emp_norm._mean"].to(
            device=self.device, dtype=torch.float32
        )
        self.proprio_std = normalizer["emp_norm._std"].to(
            device=self.device, dtype=torch.float32
        )
        if self.proprio_mean.shape != (1, PROPRIO_DIM):
            raise ValueError(
                f"Normalizer mean shape={tuple(self.proprio_mean.shape)}, expected {(1, PROPRIO_DIM)}"
            )
        self.normalizer_epsilon = float(normalizer_epsilon)

    def _load_network_state(self, state: dict[str, torch.Tensor]) -> None:
        remapped: dict[str, torch.Tensor] = {}
        conv_prefix = "encoders._parallel_blocks.depth_image.conv.conv."
        for index in (0, 2, 4):
            remapped[f"depth_encoder.{index}.weight"] = state[f"{conv_prefix}{index}.weight"]
            remapped[f"depth_encoder.{index}.bias"] = state[f"{conv_prefix}{index}.bias"]
        for index in (0, 2, 4, 6):
            remapped[f"actor.{index}.weight"] = state[f"actor.{index}.weight"]
            remapped[f"actor.{index}.bias"] = state[f"actor.{index}.bias"]
        self.network.load_state_dict(remapped, strict=True)

    @torch.inference_mode()
    def infer(self, depth_image: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.shape == (DEPTH_SHAPE[1], DEPTH_SHAPE[2]):
            depth = depth[np.newaxis, ...]
        if depth.shape != DEPTH_SHAPE:
            raise ValueError(f"Depth image shape={depth.shape}, expected {DEPTH_SHAPE}")
        proprio_array = np.asarray(proprio, dtype=np.float32)
        if proprio_array.shape != (PROPRIO_DIM,):
            raise ValueError(f"Proprio shape={proprio_array.shape}, expected {(PROPRIO_DIM,)}")
        if not np.all(np.isfinite(depth)) or not np.all(np.isfinite(proprio_array)):
            raise ValueError("Policy input contains NaN or Inf")

        depth_tensor = torch.from_numpy(depth.copy()).unsqueeze(0).to(self.device)
        proprio_tensor = torch.from_numpy(proprio_array.copy()).unsqueeze(0).to(self.device)
        proprio_tensor = (proprio_tensor - self.proprio_mean) / (
            self.proprio_std + self.normalizer_epsilon
        )
        action = self.network(depth_tensor, proprio_tensor).squeeze(0).cpu().numpy()
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Invalid policy output: shape={action.shape}")
        return action.astype(np.float32, copy=False)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        flat = np.asarray(observation, dtype=np.float32).reshape(-1)
        if flat.shape != (OBSERVATION_DIM,):
            raise ValueError(f"Observation shape={flat.shape}, expected {(OBSERVATION_DIM,)}")
        return self.infer(flat[:DEPTH_DIM].reshape(DEPTH_SHAPE), flat[DEPTH_DIM:])


def build_proprioception(
    command: np.ndarray,
    projected_gravity: np.ndarray,
    base_angular_velocity: np.ndarray,
    joint_position_relative: np.ndarray,
    joint_velocity: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """Pack terms in the exact order stored in the training environment."""
    result = np.concatenate(
        (
            np.asarray(command, dtype=np.float32).reshape(3),
            np.asarray(projected_gravity, dtype=np.float32).reshape(3),
            np.asarray(base_angular_velocity, dtype=np.float32).reshape(3),
            np.asarray(joint_position_relative, dtype=np.float32).reshape(ACTION_DIM),
            np.asarray(joint_velocity, dtype=np.float32).reshape(ACTION_DIM),
            np.asarray(last_action, dtype=np.float32).reshape(ACTION_DIM),
        )
    )
    if result.shape != (PROPRIO_DIM,):
        raise ValueError(f"Packed proprio shape={result.shape}, expected {(PROPRIO_DIM,)}")
    return result
