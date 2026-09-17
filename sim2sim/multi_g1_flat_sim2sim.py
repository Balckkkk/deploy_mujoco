#!/usr/bin/env python3
"""MuJoCo sim2sim runner for the MultiModalWBC multi_g1_flat policy.

The policy is a general motion-tracking actor.  It consumes the current motion
reference, relative pelvis orientation, robot proprioception, and previous
action.  In viewer mode the active motion can be hard-switched without closing
MuJoCo.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import os
import re
import sys
import time
from typing import Any

if "--headless" in sys.argv or "--check" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config/multi_g1_flat.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KEY_ENTER = 257
KEY_KP_ENTER = 335


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for key in ("model_path", "policy_path", "motion_dir", "output_dir"):
        config[key] = str(resolve_path(config[key]))
    return config


def normalize_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_inv(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat_wxyz)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_mul(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = normalize_quat(lhs_wxyz)
    bw, bx, by, bz = normalize_quat(rhs_wxyz)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(quat_wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def relative_orientation_matrix_6d(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
    relative = quat_mul(quat_inv(current_wxyz), target_wxyz)
    matrix = quat_to_matrix(relative)
    return matrix[:, :2].reshape(6).astype(np.float32)


def values_from_rules(
    joint_names: list[str],
    rules: list[dict[str, Any]],
    *,
    default: float,
    label: str,
) -> np.ndarray:
    values: list[float] = []
    for joint_name in joint_names:
        value = default
        matched = False
        for rule in rules:
            if re.fullmatch(str(rule["pattern"]), joint_name):
                value = float(rule["value"])
                matched = True
                break
        if not matched and default is None:
            raise ValueError(f"No {label} rule matched {joint_name}")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def setup_joint_mappings(
    model: mujoco.MjModel,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    actuator_indices: list[int] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if joint_id < 0:
            raise ValueError(f"Policy joint not found in MuJoCo model: {joint_name}")
        if actuator_id < 0:
            raise ValueError(f"Actuator not found in MuJoCo model: {joint_name}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        actuator_indices.append(int(actuator_id))
    return (
        np.asarray(qpos_indices, dtype=np.int32),
        np.asarray(qvel_indices, dtype=np.int32),
        np.asarray(actuator_indices, dtype=np.int32),
    )


def build_flat_scene_model(config: dict[str, Any]) -> mujoco.MjModel:
    """Compile a flat tracking scene with an explicit plane and fixed lights."""
    spec = mujoco.MjSpec.from_file(str(Path(config["model_path"]).resolve()))

    if len(spec.actuators) == 0:
        for joint in spec.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            actuator = spec.add_actuator()
            actuator.set_to_motor()
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.name = joint.name
            actuator.target = joint.name

    scene = config.get("scene", {})
    lighting = scene.get("lighting", {})
    headlight = lighting.get("headlight", {})
    spec.visual.headlight.ambient = headlight.get("ambient", [0.25, 0.25, 0.25])
    spec.visual.headlight.diffuse = headlight.get("diffuse", [0.75, 0.75, 0.75])
    spec.visual.headlight.specular = headlight.get("specular", [0.25, 0.25, 0.25])

    if spec.material("groundplane") is None:
        spec.add_material(name="groundplane", rgba=[0.34, 0.36, 0.38, 1.0])
    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor":
            spec.delete(geom)
    ground = scene.get("ground", {})
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=ground.get("size", [60.0, 60.0, 0.05]),
        friction=ground.get("friction", [1.0, 0.005, 0.0001]),
        rgba=ground.get("rgba", [0.34, 0.36, 0.38, 1.0]),
        material="groundplane",
        group=0,
        condim=3,
    )

    for light in lighting.get("lights", []):
        spec.worldbody.add_light(
            name=light["name"],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            pos=light.get("position", [2.0, -4.0, 6.0]),
            dir=light.get("direction", [-0.35, 0.45, -1.0]),
            ambient=light.get("ambient", [0.12, 0.12, 0.12]),
            diffuse=light.get("diffuse", [0.85, 0.82, 0.78]),
            specular=light.get("specular", [0.25, 0.25, 0.25]),
            castshadow=bool(light.get("cast_shadow", True)),
        )
    return spec.compile()


@dataclass
class MotionReference:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    anchor_pos: np.ndarray
    anchor_quat: np.ndarray
    anchor_lin_vel: np.ndarray
    anchor_ang_vel: np.ndarray
    frame: int


@dataclass
class MotionClip:
    path: Path
    name: str
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    anchor_body_index: int

    @classmethod
    def load(cls, path: Path, *, anchor_body_index: int) -> "MotionClip":
        with np.load(path, allow_pickle=False) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
            body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
            body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
            body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
            body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
            raise ValueError(f"{path} joint_pos shape={joint_pos.shape}, expected (T, 29)")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"{path} joint_vel shape={joint_vel.shape}, expected {joint_pos.shape}")
        if body_pos_w.ndim != 3 or body_quat_w.ndim != 3:
            raise ValueError(f"{path} has invalid body arrays")
        if anchor_body_index < 0 or anchor_body_index >= body_pos_w.shape[1]:
            raise ValueError(
                f"anchor_body_index={anchor_body_index} outside body array with {body_pos_w.shape[1]} bodies"
            )
        return cls(
            path=path,
            name=path.stem,
            fps=fps,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            anchor_body_index=anchor_body_index,
        )

    @property
    def length(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration(self) -> float:
        return self.length / max(self.fps, 1.0)

    def reference_at(self, frame_float: float) -> MotionReference:
        frame = int(np.clip(round(frame_float), 0, self.length - 1))
        body_index = self.anchor_body_index
        return MotionReference(
            joint_pos=self.joint_pos[frame].copy(),
            joint_vel=self.joint_vel[frame].copy(),
            anchor_pos=self.body_pos_w[frame, body_index].copy(),
            anchor_quat=normalize_quat(self.body_quat_w[frame, body_index]).astype(np.float64),
            anchor_lin_vel=self.body_lin_vel_w[frame, body_index].copy(),
            anchor_ang_vel=self.body_ang_vel_w[frame, body_index].copy(),
            frame=frame,
        )


class MotionLibrary:
    def __init__(
        self,
        motion_dir: Path,
        *,
        glob_pattern: str,
        name_filter: str | None,
        initial_motion: str | None,
        anchor_body_index: int,
    ) -> None:
        self.anchor_body_index = int(anchor_body_index)
        self.motion_dir = resolve_path(motion_dir)
        self.glob_pattern = glob_pattern
        self.name_filter = name_filter
        candidates = sorted(resolve_path(motion_dir).glob(glob_pattern))
        if name_filter:
            token = name_filter.lower()
            candidates = [path for path in candidates if token in path.stem.lower()]
        if not candidates:
            raise FileNotFoundError(f"No motion files found in {motion_dir} matching {glob_pattern!r}")
        self.paths = candidates
        self.index = self._initial_index(initial_motion)

    def _initial_index(self, initial_motion: str | None) -> int:
        if not initial_motion:
            return 0
        candidate = Path(initial_motion).expanduser()
        if candidate.suffix == ".npz" and candidate.exists():
            resolved = candidate.resolve()
            for index, path in enumerate(self.paths):
                if path.resolve() == resolved:
                    return index
            self.paths.insert(0, resolved)
            return 0
        token = initial_motion.lower()
        for index, path in enumerate(self.paths):
            if path.stem.lower() == token or token in path.stem.lower():
                return index
        raise ValueError(f"Could not find initial motion matching {initial_motion!r}")

    @property
    def count(self) -> int:
        return len(self.paths)

    def description(self) -> str:
        filter_text = self.name_filter if self.name_filter else "<none>"
        return (
            f"dir={self.motion_dir}, glob={self.glob_pattern}, "
            f"filter={filter_text}, loaded={self.count}"
        )

    def load(self, index: int) -> MotionClip:
        self.index = index % self.count
        return MotionClip.load(self.paths[self.index], anchor_body_index=self.anchor_body_index)


class MotionPlayer:
    def __init__(
        self,
        library: MotionLibrary,
        *,
        start_frame: int,
        policy_dt: float,
        loop: bool,
    ) -> None:
        self.library = library
        self.start_frame = int(start_frame)
        self.policy_dt = float(policy_dt)
        self.loop = bool(loop)
        self.paused = False
        self.clip = self.library.load(self.library.index)
        self.frame_float = float(np.clip(self.start_frame, 0, self.clip.length - 1))

    def load_index(self, index: int) -> MotionReference:
        self.clip = self.library.load(index)
        self.frame_float = float(np.clip(self.start_frame, 0, self.clip.length - 1))
        return self.reference()

    def next(self) -> MotionReference:
        return self.load_index(self.library.index + 1)

    def previous(self) -> MotionReference:
        return self.load_index(self.library.index - 1)

    def replay(self) -> MotionReference:
        self.frame_float = float(np.clip(self.start_frame, 0, self.clip.length - 1))
        return self.reference()

    def reference(self) -> MotionReference:
        return self.clip.reference_at(self.frame_float)

    def advance(self) -> None:
        if self.paused:
            return
        self.frame_float += self.clip.fps * self.policy_dt
        if self.frame_float >= self.clip.length:
            if self.loop:
                self.frame_float = float(self.start_frame)
            else:
                self.frame_float = float(self.clip.length - 1)
                self.paused = True

    def status(self) -> str:
        ref = self.reference()
        return (
            f"[{self.library.index + 1}/{self.library.count}] {self.clip.name} "
            f"frame={ref.frame}/{self.clip.length - 1} fps={self.clip.fps:.1f} "
            f"{'paused' if self.paused else 'playing'}"
        )


class MultiInputOnnxPolicy:
    def __init__(self, policy_path: Path, expected_terms: dict[str, int], action_dim: int) -> None:
        if not policy_path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        self.inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise ValueError(f"Expected one ONNX output, got {len(outputs)}")
        self.output_name = outputs[0].name
        self.input_names = [item.name for item in self.inputs]
        self.expected_terms = {name: int(dim) for name, dim in expected_terms.items()}
        self.action_dim = int(action_dim)
        self.flat_input_name: str | None = None
        self.flat_observation_dim = int(sum(self.expected_terms.values()))

        missing = [name for name in self.expected_terms if name not in self.input_names]
        if missing:
            if len(self.inputs) != 1:
                raise ValueError(f"ONNX policy is missing inputs: {missing}; inputs={self.input_names}")
            self.flat_input_name = self.inputs[0].name
            shape_dim = self.inputs[0].shape[-1] if self.inputs[0].shape else None
            if isinstance(shape_dim, int) and shape_dim != self.flat_observation_dim:
                raise ValueError(
                    f"ONNX input {self.flat_input_name} shape={self.inputs[0].shape}, "
                    f"expected last dim {self.flat_observation_dim}"
                )
        else:
            for item in self.inputs:
                if item.name not in self.expected_terms:
                    continue
                expected_dim = self.expected_terms[item.name]
                shape_dim = item.shape[-1] if item.shape else None
                if isinstance(shape_dim, int) and shape_dim != expected_dim:
                    raise ValueError(f"ONNX input {item.name} shape={item.shape}, expected last dim {expected_dim}")
        output_dim = outputs[0].shape[-1] if outputs[0].shape else None
        if isinstance(output_dim, int) and output_dim != self.action_dim:
            raise ValueError(f"ONNX output shape={outputs[0].shape}, expected last dim {self.action_dim}")

    def infer(self, terms: dict[str, np.ndarray]) -> np.ndarray:
        feed: dict[str, np.ndarray] = {}
        if self.flat_input_name is not None:
            observation_parts: list[np.ndarray] = []
            for name, expected_dim in self.expected_terms.items():
                if name not in terms:
                    raise ValueError(f"Missing policy input term {name}")
                array = np.asarray(terms[name], dtype=np.float32).reshape(-1)
                if array.shape != (expected_dim,):
                    raise ValueError(f"Policy term {name} shape={array.shape}, expected ({expected_dim},)")
                observation_parts.append(array)
            observation = np.concatenate(observation_parts, dtype=np.float32).reshape(1, self.flat_observation_dim)
            feed[self.flat_input_name] = observation
            action = np.asarray(self.session.run([self.output_name], feed)[0], dtype=np.float32).reshape(-1)
            if action.shape != (self.action_dim,):
                raise ValueError(f"Policy action shape={action.shape}, expected ({self.action_dim},)")
            if not np.all(np.isfinite(action)):
                raise ValueError("Policy produced NaN or Inf action")
            return action

        for item in self.inputs:
            name = item.name
            if name not in terms:
                raise ValueError(f"Missing policy input term {name}")
            expected_dim = self.expected_terms.get(name)
            array = np.asarray(terms[name], dtype=np.float32).reshape(1, -1)
            if expected_dim is not None and array.shape[-1] != expected_dim:
                raise ValueError(f"Policy term {name} shape={array.shape}, expected [1, {expected_dim}]")
            feed[name] = array
        action = np.asarray(self.session.run([self.output_name], feed)[0], dtype=np.float32).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Policy action shape={action.shape}, expected ({self.action_dim},)")
        if not np.all(np.isfinite(action)):
            raise ValueError("Policy produced NaN or Inf action")
        return action


class MultiG1FlatRuntime:
    def __init__(self, config: dict[str, Any], args: argparse.Namespace) -> None:
        from ame_terrain import build_mujoco_model_with_terrain, generate_ame_terrain

        self.config = config
        self.policy_dt = float(config["simulation_dt"]) * int(config["control_decimation"])
        self.terrain_name = str(config.get("terrain", "flat"))
        if self.terrain_name == "flat":
            self.model = build_flat_scene_model(config)
        else:
            terrain = generate_ame_terrain(
                self.terrain_name,
                difficulty=float(config.get("terrain_difficulty", 0.5)),
                seed=int(config.get("terrain_seed", 42)),
            )
            self.model = build_mujoco_model_with_terrain(config["model_path"], terrain)
        self.model.opt.timestep = float(config["simulation_dt"])
        self.data = mujoco.MjData(self.model)
        self.joint_names = list(config["policy_joints"])
        self.num_actions = int(config["num_actions"])
        if len(self.joint_names) != self.num_actions:
            raise ValueError(f"policy_joints has {len(self.joint_names)} entries, expected {self.num_actions}")
        self.qpos_indices, self.qvel_indices, self.actuator_indices = setup_joint_mappings(
            self.model, self.joint_names
        )
        self.default_joint_pos = values_from_rules(
            self.joint_names,
            config["default_joint_pos_rules"],
            default=0.0,
            label="default_joint_pos",
        )
        self.kp = values_from_rules(
            self.joint_names,
            config["stiffness_rules"],
            default=None,
            label="stiffness",
        )
        self.kd = values_from_rules(
            self.joint_names,
            config["damping_rules"],
            default=None,
            label="damping",
        )
        self.effort_limit = values_from_rules(
            self.joint_names,
            config["effort_limit_rules"],
            default=None,
            label="effort_limit",
        )
        self.armature = values_from_rules(
            self.joint_names,
            config["armature_rules"],
            default=0.0,
            label="armature",
        )
        self.model.dof_armature[self.qvel_indices] = self.armature
        self.action_scale = 0.25 * self.effort_limit / self.kp
        self.action_low = float(config["action_clip"][0])
        self.action_high = float(config["action_clip"][1])
        self.base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            str(config["base_body"]),
        )
        if self.base_body_id < 0:
            raise ValueError(f"Base body not found: {config['base_body']}")

        library = MotionLibrary(
            Path(config["motion_dir"]),
            glob_pattern=args.motion_glob,
            name_filter=args.motion_filter,
            initial_motion=args.motion,
            anchor_body_index=int(config.get("anchor_body_index", 0)),
        )
        self.player = MotionPlayer(
            library,
            start_frame=args.start_frame if args.start_frame is not None else int(config.get("start_frame", 0)),
            policy_dt=self.policy_dt,
            loop=bool(config.get("loop_motion", True)),
        )
        self.policy = MultiInputOnnxPolicy(
            Path(config["policy_path"]),
            {str(key): int(value) for key, value in config["observation_terms"].items()},
            self.num_actions,
        )
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_q = self.default_joint_pos.copy()
        self.last_ref = self.player.reference()

    def reset_to_reference(self, ref: MotionReference | None = None) -> None:
        ref = ref or self.player.reference()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:3] = np.asarray(ref.anchor_pos, dtype=np.float64)
        self.data.qpos[3:7] = normalize_quat(ref.anchor_quat)
        self.data.qpos[self.qpos_indices] = np.asarray(ref.joint_pos, dtype=np.float64)
        self.data.qvel[:] = 0.0
        self.data.qvel[:3] = np.asarray(ref.anchor_lin_vel, dtype=np.float64)
        self.data.qvel[3:6] = np.asarray(ref.anchor_ang_vel, dtype=np.float64)
        self.data.qvel[self.qvel_indices] = np.asarray(ref.joint_vel, dtype=np.float64)
        self.last_action[:] = 0.0
        self.target_q = ref.joint_pos.astype(np.float32, copy=True)
        self.last_ref = ref
        mujoco.mj_forward(self.model, self.data)

    def base_angular_velocity(self) -> np.ndarray:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            velocity,
            1,
        )
        return velocity[:3].astype(np.float32)

    def build_observation_terms(self, ref: MotionReference) -> dict[str, np.ndarray]:
        command = np.concatenate((ref.joint_pos, ref.joint_vel), dtype=np.float32)
        current_anchor_quat = self.data.xquat[self.base_body_id]
        motion_anchor_ori_b = relative_orientation_matrix_6d(current_anchor_quat, ref.anchor_quat)
        joint_pos = self.data.qpos[self.qpos_indices].astype(np.float32) - self.default_joint_pos
        joint_vel = self.data.qvel[self.qvel_indices].astype(np.float32)
        return {
            "command": command,
            "motion_anchor_ori_b": motion_anchor_ori_b,
            "base_ang_vel": self.base_angular_velocity(),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "actions": np.clip(self.last_action, self.action_low, self.action_high).astype(np.float32),
        }

    def policy_step(self) -> np.ndarray:
        ref = self.player.reference()
        terms = self.build_observation_terms(ref)
        raw_action = self.policy.infer(terms)
        clipped_action = np.clip(raw_action, self.action_low, self.action_high)
        self.target_q = self.default_joint_pos + self.action_scale * clipped_action
        self.last_action = raw_action.astype(np.float32, copy=True)
        self.last_ref = ref
        self.player.advance()
        return raw_action

    def physics_step(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_indices]
        dq = self.data.qvel[self.qvel_indices]
        torque = self.kp * (self.target_q - q) - self.kd * dq
        torque = np.clip(torque, -self.effort_limit, self.effort_limit)
        self.data.ctrl[self.actuator_indices] = torque
        mujoco.mj_step(self.model, self.data)
        return torque.astype(np.float32)

    def print_status(self, prefix: str = "Motion") -> None:
        print(f"{prefix}: {self.player.status()}")

    def handle_command(self, command: str) -> None:
        if command == "next":
            ref = self.player.next()
            self.reset_to_reference(ref)
            self.policy_step()
            self.print_status("Switched")
        elif command == "previous":
            ref = self.player.previous()
            self.reset_to_reference(ref)
            self.policy_step()
            self.print_status("Switched")
        elif command == "replay":
            ref = self.player.replay()
            self.reset_to_reference(ref)
            self.policy_step()
            self.print_status("Replayed")
        elif command == "toggle_pause":
            self.player.paused = not self.player.paused
            self.print_status("Paused" if self.player.paused else "Resumed")
        elif command == "reset":
            ref = self.player.reference()
            self.reset_to_reference(ref)
            self.policy_step()
            self.print_status("Reset")

    def run_check(self) -> None:
        self.reset_to_reference()
        terms = self.build_observation_terms(self.player.reference())
        action = self.policy.infer(terms)
        print(f"Model: nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")
        print(f"Policy: {self.config['policy_path']}")
        print(f"Scene: terrain={self.terrain_name}, explicit flat ground={'yes' if self.terrain_name == 'flat' else 'no'}")
        print(f"Inputs: {self.policy.input_names} -> {self.policy.output_name}")
        print(f"Motion library: {self.player.library.description()}")
        print(f"Current motion: {self.player.clip.path}")
        print(f"Motion: {self.player.status()}")
        print(
            "Observation terms: "
            + ", ".join(f"{name}={np.asarray(value).shape[0]}" for name, value in terms.items())
        )
        print(
            f"Action: shape={action.shape}, min={action.min():.4f}, "
            f"max={action.max():.4f}, mean={action.mean():.4f}"
        )
        print("multi_g1_flat sim2sim check passed.")

    def open_logger(self, output_dir: Path, enabled: bool) -> tuple[Any, csv.DictWriter | None]:
        if not enabled:
            return None, None
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.player.clip.name}_trajectory.csv"
        file = path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "time",
                "motion",
                "frame",
                "pelvis_x",
                "pelvis_y",
                "pelvis_z",
                "action_min",
                "action_max",
                "torque_min",
                "torque_max",
            ],
        )
        writer.writeheader()
        print(f"Logging trajectory: {path}")
        return file, writer

    def log_row(self, writer: csv.DictWriter | None, torque: np.ndarray) -> None:
        if writer is None:
            return
        writer.writerow(
            {
                "time": f"{self.data.time:.4f}",
                "motion": self.player.clip.name,
                "frame": self.last_ref.frame,
                "pelvis_x": f"{self.data.qpos[0]:.6f}",
                "pelvis_y": f"{self.data.qpos[1]:.6f}",
                "pelvis_z": f"{self.data.qpos[2]:.6f}",
                "action_min": f"{self.last_action.min():.6f}",
                "action_max": f"{self.last_action.max():.6f}",
                "torque_min": f"{torque.min():.6f}",
                "torque_max": f"{torque.max():.6f}",
            }
        )

    def run_headless(self, *, duration: float, log: bool) -> None:
        self.reset_to_reference()
        self.policy_step()
        print(f"Motion library: {self.player.library.description()}")
        output_dir = Path(self.config["output_dir"])
        log_file, writer = self.open_logger(output_dir, log)
        counter = 0
        torque = np.zeros(self.num_actions, dtype=np.float32)
        try:
            while self.data.time < duration:
                torque = self.physics_step()
                counter += 1
                if counter % int(self.config["control_decimation"]) == 0:
                    self.policy_step()
                    self.log_row(writer, torque)
        finally:
            if log_file is not None:
                log_file.close()
        print(f"Headless run finished: sim_time={self.data.time:.3f}s, {self.player.status()}")

    def run_viewer(self, *, duration: float, realtime: bool, log: bool) -> None:
        self.reset_to_reference()
        self.policy_step()
        print(f"Motion library: {self.player.library.description()}")
        print(f"Scene: terrain={self.terrain_name}, explicit flat ground={'yes' if self.terrain_name == 'flat' else 'no'}")
        self.print_status("Loaded")
        print("Keys: N next, P previous, R replay, Space pause/resume, Enter reset current")
        pending: deque[str] = deque()

        def key_callback(keycode: int) -> None:
            if keycode in (ord("N"), ord("n")):
                pending.append("next")
            elif keycode in (ord("P"), ord("p")):
                pending.append("previous")
            elif keycode in (ord("R"), ord("r")):
                pending.append("replay")
            elif keycode == ord(" "):
                pending.append("toggle_pause")
            elif keycode in (KEY_ENTER, KEY_KP_ENTER):
                pending.append("reset")

        output_dir = Path(self.config["output_dir"])
        log_file, writer = self.open_logger(output_dir, log)
        counter = 0
        torque = np.zeros(self.num_actions, dtype=np.float32)
        wall_start = time.perf_counter()
        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
                while viewer.is_running() and self.data.time < duration:
                    step_start = time.perf_counter()
                    while pending:
                        self.handle_command(pending.popleft())
                    torque = self.physics_step()
                    counter += 1
                    if counter % int(self.config["control_decimation"]) == 0:
                        self.policy_step()
                        self.log_row(writer, torque)
                    viewer.sync()
                    if realtime:
                        elapsed = time.perf_counter() - step_start
                        sleep_time = self.model.opt.timestep - elapsed
                        if sleep_time > 0.0:
                            time.sleep(sleep_time)
        finally:
            if log_file is not None:
                log_file.close()
        wall_elapsed = time.perf_counter() - wall_start
        ratio = self.data.time / wall_elapsed if wall_elapsed > 0.0 else 0.0
        print(f"Viewer run finished: sim={self.data.time:.2f}s wall={wall_elapsed:.2f}s ratio={ratio:.3f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MultiModalWBC multi_g1_flat MuJoCo sim2sim")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to multi_g1_flat YAML config")
    parser.add_argument("--policy", default=None, help="Override exported ONNX policy path")
    parser.add_argument("--motion-dir", default=None, help="Override directory containing motion .npz files")
    parser.add_argument("--motion-glob", default="*.npz", help="Glob used inside --motion-dir")
    parser.add_argument("--motion-filter", default=None, help="Only list motions whose name contains this text")
    parser.add_argument("--motion", default=None, help="Initial motion name substring or .npz path")
    parser.add_argument("--start-frame", type=int, default=None, help="Frame used when replaying or switching")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration in seconds")
    parser.add_argument("--check", action="store_true", help="Validate model, motion, observation, and ONNX")
    parser.add_argument("--headless", action="store_true", help="Run without MuJoCo viewer")
    parser.add_argument("--no-realtime", action="store_true", help="Do not sleep to wall-clock timestep")
    parser.add_argument("--no-log", action="store_true", help="Disable trajectory CSV logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    if args.policy is not None:
        config["policy_path"] = str(resolve_path(args.policy))
    if args.motion_dir is not None:
        config["motion_dir"] = str(resolve_path(args.motion_dir))
    duration = float(args.duration if args.duration is not None else config["simulation_duration"])
    realtime = bool(config.get("realtime", True)) and not args.no_realtime
    runtime = MultiG1FlatRuntime(config, args)
    if args.check:
        runtime.run_check()
    elif args.headless:
        runtime.run_headless(duration=duration, log=not args.no_log)
    else:
        runtime.run_viewer(duration=duration, realtime=realtime, log=not args.no_log)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
