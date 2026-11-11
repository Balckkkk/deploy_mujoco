"""Replay the frozen SuperMimic Scan80000 tracker in single-robot MuJoCo."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from pathlib import Path
import re
import sys
import time

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scan80000_policy import (ACTION_DIM, JOINT_NAMES, Scan80000Policy, build_goal,
                              build_proprio, frontbackward_velocity, quat_to_mat)
from sim2sim.scan80000_terrain import TrainingHeightMap, add_terrain

ASSET = ROOT / "asset/sp_g1"
POLICY = ROOT / "policy/scan80000"
DEFAULT_MOTION = POLICY / "motions/low_step.npz"


def joint_values(rules, default=None):
    values = []
    for name in JOINT_NAMES:
        matches = [value for pattern, value in rules.items() if re.fullmatch(pattern, name)]
        if len(matches) > 1 or (not matches and default is None):
            raise ValueError(f"Expected a single configuration rule for {name}, got {matches}")
        values.append(matches[0] if matches else default)
    return np.asarray(values, dtype=np.float32)


def configure_robot(spec, robot):
    """Reproduce mjlab BuiltinPdActuator's position + velocity pair and joint sum clamp."""
    for group in robot["articulation"]["actuators"]:
        names = [n for n in JOINT_NAMES if any(re.fullmatch(p, n) for p in group["target_names_expr"])]
        for name in names:
            joint = spec.joint(name)
            joint.armature = group["armature"]
            joint.actfrclimited = True
            joint.actfrcrange[:] = [-group["effort_limit"], group["effort_limit"]]
            pos = spec.add_actuator(name=f"{name}_pd_pos", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            pos.set_to_position(group["stiffness"])
            pos.ctrllimited = False
        for name in names:
            vel = spec.add_actuator(name=f"{name}_pd_vel", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            vel.set_to_velocity(group["damping"])
            vel.ctrllimited = False


def configure_physics(option, physics):
    for key in ("timestep", "impratio", "iterations", "tolerance", "ls_iterations", "ls_tolerance", "ccd_iterations"):
        setattr(option, key, physics[key])
    option.gravity[:] = physics["gravity"]
    option.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + physics["integrator"].upper())
    option.solver = getattr(mujoco.mjtSolver, "mjSOL_" + physics["solver"].upper())
    option.cone = getattr(mujoco.mjtCone, "mjCONE_" + physics["cone"].upper())
    option.jacobian = getattr(mujoco.mjtJacobian, "mjJAC_" + physics["jacobian"].upper())


class ReferenceMotion:
    """50 Hz planner-dataset NPZ in the original terrain tile frame."""

    def __init__(self, path, fps):
        with np.load(path, allow_pickle=False) as clip:
            rate = float(clip["framerate"].item())
            if not np.isclose(rate, fps):
                raise ValueError(f"{path} is {rate:g} Hz; expected a {fps:g} Hz planner-dataset export")
            self.pos = np.asarray(clip["base_pos_w"], dtype=np.float64)
            self.quat = np.asarray(clip["base_quat_w"], dtype=np.float64)
            names = clip["joint_names"].astype(str).tolist()
            if len(names) != len(set(names)) or not set(JOINT_NAMES).issubset(names):
                raise ValueError(f"{path} has duplicate or missing joint names")
            self.joints = np.asarray(clip["joint_pos"], dtype=np.float64)[:, [names.index(n) for n in JOINT_NAMES]]
            self.boxes = np.asarray(clip["boxes"], dtype=np.float64) if "boxes" in clip else np.empty((0, 10))
        n = len(self.pos)
        if n < 2 or self.pos.shape != (n, 3) or self.quat.shape != (n, 4) or self.joints.shape != (n, ACTION_DIM):
            raise ValueError(f"Inconsistent frame arrays in {path}")
        if self.boxes.ndim != 2 or self.boxes.shape[1] != 10 or np.any(self.boxes[:, 7:] <= 0):
            raise ValueError(f"Expected box centers, wxyz rotations and positive full sizes in {path}")
        if not all(np.isfinite(x).all() for x in (self.pos, self.quat, self.joints, self.boxes)):
            raise ValueError(f"Nonfinite motion or terrain in {path}")
        for rot in (self.quat, self.boxes[:, 3:7]):
            if not np.allclose(np.linalg.norm(rot, axis=1), 1, atol=1e-3):
                raise ValueError(f"Expected unit wxyz quaternions in {path}")
        self.quat /= np.linalg.norm(self.quat, axis=1, keepdims=True)
        self.vel = frontbackward_velocity(self.joints, 1 / fps)


class ScanReplay:
    def __init__(self, motion=DEFAULT_MOTION, checkpoint=POLICY / "model_80000.pt"):
        env = yaml.safe_load((ASSET / "params/env.yaml").read_text())
        agent = yaml.safe_load((ASSET / "params/agent.yaml").read_text())
        robot = env["scene"]["entities"]["robot"]
        sensors = {sensor["name"]: sensor for sensor in env["scene"]["sensors"]}
        ref = sensors["motion_reference"]
        buffer = ref["motion_buffers"]["TerrainMotion"]
        fps = buffer["motion_target_framerate"]
        self.decimation = env["decimation"]
        self.dt = env["sim"]["mujoco"]["timestep"] * self.decimation
        if not np.isclose(fps * self.dt, 1):
            raise ValueError("Expected one reference frame per policy step")
        self.horizon = ref["num_frames"]
        self.stride = round(ref["frame_interval_s"] * fps)
        self.lift = buffer["motion_start_height_offset"]
        self.default_q = joint_values(robot["init_state"]["joint_pos"], default=0.0)
        self.scale = joint_values(env["actions"]["joint_pos"]["scale"])
        self.ref = ReferenceMotion(motion, fps)
        spec = mujoco.MjSpec.from_file(str(ASSET / "xml" / Path(ref["robot_model_path"]).name))
        configure_robot(spec, robot)
        configure_physics(spec.option, env["sim"]["mujoco"])
        add_terrain(spec, self.ref.boxes)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(n).id for n in JOINT_NAMES]
        self.q = self.model.jnt_qposadr[joints]
        self.v = self.model.jnt_dofadr[joints]
        self.ctrl = [self.model.actuator(n + "_pd_pos").id for n in JOINT_NAMES]
        self.links = [self.model.body(n).id for n in ref["link_of_interests"]]
        self.history = deque(maxlen=env["observations"]["policy"]["terms"]["joint_pos"]["history_length"])
        self.last_action = np.zeros(ACTION_DIM, np.float32)
        self.policy = Scan80000Policy(checkpoint)
        self.reset()
        self.scanner = TrainingHeightMap(self.model, self.data, sensors["height_scanner"], agent["heightmap_resolution"])
        terms = env["terminations"]
        self.height_limit = terms["base_pos_too_far"]["params"]["distance_threshold"]
        self.gravity_limit = terms["base_pg_too_far"]["params"]["projected_gravity_threshold"]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.frame = 0
        self.last_action.fill(0)
        self.data.qpos[:3] = self.ref.pos[0]
        self.data.qpos[3:7] = self.ref.quat[0]
        self.data.qpos[self.q] = self.ref.joints[0]
        mujoco.mj_kinematics(self.model, self.data)
        self.data.qpos[2] += self.lift + max(0.0, -self.data.xpos[self.links, 2].min())
        # At frame zero the training central difference repeats the first frame on the left.
        self.data.qvel[:3] = (self.ref.pos[1] - self.ref.pos[0]) / (2 * self.dt)
        mujoco.mju_subQuat(self.data.qvel[3:6], self.ref.quat[1], self.ref.quat[0])
        self.data.qvel[3:6] /= 2 * self.dt
        self.data.qvel[self.v] = self.ref.vel[0]
        self.data.ctrl[self.ctrl] = self.ref.joints[0]
        mujoco.mj_forward(self.model, self.data)
        self.history.clear()
        self.history.extend([self.proprio()] * self.history.maxlen)

    def proprio(self):
        return dict(projected_gravity=-quat_to_mat(self.data.qpos[3:7])[2],
                    base_ang_vel=self.data.qvel[3:6].copy(),
                    joint_pos=self.data.qpos[self.q] - self.default_q,
                    joint_vel=self.data.qvel[self.v], last_action=self.last_action.copy())

    def observation(self):
        rotation = quat_to_mat(self.data.qpos[3:7])
        goal = build_goal(self.ref.pos, self.ref.quat, self.ref.joints, rotation, self.frame,
                          self.horizon, self.stride, default_q=self.default_q, ref_joint_vel=self.ref.vel)
        scan = self.scanner.scan(self.data.qpos[:3], rotation)
        return np.concatenate((goal, scan, build_proprio(self.history)))

    def step(self, observation=None):
        if self.frame >= len(self.ref.pos) - 1:
            raise StopIteration("End of reference motion")
        if observation is None:
            observation = self.observation()
        action = self.policy.infer(observation)
        if not np.isfinite(action).all():
            raise FloatingPointError(f"Nonfinite action at reference frame {self.frame}")
        self.last_action = action
        self.data.ctrl[self.ctrl] = self.default_q + self.scale * action
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.frame += 1
        if (self.data.warning.number.any() or not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all() or not np.isclose(self.data.time, self.frame * self.dt)):
            raise FloatingPointError(f"MuJoCo state or warning at frame {self.frame}: {self.data.warning.number}")
        self.history.append(self.proprio())

    def errors(self):
        delta = self.data.qpos[:3] - self.ref.pos[self.frame]
        joint_mae = np.abs(self.data.qpos[self.q] - self.ref.joints[self.frame]).mean()
        gravity = np.linalg.norm(quat_to_mat(self.ref.quat[self.frame])[2] - quat_to_mat(self.data.qpos[3:7])[2])
        return np.linalg.norm(delta), joint_mae, abs(delta[2]), gravity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION, help="50 Hz reference NPZ with tile boxes")
    parser.add_argument("--checkpoint", type=Path, default=POLICY / "model_80000.pt")
    parser.add_argument("--seconds", type=float, help="limit playback duration (default: whole clip)")
    parser.add_argument("--check", "--headless", action="store_true", help="run headless and report tracking failures")
    heightmap = parser.add_mutually_exclusive_group()
    heightmap.add_argument("--show-heightmap", "--heightmap-debug", action="store_true", default=None,
                           help="show policy height markers and map (default in the viewer)")
    heightmap.add_argument("--no-heightmap", dest="show_heightmap", action="store_false",
                           help="hide the height markers and map")
    parser.add_argument("--show_ref", "--show-ref", action="store_true",
                        help="overlay the current reference pose as a translucent pink G1")
    args = parser.parse_args()
    if args.check and (args.show_heightmap or args.show_ref):
        parser.error("Showing an overlay requires a viewer; remove --check/--headless")
    if args.show_heightmap is None:
        args.show_heightmap = not args.check
    rt = ScanReplay(args.motion, args.checkpoint)
    steps = len(rt.ref.pos) - 1
    if args.seconds is not None:
        if not np.isfinite(args.seconds) or args.seconds < rt.dt:
            parser.error(f"--seconds must be finite and at least {rt.dt:g}")
        steps = min(steps, int(np.floor(args.seconds / rt.dt + 1e-9)))

    print(f"{args.motion.name} | checkpoint iter {rt.policy.iteration} | {1/rt.dt:g} Hz | {steps} steps", flush=True)
    reference = None
    viewer_context = nullcontext()
    if not args.check:
        import mujoco.viewer as mj_viewer
        from sim2sim.scan80000_view import ReferenceGhost, draw_heightmap
        if args.show_ref:
            reference = ReferenceGhost(rt.model, rt.q)
        viewer_context = mj_viewer.launch_passive(rt.model, rt.data)
    sums = np.zeros(2)
    first_failure = None
    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -25 if args.show_heightmap else -14
        start = time.monotonic()
        for _ in range(steps):
            if viewer is not None and not viewer.is_running():
                break
            observation = rt.observation()
            if viewer is not None:
                if args.show_heightmap:
                    draw_heightmap(viewer, rt.scanner, rt.data.qpos[:7], observation)
                with viewer.lock():
                    if reference is not None:
                        if not args.show_heightmap:
                            viewer.user_scn.ngeom = 0
                        reference.append(viewer.user_scn, rt.ref, rt.frame)
                    viewer.cam.lookat[:] = rt.data.qpos[:3]
                viewer.sync()
            rt.step(observation)
            pos, joint, height, gravity = rt.errors()
            sums += (pos, joint)
            if first_failure is None and (height > rt.height_limit or gravity > rt.gravity_limit):
                first_failure = rt.frame
                print(f"Tracking limit at {rt.data.time:.2f}s: height error {height:.3f} m, "
                      f"projected gravity error {gravity:.3f}", flush=True)
            if viewer is not None:
                time.sleep(max(0.0, start + rt.data.time - time.monotonic()))
    if rt.frame:
        print(f"{rt.data.time:.2f}s | mean root error {sums[0]/rt.frame:.4f} m | "
              f"mean joint error {sums[1]/rt.frame:.4f} rad | final root z {rt.data.qpos[2]:.3f} m")
    return int(args.check and first_failure is not None)


if __name__ == "__main__":
    sys.exit(main())
