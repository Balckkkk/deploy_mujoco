"""Early Scan80000 replay using the perceptive G1 model and a uniform action scale.

Use sim2sim/supermimic_sim2sim.py to replay with the frozen training configuration.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from scan80000_policy import (JOINT_NAMES, ACTION_DIM, GOAL_DIM, SCAN_DIM, PROPRIO_DIM, DEFAULT_Q,
                              Scan80000Policy, build_goal, build_proprio, quat_to_mat)
from scan80000_heightmap import Scan187HeightMap, Scan187Config

DEFAULT_MODEL = Path("assets/g1/mjcf/g1_29dof_perceptive.xml")
DEFAULT_REFERENCE = Path("config/g1_perceptive_isaac_reference.json")
ACTION_SCALE = 0.5
FPS = 50
CONTROL_DT = 1.0 / FPS
SIM_DT = 0.005
DECIMATION = 4
HISTORY = 8
GOAL_HORIZON, GOAL_STRIDE = 10, 5


def setup_joint_mappings(model, joint_names):
    qpos, qvel, act = [], [], []
    for name in joint_names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if j < 0 or a < 0:
            raise ValueError(f"missing joint/actuator {name}")
        qpos.append(int(model.jnt_qposadr[j]))
        qvel.append(int(model.jnt_dofadr[j]))
        act.append(int(a))
    return (np.asarray(qpos, np.int32), np.asarray(qvel, np.int32), np.asarray(act, np.int32))


def load_pd(reference_path, joint_names):
    """Return the fixed joint gains used by the early replay."""
    import re
    groups = [
        (r".*_hip_(yaw|roll|pitch)_joint", 90.0, 2.0, 88.0),
        (r".*_knee_joint", 140.0, 2.5, 139.0),
        (r"waist_(roll|pitch)_joint", 60.0, 2.5, 50.0),
        (r"waist_yaw_joint", 90.0, 2.5, 88.0),
        (r".*_ankle_(pitch|roll)_joint", 20.0, 1.0, 20.0),
        (r".*_shoulder_(pitch|roll|yaw)_joint|.*_elbow_joint", 25.0, 1.0, 25.0),
        (r".*_wrist_roll_joint", 25.0, 1.0, 25.0),
        (r".*_wrist_(pitch|yaw)_joint", 5.0, 0.5, 5.0),
    ]
    kp, kd, eff = [], [], []
    for name in joint_names:
        for pat, k, d, e in groups:
            if re.fullmatch(pat, name):
                kp.append(k), kd.append(d), eff.append(e)
                break
        else:
            raise ValueError(f"no PD group for joint {name}")
    return (np.asarray(kp, np.float32), np.asarray(kd, np.float32), np.asarray(eff, np.float32))


def projected_gravity(q):
    w, x, y, z = np.asarray(q, np.float64)
    return np.array((2 * (w * y - z * x), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)), np.float32)


def base_angular_velocity(model, data, body_id):
    v = np.zeros(6, np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, v, 1)
    return v[:3].astype(np.float32)


class ScanReplay:
    def __init__(self, args):
        spec = mujoco.MjSpec.from_file(str(DEFAULT_MODEL))
        spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(50.0, 50.0, 0.1),
                                friction=(1.0, 0.005, 0.0001), rgba=(0.32, 0.34, 0.37, 1.0), group=0, condim=3)
        # reference motion (tile coordinates, centered), read before building boxes
        with np.load(args.motion, allow_pickle=False) as d:
            names = [str(n) for n in d["joint_names"]]
            self.ref_pos = np.asarray(d["base_pos_w"], np.float64)
            self.ref_quat = np.asarray(d["base_quat_w"], np.float64)
            self.ref_joint = np.asarray(d["joint_pos"], np.float64)[:, [names.index(n) for n in JOINT_NAMES]]
            self.boxes = np.asarray(d["boxes"]) if "boxes" in d else None
        self._translate_ref_to_origin()
        self._add_boxes(spec)
        self.model = spec.compile()
        self.model.opt.timestep = SIM_DT
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        self.qpos_idx, self.qvel_idx, self.act_idx = setup_joint_mappings(self.model, JOINT_NAMES)
        # Fixed contact and joint dynamics used by this prototype.
        self.model.geom_condim[:] = 3
        self.model.dof_damping[self.qvel_idx] = 0.5
        self.model.dof_frictionloss[self.qvel_idx] = 0.1
        self.model.dof_armature[self.qvel_idx] = 0.03
        self.kp, self.kd, self.effort = load_pd(DEFAULT_REFERENCE, JOINT_NAMES)
        self.base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.scanner = Scan187HeightMap(self.model, self.data, Scan187Config())
        self.policy = Scan80000Policy(args.checkpoint)
        self.frame = 0
        self.last_action = np.zeros(ACTION_DIM, np.float32)
        self.target = np.zeros(ACTION_DIM, np.float32)
        self.history = deque(maxlen=HISTORY)

    def _translate_ref_to_origin(self):
        shift = self.ref_pos[0].copy()
        shift[2] = 0.0
        self.ref_pos = self.ref_pos - shift
        if self.boxes is not None:
            self.boxes = self.boxes.copy()
            self.boxes[:, :2] -= shift[:2]

    def _add_boxes(self, spec):
        if self.boxes is None or len(self.boxes) == 0:
            return
        for i, b in enumerate(self.boxes):
            if i == 0 and b[7] > 20.0:  # the ground slab itself
                continue
            body = spec.worldbody.add_body(name=f"box_{i}", pos=b[:3].tolist(), quat=b[3:7].tolist(), mocap=True)
            body.add_geom(name=f"box_{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=(b[7:] / 2).tolist(),
                          friction=(0.5, 0.005, 0.0001), rgba=(0.7, 0.55, 0.3, 1.0), group=0, condim=3)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.ref_pos[0]
        self.data.qpos[2] += 0.1  # reset lift, as in the Warp reset (settles back down)
        self.data.qpos[3:7] = self.ref_quat[0]
        self.data.qpos[self.qpos_idx] = self.ref_joint[0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.frame = 0
        self.last_action = np.zeros(ACTION_DIM, np.float32)
        self.target = self.ref_joint[0].copy()
        self.history.clear()
        for _ in range(HISTORY):
            self._append_history()

    def _append_history(self):
        q = self.data.qpos[self.qpos_idx].astype(np.float32)
        self.history.append(dict(
            projected_gravity=projected_gravity(self.data.qpos[3:7]),
            base_ang_vel=base_angular_velocity(self.model, self.data, self.base_body),
            joint_pos=q - DEFAULT_Q,
            joint_vel=self.data.qvel[self.qvel_idx].astype(np.float32),
            last_action=self.last_action.copy(),
        ))

    def policy_step(self):
        robot_R = quat_to_mat(self.data.qpos[3:7])
        goal = build_goal(self.ref_pos, self.ref_quat, self.ref_joint, robot_R, self.frame,
                          GOAL_HORIZON, GOAL_STRIDE)
        scan = self.scanner.scan()
        proprio = build_proprio(list(self.history))
        obs = np.concatenate((goal, scan, proprio))
        assert obs.shape == (GOAL_DIM + SCAN_DIM + PROPRIO_DIM,), obs.shape
        action = self.policy.infer(obs)
        self.last_action = action
        self.target = DEFAULT_Q + ACTION_SCALE * action
        return action

    def physics_step(self):
        q = self.data.qpos[self.qpos_idx]
        qd = self.data.qvel[self.qvel_idx]
        torque = np.clip(self.kp * (self.target - q) - self.kd * qd, -self.effort, self.effort)
        self.data.ctrl[self.act_idx] = torque
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        return torque

    def step(self):
        self.policy_step()
        self.physics_step()
        self.frame += 1
        self._append_history()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("policy/scan80000/model_80000.pt"))
    ap.add_argument("--motion", type=Path, required=True, help="planner-dataset npz motion to replay")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--check", action="store_true", help="headless smoke run that prints tracking error")
    args = ap.parse_args()
    rt = ScanReplay(args)
    rt.reset()
    base_z0 = rt.data.qpos[2]
    print(f"motion frames={len(rt.ref_pos)} base_z0={base_z0:.3f}", flush=True)
    if args.check:
        errs = []
        for _ in range(int(args.seconds * FPS)):
            rt.step()
            e_pos = np.linalg.norm(rt.data.qpos[:3] - rt.ref_pos[min(rt.frame, len(rt.ref_pos) - 1)])
            e_joint = np.abs(rt.data.qpos[rt.qpos_idx] - rt.ref_joint[min(rt.frame, len(rt.ref_joint) - 1)]).mean()
            errs.append((e_pos, e_joint))
            if rt.frame % 50 == 0:
                print(f"t={rt.frame/FPS:.1f}s base_z={rt.data.qpos[2]:.3f} pos_err={e_pos:.3f} joint_err={e_joint:.3f}",
                      flush=True)
        ep, ej = np.array(errs).mean(0)
        print(f"DONE mean pos_err={ep:.3f} joint_err={ej:.3f} final_base_z={rt.data.qpos[2]:.3f}")
        return
    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(rt.model, rt.data) as viewer:
        while viewer.is_running():
            rt.step()
            viewer.sync()
            if rt.frame >= len(rt.ref_pos) - GOAL_HORIZON * GOAL_STRIDE:
                rt.frame = 0


if __name__ == "__main__":
    main()
