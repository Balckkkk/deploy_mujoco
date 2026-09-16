#!/usr/bin/env python3
"""Print the resolved Isaac Lab articulation ordering for the perceptive G1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, help="Write the resolved articulation data as JSON.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym

import instinctlab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


TASK_NAME = "Instinct-Perceptive-VaultClimb-VaeDistill-G1-Play-v0"


def main() -> None:
    cfg = parse_env_cfg(TASK_NAME, device="cpu", num_envs=1, use_fabric=False)
    env = gym.make(TASK_NAME, cfg=cfg)
    try:
        robot = env.unwrapped.scene["robot"]
        print("JOINT_NAMES_BEGIN", flush=True)
        for index, name in enumerate(robot.joint_names):
            print(index, name, flush=True)
        print("JOINT_NAMES_END", flush=True)
        print("BODY_NAMES_BEGIN", flush=True)
        for index, name in enumerate(robot.body_names):
            print(index, name, flush=True)
        print("BODY_NAMES_END", flush=True)

        data = robot.data
        resolved = {
            "task": TASK_NAME,
            "joint_names": robot.joint_names,
            "body_names": robot.body_names,
            "body": {
                "mass": data.default_mass[0].cpu().tolist(),
                "inertia": data.default_inertia[0].cpu().tolist(),
                "com_pose_wxyz": data.body_com_pose_b[0].cpu().tolist(),
            },
            "joint": {
                "default_position": data.default_joint_pos[0].cpu().tolist(),
                "position_limit": data.default_joint_pos_limits[0].cpu().tolist(),
                "velocity_limit": data.joint_vel_limits[0].cpu().tolist(),
                "effort_limit": data.joint_effort_limits[0].cpu().tolist(),
                "stiffness": data.default_joint_stiffness[0].cpu().tolist(),
                "damping": data.default_joint_damping[0].cpu().tolist(),
                "armature": data.default_joint_armature[0].cpu().tolist(),
                "friction": data.default_joint_friction_coeff[0].cpu().tolist(),
            },
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(resolved, indent=2) + "\n")
            print(f"WROTE {args.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
