"""Run the SuperMimic S0 tracker in MuJoCo with a recorded motion or an online S1 planner."""

from __future__ import annotations

import argparse
import json
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
from sim2sim.scan80000_terrain import TerrainContacts, TrainingHeightMap, add_terrain, peak_box_height, vary_boxes

ASSET = ROOT / "asset/sp_g1"
POLICY = ROOT / "policy/scan80000"
DEFAULT_PLANNER_CHECKPOINT = ROOT / "policy/s1_flow_w256/ckpt_045000.pt"
DEFAULT_MOTION = POLICY / "motions/low_step.npz"
TOP_SCENES = ROOT / "policy/s1_r3_123_w256/scenes"
TERRAIN_DEMOS = {"step": DEFAULT_MOTION, "climb": POLICY / "motions/climb_76.npz",
                 "step-top": TOP_SCENES / "step_top.npz", "climb-top": TOP_SCENES / "climb_top.npz"}


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

    start_frame = 0

    def __init__(self, path, fps, *, initial_pose_only=False):
        with np.load(path, allow_pickle=False) as clip:
            rate = float(clip["framerate"].item())
            if not np.isclose(rate, fps):
                raise ValueError(f"{path} is {rate:g} Hz; expected a {fps:g} Hz planner-dataset export")
            frames = slice(1) if initial_pose_only else slice(None)
            self.pos = np.asarray(clip["base_pos_w"][frames], dtype=np.float64)
            self.quat = np.asarray(clip["base_quat_w"][frames], dtype=np.float64)
            names = clip["joint_names"].astype(str).tolist()
            if len(names) != len(set(names)) or not set(JOINT_NAMES).issubset(names):
                raise ValueError(f"{path} has duplicate or missing joint names")
            self.joints = np.asarray(clip["joint_pos"][frames], dtype=np.float64)[:, [names.index(n) for n in JOINT_NAMES]]
            self.boxes = np.asarray(clip["boxes"], dtype=np.float64) if "boxes" in clip else np.empty((0, 10))
        n = len(self.pos)
        min_frames = 1 if initial_pose_only else 2
        if n < min_frames or self.pos.shape != (n, 3) or self.quat.shape != (n, 4) or self.joints.shape != (n, ACTION_DIM):
            raise ValueError(f"Inconsistent frame arrays in {path}")
        if self.boxes.ndim != 2 or self.boxes.shape[1] != 10 or np.any(self.boxes[:, 7:] <= 0):
            raise ValueError(f"Expected box centers, wxyz rotations and positive full sizes in {path}")
        if not all(np.isfinite(x).all() for x in (self.pos, self.quat, self.joints, self.boxes)):
            raise ValueError(f"Nonfinite motion or terrain in {path}")
        for rot in (self.quat, self.boxes[:, 3:7]):
            if not np.allclose(np.linalg.norm(rot, axis=1), 1, atol=1e-3):
                raise ValueError(f"Expected unit wxyz quaternions in {path}")
        self.quat /= np.linalg.norm(self.quat, axis=1, keepdims=True)
        self.vel = np.zeros_like(self.joints) if initial_pose_only else frontbackward_velocity(self.joints, 1 / fps)


class ScanReplay:
    def __init__(self, motion=DEFAULT_MOTION, checkpoint=POLICY / "model_80000.pt", *, initial_pose_only=False,
                 box_height=None, box_offset_y=0.0):
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
        self.motion = ReferenceMotion(motion, fps, initial_pose_only=initial_pose_only)
        self.motion.boxes = vary_boxes(self.motion.boxes, box_height, box_offset_y)
        spec = mujoco.MjSpec.from_file(str(ASSET / "xml" / Path(ref["robot_model_path"]).name))
        configure_robot(spec, robot)
        configure_physics(spec.option, env["sim"]["mujoco"])
        add_terrain(spec, self.motion.boxes)
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
        self.ref = self.motion
        self.last_action.fill(0)
        self.data.qpos[:3] = self.ref.pos[0]
        self.data.qpos[3:7] = self.ref.quat[0]
        self.data.qpos[self.q] = self.ref.joints[0]
        mujoco.mj_kinematics(self.model, self.data)
        self.data.qpos[2] += self.lift + max(0.0, -self.data.xpos[self.links, 2].min())
        # At frame zero the training central difference repeats the first frame on the left.
        if len(self.ref.pos) > 1:
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

    @property
    def reference_frame(self):
        return self.frame - self.ref.start_frame

    def observation(self):
        rotation = quat_to_mat(self.data.qpos[3:7])
        goal = build_goal(self.ref.pos, self.ref.quat, self.ref.joints, rotation, self.reference_frame,
                          self.horizon, self.stride, default_q=self.default_q, ref_joint_vel=self.ref.vel)
        scan = self.scanner.scan(self.data.qpos[:3], rotation)
        return np.concatenate((goal, scan, build_proprio(self.history)))

    def step(self, observation=None, *, after_substep=None):
        if self.reference_frame >= len(self.ref.pos) - 1:
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
            if after_substep is not None:
                after_substep(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.frame += 1
        if (self.data.warning.number.any() or not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all() or not np.isclose(self.data.time, self.frame * self.dt)):
            raise FloatingPointError(f"MuJoCo state or warning at frame {self.frame}: {self.data.warning.number}")
        self.history.append(self.proprio())

    def errors(self):
        i = self.reference_frame
        delta = self.data.qpos[:3] - self.ref.pos[i]
        joint_mae = np.abs(self.data.qpos[self.q] - self.ref.joints[i]).mean()
        gravity = np.linalg.norm(quat_to_mat(self.ref.quat[i])[2] - quat_to_mat(self.data.qpos[3:7])[2])
        return np.linalg.norm(delta), joint_mae, abs(delta[2]), gravity


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    scene = parser.add_mutually_exclusive_group()
    scene.add_argument("--motion", type=Path, help="50 Hz reference NPZ (planner mode uses its start pose and terrain)")
    scene.add_argument("--terrain-demo", choices=TERRAIN_DEMOS,
                       help="step/climb aim 6 m ahead; step-top/climb-top aim at a measured platform stop (16 s)")
    parser.add_argument("--box-height", type=float, metavar="M", help="peak obstacle height above floor in metres; 0 removes obstacles")
    parser.add_argument("--box-offset-y", type=float, default=0.0, metavar="M", help="move obstacles along world Y in metres")
    parser.add_argument("--metrics-json", type=Path, help="write planner contact, tracking and final target metrics")
    parser.add_argument("--gif", type=Path, help="export the planner and robot with foot contacts to an animated GIF")
    parser.add_argument("--checkpoint", type=Path, default=POLICY / "model_80000.pt")
    parser.add_argument("--planner", action="store_true", help="drive S0 with the online SuperMimic S1 planner")
    parser.add_argument("--planner-checkpoint", type=Path, help="S1 checkpoint (default: flow_w256 step 45000)")
    parser.add_argument("--planner-reanchor", choices=("xy", "xy-yaw", "world"),
                        help="align the retained reference in XY (default), include yaw, or keep its world pose")
    parser.add_argument("--goal", type=float, nargs=3, metavar=("FORWARD_M", "LEFT_M", "YAW_DEG"),
                        help="planner target in the starting yaw frame (default: 2 0 0)")
    parser.add_argument("--seed", type=int, help="planner sampling seed (default: 0)")
    parser.add_argument("--seconds", type=float, help="duration (default: whole clip for replay, 12 s for planner, 16 s for terrain demos)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="continue online simulation after tracking or terrain coverage failures; "
                             "without --seconds the live viewer runs until closed")
    parser.add_argument("--check", "--headless", action="store_true", help="run headless and report tracking failures")
    heightmap = parser.add_mutually_exclusive_group()
    heightmap.add_argument("--show-heightmap", "--heightmap-debug", action="store_true", default=None,
                           help="show policy height markers and map (default in the viewer)")
    heightmap.add_argument("--no-heightmap", dest="show_heightmap", action="store_false",
                           help="hide the height markers and map")
    parser.add_argument("--show_ref", "--show-ref", action="store_true",
                        help="overlay the current reference pose as a translucent pink G1")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.terrain_demo:
        args.planner = True
        args.motion = TERRAIN_DEMOS[args.terrain_demo]
        if args.goal is None:
            with np.load(args.motion, allow_pickle=False) as scene:
                args.goal = tuple(scene["goal"]) if "goal" in scene else (6, 0, 0)
        args.show_ref = args.show_ref or not args.check
    if args.no_early_stop and not args.planner:
        parser.error("--no-early-stop requires --planner or --terrain-demo")
    if args.gif and (not args.planner or args.gif.suffix.lower() != ".gif"):
        parser.error("--gif requires planner mode (--terrain-demo or --planner) and a .gif output path")
    if args.check and not args.gif and (args.show_heightmap or args.show_ref):
        parser.error("Showing an overlay requires a viewer; remove --check/--headless")
    if args.show_heightmap is None:
        args.show_heightmap = bool(args.gif) or not args.check
    if not args.planner and any(x is not None for x in (args.planner_checkpoint, args.planner_reanchor, args.goal, args.seed)):
        parser.error("--goal, --seed, --planner-checkpoint and --planner-reanchor require --planner or --terrain-demo")
    if args.metrics_json and not args.planner:
        parser.error("--metrics-json requires planner mode")
    if not np.isfinite(args.box_offset_y) or (args.box_height is not None and
            (not np.isfinite(args.box_height) or args.box_height < 0)):
        parser.error("Expected a finite, nonnegative --box-height and a finite --box-offset-y")
    motion = args.motion or (POLICY / "motions/walk2_subject3.npz" if args.planner else DEFAULT_MOTION)
    rt = ScanReplay(motion, args.checkpoint, initial_pose_only=args.planner,
                    box_height=args.box_height, box_offset_y=args.box_offset_y)
    planner = None
    if args.planner:
        try:
            from sim2sim.supermimic_planner import OnlinePlanner, TerrainCoverageError
        except ModuleNotFoundError as exc:
            if exc.name == "perspmimic":
                parser.error("Add the SuperMimic checkout to PYTHONPATH to use --planner")
            raise
        planner = OnlinePlanner(rt, args.planner_checkpoint or DEFAULT_PLANNER_CHECKPOINT,
                                args.goal or (2, 0, 0), args.seed or 0, reanchor=args.planner_reanchor)
        if args.seconds is None and (not args.no_early_stop or args.check or args.gif):
            args.seconds = 16.0 if args.terrain_demo else 12.0
    steps = None if planner else len(rt.motion.pos) - 1
    if args.seconds is not None:
        if not np.isfinite(args.seconds) or args.seconds < rt.dt:
            parser.error(f"--seconds must be finite and at least {rt.dt:g}")
        requested = int(np.floor(args.seconds / rt.dt + 1e-9))
        steps = requested if planner else min(steps, requested)

    horizon = "until viewer closes" if steps is None else f"{steps} steps"
    print(f"{motion.name} | S0 iter {rt.policy.iteration} | {1/rt.dt:g} Hz | {horizon}", flush=True)
    if planner:
        print(f"S1 step {planner.checkpoint_step} | 5 Hz | goal xy {planner.target_xy} | seed {args.seed or 0}", flush=True)
    peak = peak_box_height(rt.motion.boxes)
    if args.terrain_demo or args.box_height is not None or args.box_offset_y:
        print(f"Obstacle height {peak:.3f} m | world Y offset {args.box_offset_y:+.3f} m", flush=True)
    contacts = None
    if planner and (args.terrain_demo or args.metrics_json or args.gif):
        contacts = TerrainContacts(rt, planner.target_xy, goal_height=planner.target_ground[2],
                                   goal_error=planner.goal_error)
    reference = None
    recorder = None
    viewer_context = nullcontext()
    if args.gif:
        from sim2sim.supermimic_terrain_gif import TerrainReplayGIF
        recorder = TerrainReplayGIF(rt, planner, args.seed or 0, args.show_heightmap, contacts)
    elif not args.check:
        import mujoco.viewer as mj_viewer
        from sim2sim.scan80000_view import ReferenceGhost, draw_heightmap, draw_planner_goal
        if args.show_ref:
            reference = ReferenceGhost(rt.model, rt.q)
        viewer_context = mj_viewer.launch_passive(rt.model, rt.data)
    sums = np.zeros(2)
    first_violation_s = {}
    terrain_unavailable = False
    stop = "duration"
    closest = (float(np.linalg.norm(rt.data.qpos[:2] - planner.target_xy)), 0.) if planner else None
    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -25 if args.show_heightmap else -14
        start = time.monotonic()
        while steps is None or rt.frame < steps:
            if viewer is not None and not viewer.is_running():
                stop = "viewer_closed"
                break
            if planner:
                try:
                    replanned = planner.update(rt)
                except TerrainCoverageError as exc:
                    first_violation_s.setdefault("terrain_coverage", rt.data.time)
                    if not args.no_early_stop:
                        stop = "terrain_coverage"
                        print(f"Planner stopped at {rt.data.time:.2f}s: {exc}", flush=True)
                        break
                    planner.extend_last_reference(rt)
                    if not terrain_unavailable:
                        print(f"Terrain coverage lost at {rt.data.time:.2f}s: following the remaining reference, "
                              "then holding its last pose until S1 can replan", flush=True)
                    terrain_unavailable = True
                else:
                    if terrain_unavailable and replanned:
                        print(f"S1 planning resumed at {rt.data.time:.2f}s", flush=True)
                        terrain_unavailable = False
            if recorder and rt.frame % recorder.frame_stride == 0:
                recorder.capture()
            observation = rt.observation()
            if viewer is not None:
                if args.show_heightmap:
                    draw_heightmap(viewer, rt.scanner, rt.data.qpos[:7], observation)
                with viewer.lock():
                    if not args.show_heightmap and (reference is not None or planner is not None):
                        viewer.user_scn.ngeom = 0
                    if reference is not None:
                        reference.append(viewer.user_scn, rt.ref, rt.reference_frame)
                    if planner:
                        draw_planner_goal(viewer.user_scn, planner.target_ground, planner.target_yaw)
                    viewer.cam.lookat[:] = rt.data.qpos[:3]
                viewer.sync()
            rt.step(observation, after_substep=contacts.measure_contacts if contacts else None)
            pos, joint, height, gravity = rt.errors()
            sums += (pos, joint)
            if planner:
                closest = min(closest, (float(np.linalg.norm(rt.data.qpos[:2] - planner.target_xy)), rt.data.time))
            if "tracking_limit" not in first_violation_s and (height > rt.height_limit or gravity > rt.gravity_limit):
                first_violation_s["tracking_limit"] = rt.data.time
                print(f"Tracking limit at {rt.data.time:.2f}s: height error {height:.3f} m, "
                      f"projected gravity error {gravity:.3f}", flush=True)
                if planner and not args.no_early_stop:
                    stop = "tracking_limit"
                    break
            if viewer is not None:
                time.sleep(max(0.0, start + rt.data.time - time.monotonic()))
    if recorder:
        recorder.capture()
        recorder.save(args.gif)
        print(f"GIF {args.gif}")
    if contacts:
        times = [f"{t:.2f} s" if t is not None else "not observed" for t in contacts.events]
        print(f"Left on top: {times[0]}, right on top: {times[1]}, both landed beyond: {times[2]}")
        if contacts.goal_on_top:
            print(f"Stable bilateral stance at platform goal: longest {contacts.longest_hold_s:.3f} s, "
                  f"at end {contacts.final_hold_s:.3f} s")
    if rt.frame:
        print(f"{rt.data.time:.2f}s | mean root error {sums[0]/rt.frame:.4f} m | "
              f"mean joint error {sums[1]/rt.frame:.4f} rad | final root z {rt.data.qpos[2]:.3f} m")
    if planner:
        distance, yaw_error = planner.goal_error(rt.data.qpos)
        reached = distance < 0.3 and yaw_error < 20 and rt.frame == steps and not first_violation_s
        print(f"{planner.replans} replans | final goal XY error {distance:.3f} m | yaw error {yaw_error:.1f} deg | "
              f"goal reached: {reached}", flush=True)
        if args.metrics_json:
            left, right, landed = contacts.events
            crossed = landed is not None if contacts.tops else None
            if contacts.goal_on_top:
                task_ok = reached and contacts.final_hold_s >= 1.0
            else:
                task_ok = reached and crossed is not False
            metrics = dict(scene=args.terrain_demo, motion=str(motion), seed=args.seed or 0,
                           planner_reanchor=planner.reanchor,
                           s1_step=planner.checkpoint_step, s0_step=rt.policy.iteration, box_height_m=peak,
                           box_offset_y_m=args.box_offset_y, goal_xy_m=planner.target_xy.tolist(),
                           final_xy_m=rt.data.qpos[:2].tolist(), duration_s=rt.data.time, stop=stop,
                           no_early_stop=args.no_early_stop, first_violation_s=first_violation_s,
                           left_top_s=left, right_top_s=right, landed_beyond_s=landed, crossed=crossed,
                           goal_reached=bool(reached), task_success=bool(task_ok),
                           task="stop_on_top" if contacts.goal_on_top else "cross" if contacts.tops else "flat_goal",
                           final_top_hold_s=contacts.final_hold_s if contacts.goal_on_top else None,
                           longest_top_hold_s=contacts.longest_hold_s if contacts.goal_on_top else None,
                           final_xy_error_m=float(distance), final_yaw_error_deg=float(yaw_error),
                           closest_xy_error_m=closest[0], closest_time_s=closest[1],
                           mean_root_error_m=float(sums[0]/rt.frame) if rt.frame else None,
                           mean_joint_error_rad=float(sums[1]/rt.frame) if rt.frame else None)
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        return int(args.check and not reached)
    return int(args.check and bool(first_violation_s))


if __name__ == "__main__":
    sys.exit(main())
