"""Run paired S1→S0 terrain perturbations or evaluate the test split on its original layouts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "sim2sim/supermimic_sim2sim.py"
HEIGHTS = {"step": [0.20, 0.30, 0.40, 0.55], "climb": [0.55, 0.70, 0.82, 0.95]}


@dataclass
class Case:
    scene: str
    variant: str
    height: float | None = None
    offset: float = 0.
    motion: Path | None = None
    goal: tuple[float, float, float] = (6., 0., 0.)


def crossing_goal(motion):
    """Aim through the mean box center to ground 1.2 m past the far edge, using only frame zero and terrain."""
    with np.load(motion, allow_pickle=False) as clip:
        start = clip["base_pos_w"][0, :2].astype(float)
        w, x, y, z = clip["base_quat_w"][0].astype(float)
        boxes = clip["boxes"][1:]
    if not len(boxes):
        raise ValueError(f"No obstacle in {motion}")
    toward = boxes[:, :2].mean(0) - start
    if np.linalg.norm(toward) < 1e-6:
        raise ValueError(f"No approach direction from start pose in {motion}")
    toward /= np.linalg.norm(toward)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    bw, bx, by, bz = boxes[:, 3:7].T
    angles = np.arctan2(2 * (bw * bz + bx * by), 1 - 2 * (by * by + bz * bz))
    axes = np.column_stack((np.cos(angles), np.sin(angles)))
    sides = np.column_stack((-axes[:, 1], axes[:, 0]))
    radius = (np.abs(axes @ toward) * boxes[:, 7] + np.abs(sides @ toward) * boxes[:, 8]) / 2
    length = np.max((boxes[:, :2] - start) @ toward + radius) + 1.2
    heading = np.arctan2(toward[1], toward[0]) - yaw
    return (float(length * np.cos(heading)), float(length * np.sin(heading)),
            float(np.degrees(np.arctan2(np.sin(heading), np.cos(heading)))))


def run_trial(case, seed, seconds, path, no_early_stop=False):
    scene_args = ["--planner", "--motion", str(case.motion)] if case.motion else ["--terrain-demo", case.scene]
    command = [sys.executable, str(RUNNER), *scene_args, "--seed", str(seed),
               "--goal", *map(str, case.goal), "--seconds", str(seconds), "--check", "--metrics-json", str(path)]
    if no_early_stop:
        command.append("--no-early-stop")
    if case.height is not None:
        command.extend(["--box-height", str(case.height)])
    if case.offset:
        command.extend(["--box-offset-y", str(case.offset)])
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # --check returns 1 when a completed trial misses its goal.
    if result.returncode not in (0, 1) or not path.is_file():
        raise RuntimeError(f"Trial {case.scene}/{case.variant}/seed {seed} failed to produce metrics:\n{result.stdout}")
    row = json.loads(path.read_text())
    if result.returncode != int(not row["goal_reached"]):
        raise RuntimeError(f"Unexpected exit status {result.returncode}: {result.stdout}")
    return dict(row, scene=case.scene, variant=case.variant, goal_forward_left_yaw=case.goal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scenes", nargs="+", choices=HEIGHTS)
    mode.add_argument("--heldout-dataset", type=Path, help="evaluate original step/climb layouts in index.json split=test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--heights", type=float, nargs="+", help="absolute test heights in metres for each selected scene")
    parser.add_argument("--offsets", type=float, nargs="+", help="world Y offsets at original height (default: -0.25 0.25)")
    parser.add_argument("--seconds", type=float, default=16)
    parser.add_argument("--no-early-stop", action="store_true",
                        help="run each trial to its duration after tracking or terrain coverage failures")
    parser.add_argument("--workers", type=int, default=1, help="concurrent CPU trials")
    parser.add_argument("--output", type=Path, help="JSON report path (default under sim2sim_output/supermimic/)")
    args = parser.parse_args()
    if args.workers < 1 or not math.isfinite(args.seconds) or args.seconds < .02:
        parser.error("Expected a positive worker count and a duration of at least 0.02 s")
    if any(not math.isfinite(h) or h < 0 for h in args.heights or []) or any(not math.isfinite(y) for y in args.offsets or []):
        parser.error("Heights must be finite and nonnegative; offsets must be finite")
    seeds = list(dict.fromkeys(args.seeds))
    conditions = []
    if args.heldout_dataset:
        if args.heights or args.offsets:
            parser.error("Use --scenes for height and offset sweeps")
        dataset = args.heldout_dataset.resolve()
        for name, entry in sorted(json.loads((dataset / "index.json").read_text()).items()):
            if entry["split"] == "test" and entry["label"] in HEIGHTS:
                path = dataset / entry["file"]
                conditions.append(Case(entry["label"], name, motion=path, goal=crossing_goal(path)))
        if not conditions:
            parser.error("No step/climb layouts in the dataset test split")
    else:
        for scene in dict.fromkeys(args.scenes or HEIGHTS):
            conditions.append(Case(scene, "original"))
            conditions.extend(Case(scene, f"height_{h}", height=h) for h in dict.fromkeys(args.heights or HEIGHTS[scene]))
            conditions.extend(Case(scene, f"offset_{y}", offset=y) for y in dict.fromkeys(args.offsets or [-.25, .25]) if y)

    jobs = [(condition, seed) for condition in conditions for seed in seeds]
    rows = [None] * len(jobs)
    if args.output is None:
        name = "terrain_heldout.json" if args.heldout_dataset else "terrain_sweep.json"
        args.output = ROOT / "sim2sim_output/supermimic" / name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_trial, *job, args.seconds, Path(tmp).resolve() / f"{i}.json", args.no_early_stop): i
                   for i, job in enumerate(jobs)}
        for count, future in enumerate(as_completed(futures), start=1):
            row = rows[futures[future]] = future.result()
            print(f"[{count}/{len(rows)}] {row['scene']} {row['variant']} seed {row['seed']} | crossed {row['crossed']} | "
                  f"goal {row['goal_reached']} | XY {row['final_xy_error_m']:.3f} m | {row['stop']}", flush=True)
        summary = []
        for case in conditions:
            group = [r for r in rows if r["scene"] == case.scene and r["variant"] == case.variant]
            summary.append(dict(scene=case.scene, variant=case.variant, height_m=group[0]["box_height_m"],
                                offset_y_m=group[0]["box_offset_y_m"], trials=len(group),
                                crossed=sum(r["crossed"] is True for r in group) if group[0]["crossed"] is not None else None,
                                goal_reached=sum(r["goal_reached"] for r in group), task_success=sum(r["task_success"] for r in group),
                                median_final_xy_error_m=median(r["final_xy_error_m"] for r in group),
                                trials_with_violation=sum(bool(r["first_violation_s"]) for r in group),
                                stopped_early=sum(r["stop"] != "duration" for r in group)))
        checkpoints = ["policy/scan80000/model_80000.pt", "policy/s1_flow_w256/ckpt_045000.pt"]
        report = dict(seeds=seeds, requested_seconds=args.seconds, no_early_stop=args.no_early_stop,
                      goal_protocol="through obstacle center, 1.2 m beyond the far edge" if args.heldout_dataset else "6 m forward, final yaw 0",
                      checkpoint_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in checkpoints},
                      scoring={"crossed": "Both feet on top, then both supported on floor beyond the far edge; null on flat control.",
                               "goal_reached": "Full duration without tracking/map failure, final XY < 0.30 m and yaw error < 20 deg.",
                               "task_success": "Goal reached and obstacle crossed (goal alone on flat control)."},
                      summary=summary, trials=rows)
        finished = Path(tmp) / "report.json"
        finished.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        finished.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
