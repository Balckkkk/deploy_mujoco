# G1 perceptive sim2sim: phase 3 acceptance

Phase 3 adds the ground and the training vault/climb obstacles. It does not yet
run the learned policy.

## Training geometry

| Object | Full size (x, y, z), m | Nominal center (x, y, z), m | Exposed top, m |
|---|---:|---:|---:|
| Vault | `(0.400000, 10.0, 0.600000)` | `(1.700000, 0.5, 0.055000)` | `0.355000` |
| Climb | `(0.355955, 10.0, 0.704494)` | `(5.229532, 0.0, 0.352247)` | `0.704494` |

The vault center and size come directly from the motion NPZ and the target
run's resolved `env.yaml`. Its bottom is below the ground plane; this is
intentional and is not corrected in sim2sim.

The ground uses sliding friction 1.0. Obstacles use sliding friction 0.5 and
are represented as MuJoCo mocap bodies, matching Isaac's kinematic rigid
objects. They remain collidable and visible to the robot depth camera while
supporting reset-time pose updates without recompilation.

## Modes

- `vault`: training-style vault environment; climb box is parked at its hidden pose.
- `climb`: training-style climb environment; vault box is parked at its hidden pose.
- `both`: diagnostic view of both source-task obstacles at their nominal positions.
- `sequential`: diagnostic/intended sequential layout with climb moved to x=7 m.

With `--randomize`, each active obstacle uses the target training ranges:

- x and y translation: `[-0.05, 0.05]` m
- z, roll, pitch: fixed at zero offset
- yaw: `[-45°, 45°]`

The seed makes reset sampling repeatable. Orange identifies vault and cyan
identifies climb in debug rendering; the training materials are otherwise
uniform gray and do not affect depth.

## Commands

Interactive scene viewer:

```bash
cd /home/black/Code/deploy_mujoco
/home/black/anaconda3/envs/deploy_mujoco/bin/python \
  tools/view_g1_obstacle_scene.py --mode both
```

Training-style randomized vault scene:

```bash
cd /home/black/Code/deploy_mujoco
/home/black/anaconda3/envs/deploy_mujoco/bin/python \
  tools/view_g1_obstacle_scene.py --mode vault --randomize --seed 42
```

Headless geometry/depth report and screenshot export:

```bash
cd /home/black/Code/deploy_mujoco
MUJOCO_GL=egl /home/black/anaconda3/envs/deploy_mujoco/bin/python \
  tools/view_g1_obstacle_scene.py --headless --mode both
```

Regression tests:

```bash
cd /home/black/Code/deploy_mujoco
MUJOCO_GL=egl /home/black/anaconda3/envs/deploy_mujoco/bin/python -m unittest -v \
  tests/test_g1_obstacle_scene.py \
  tests/test_depth_camera.py \
  tests/test_g1_perceptive_model.py
```

Headless artifacts are written to `obstacle_scene_debug/`: overview RGB,
robot depth image/tensor, and a JSON geometry report.
