# G1 perceptive vault/climb sim2sim

## Runtime contract

- Checkpoint: `model_16000.pt` (iteration 16000).
- Physics: MuJoCo at 200 Hz (`dt=0.005 s`).
- Policy and depth camera: 50 Hz (decimation 4).
- Observation order: depth image (1536), velocity command (3), projected gravity (3), torso angular velocity (3), relative joint position (29), joint velocity (29), previous action (29).
- The checkpoint normalizer is applied only to the 96 proprioceptive values. Depth pixels remain in the trained `[0, 1]` range.
- Action semantics: `q_target = q_default + action`; torques use the captured Isaac stiffness/damping and effort limits.
- Reset state and the default per-frame velocity command come from the corresponding training motion NPZ.

The standalone actor in `g1_perceptive_policy.py` loads the original PyTorch checkpoint directly and has no Isaac Lab dependency. `g1_perceptive_sim2sim.py` combines it with the reusable depth-camera and obstacle-scene modules.

## Interactive runs

Run from `/home/black/Code/deploy_mujoco`:

```bash
/home/black/anaconda3/envs/deploy_mujoco/bin/python g1_perceptive_sim2sim.py --mode vault
```

```bash
/home/black/anaconda3/envs/deploy_mujoco/bin/python g1_perceptive_sim2sim.py --mode climb
```

To display the exact normalized depth frame used by the policy at the same
time, add `--show-depth`. The live image appears in the lower-right corner of
the MuJoCo viewer:

```bash
/home/black/anaconda3/envs/deploy_mujoco/bin/python \
  g1_perceptive_sim2sim.py --mode vault --show-depth
```

The optional composed scene is:

```bash
/home/black/anaconda3/envs/deploy_mujoco/bin/python g1_perceptive_sim2sim.py --mode sequential
```

## Headless regression

```bash
/home/black/anaconda3/envs/deploy_mujoco/bin/python g1_perceptive_sim2sim.py \
  --mode vault --headless --no-realtime
```

```bash
/home/black/anaconda3/envs/deploy_mujoco/bin/python g1_perceptive_sim2sim.py \
  --mode climb --headless --no-realtime
```

Results are saved under `sim2sim_output/<mode>/` as `summary.json`, `trajectory.csv`, and final metric/policy depth arrays. Use `--randomize-obstacle --seed N` for the trained pose range, or `--cmd VX VY WZ` to override the baked command.

## Validation result

- Vault: completed the obstacle, final torso x `6.570 m`, minimum torso height `0.719 m`, no fall.
- Climb: completed the obstacle, final torso x `7.660 m`, minimum torso height `0.731 m`, no fall.
- Sequential diagnostic: crossed both obstacles, then fell after the second-obstacle landing at about `18.1 s`; the independent vault and climb deployments are the acceptance paths.
- Full unit test suite: 40 tests passed.
