# G1 perceptive sim2sim: phase 1 acceptance

This phase establishes the robot-model and low-level-control contract before
adding a depth camera, obstacle scene, or policy inference.

## Sources of truth

- Training task: `Instinct-Perceptive-VaultClimb-VaeDistill-G1-Play-v0`
- Training URDF: `g1_29dof_torsobase_popsicle.urdf`
- Isaac reference: captured from the instantiated PLAY environment, after
  fixed-joint merging and actuator configuration
- MuJoCo model: `assets/g1/mjcf/g1_29dof_perceptive.xml`

The existing `g1_29dof_ame.xml` is not reused as the physical model because it
has a pelvis root, reversed waist tree, and collision bodies that differ from
the perceptive task. The new model is rooted at `torso_link` exactly like the
training articulation.

## Enforced parity

- 29 one-DoF joints plus one six-DoF floating base (`nq=36`, `nv=35`)
- 29 named torque actuators (`nu=29`)
- 30 moving rigid bodies after fixed-joint merging
- 29 URDF collision declarations: one sphere and 28 cylinders converted to
  MuJoCo capsules
- Per-body mass, center of mass, and full inertia tensor captured from Isaac
- Per-joint limits, armature, effort limits, default pose, stiffness, and
  damping captured from Isaac
- Simulation step 0.005 s and policy/control step 0.02 s
- Initial free-body pose is torso position `[0, 0, 0.82]`

Isaac reports joints in breadth-first articulation order while MuJoCo stores
joints in depth-first tree order. Policy observations and actions must
therefore use the explicit name mapping in `config/g1_perceptive.yaml`; raw
MuJoCo array order must never be sent directly to the policy.

The home pose has no robot self-contact in MuJoCo. The training configuration
enables articulation self-collision even though the URDF converter also records
`self_collision: false`; the phase-1 model keeps the URDF collision shapes
active and relies on MuJoCo's standard adjacent-body filtering.

## Reproduction and acceptance

```bash
/home/black/anaconda3/envs/deploy_mujoco/bin/python tools/build_g1_perceptive_mjcf.py
/home/black/anaconda3/envs/deploy_mujoco/bin/python tools/check_g1_perceptive_model.py
/home/black/anaconda3/envs/deploy_mujoco/bin/python -m unittest -v tests/test_g1_perceptive_model.py
```

The standing test is a constant-pose structural smoke test, not policy
behavior. It uses the captured default pose and effort limits with an 8x
test-only stiffness scale; deployment retains the unscaled training gains.
The scale is needed because the learned controller, rather than a passive
default-pose PD loop, supplies balance corrections during training.
