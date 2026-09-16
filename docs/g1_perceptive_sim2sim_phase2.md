# G1 perceptive sim2sim: phase 2 acceptance

Phase 2 adds a reusable MuJoCo depth-camera interface and validates the exact
camera contract recorded by the target training run. It does not yet add the
phase-3 vault/climb scene or policy inference.

## Training contract

- Parent body: `torso_link`
- Position in torso: `[0.04812725697526883, 0.015, 0.4370303957245772]` m
- Orientation: robotics/world camera convention, 34 degrees downward pitch
- Resolution: height 32, width 48
- Vertical FOV: 70 degrees
- Horizontal FOV: approximately 92.8113 degrees (square pixels)
- Update period: 0.02 s (50 Hz)
- Measurement: axial `distance_to_image_plane`, not Euclidean ray distance
- Valid ray start distance: 0.05 m
- Preprocessing: invalid/far values become 3 m, clip to `[0, 3]`, normalize to `[0, 1]`
- Policy layout: channel-first `(1, 32, 48)`
- Visible MuJoCo geom groups: scene group 0 and robot visuals group 2;
  collision capsules in group 3 are hidden from the depth renderer

The target run contains only normalization in its final noise pipeline. The
older parent configuration's crop-and-resize step is deliberately not applied.

## Reusable interface

`depth_camera.py` provides:

- `DepthCameraConfig.from_dict(...)`
- `MujocoDepthCamera.capture(...)` with 50 Hz frame holding
- metric `depth_m` and policy-ready `policy_image`
- matching camera intrinsics
- camera-frame and world-frame point-cloud deprojection
- explicit RGB debug capture using the same visibility mask

The model camera is generated into
`assets/g1/mjcf/g1_29dof_perceptive.xml`; its quaternion converts the training
`+X forward, +Z up` camera convention to MuJoCo/OpenGL's `-Z forward, +Y up`
convention.

## Tests

Headless numerical check and debug-image export:

```bash
cd /home/black/Code/deploy_mujoco
MUJOCO_GL=egl /home/black/anaconda3/envs/deploy_mujoco/bin/python \
  tools/test_g1_depth_camera.py
```

Interactive RGB/depth comparison window:

```bash
cd /home/black/Code/deploy_mujoco
MUJOCO_GL=egl /home/black/anaconda3/envs/deploy_mujoco/bin/python \
  tools/test_g1_depth_camera.py --visualize
```

Unit and regression tests:

```bash
cd /home/black/Code/deploy_mujoco
MUJOCO_GL=egl /home/black/anaconda3/envs/deploy_mujoco/bin/python -m unittest -v \
  tests/test_depth_camera.py tests/test_g1_perceptive_model.py
```

The first two commands save `rgb.png`, `depth.png`, `comparison.png`, metric
depth, the policy tensor, and `report.json` under `depth_camera_debug/`.
