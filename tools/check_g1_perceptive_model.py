#!/usr/bin/env python3
"""Check G1 URDF/Isaac/MuJoCo parity and run a PD standing smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config/g1_perceptive.yaml"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def object_name(model: mujoco.MjModel, object_type, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(object_id))


def joint_mapping(model: mujoco.MjModel, names: list[str]):
    qpos, dof, actuator = [], [], []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise AssertionError(f"Missing joint/actuator: {name}")
        qpos.append(model.jnt_qposadr[joint_id])
        dof.append(model.jnt_dofadr[joint_id])
        actuator.append(actuator_id)
    return tuple(np.asarray(values, dtype=np.int32) for values in (qpos, dof, actuator))


def full_body_inertia(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body_id])
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T


def check_structure(config: dict, reference: dict, model: mujoco.MjModel) -> dict:
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    assert model.nbody == 31
    assert model.opt.timestep == float(config["sim_dt"])
    assert float(config["control_dt"]) == float(config["sim_dt"]) * int(config["control_decimation"])

    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["base_body"])
    base_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, config["base_joint"])
    assert base_body == 1
    assert model.jnt_bodyid[base_joint] == base_body
    assert model.jnt_type[base_joint] == mujoco.mjtJoint.mjJNT_FREE

    policy_names = config["policy_joint_names"]
    assert policy_names == reference["joint_names"]
    qpos_ids, dof_ids, actuator_ids = joint_mapping(model, policy_names)
    model_joint_names = {
        object_name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
        if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
    }
    assert model_joint_names == set(policy_names)
    assert {
        object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
    } == set(policy_names)

    ref_joint = reference["joint"]
    for index, name in enumerate(policy_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        np.testing.assert_allclose(model.jnt_range[joint_id], ref_joint["position_limit"][index], atol=5e-7)
        np.testing.assert_allclose(model.dof_armature[dof_ids[index]], ref_joint["armature"][index], atol=5e-9)
        effort = ref_joint["effort_limit"][index]
        np.testing.assert_allclose(model.actuator_ctrlrange[actuator_ids[index]], [-effort, effort], atol=1e-7)

    body_names = {
        object_name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(1, model.nbody)
    }
    assert body_names == set(reference["body_names"])
    ref_body = reference["body"]
    max_mass_error = 0.0
    max_com_error = 0.0
    max_inertia_error = 0.0
    for index, name in enumerate(reference["body_names"]):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        mass_error = abs(model.body_mass[body_id] - ref_body["mass"][index])
        com_error = np.max(np.abs(model.body_ipos[body_id] - ref_body["com_pose_wxyz"][index][:3]))
        expected = np.asarray(ref_body["inertia"][index]).reshape(3, 3)
        inertia_error = np.max(np.abs(full_body_inertia(model, body_id) - expected))
        max_mass_error = max(max_mass_error, float(mass_error))
        max_com_error = max(max_com_error, float(com_error))
        max_inertia_error = max(max_inertia_error, float(inertia_error))
    assert max_mass_error < 1e-8
    assert max_com_error < 1e-8
    # MuJoCo diagonalizes full inertia tensors at compile time. Float32 Isaac
    # reference values therefore return with a few 1e-8 of eigensolver noise.
    assert max_inertia_error < 5e-8

    collision_ids = np.flatnonzero(model.geom_group == 3)
    source_root = ET.parse(project_path(config["source_urdf_path"])).getroot()
    source_collisions = source_root.findall(".//collision")
    source_spheres = sum(collision.find("geometry/sphere") is not None for collision in source_collisions)
    source_cylinders = sum(collision.find("geometry/cylinder") is not None for collision in source_collisions)
    assert len(source_collisions) == 29
    assert len(collision_ids) == len(source_collisions)
    assert sum(model.geom_type[index] == mujoco.mjtGeom.mjGEOM_SPHERE for index in collision_ids) == source_spheres
    assert sum(model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CAPSULE for index in collision_ids) == source_cylinders

    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, config["home_keyframe"])
    assert home_id >= 0
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qpos[:3], config["base_position"], atol=1e-12)
    np.testing.assert_allclose(data.qpos[qpos_ids], ref_joint["default_position"], atol=5e-8)
    assert data.ncon == 0, "The home pose has unexpected robot self-contact"

    return {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "bodies": model.nbody - 1,
        "collisions": len(collision_ids),
        "total_mass": float(np.sum(model.body_mass)),
        "max_mass_error": max_mass_error,
        "max_com_error": max_com_error,
        "max_inertia_error": max_inertia_error,
    }


def run_stand_test(config: dict, reference: dict) -> dict:
    spec = mujoco.MjSpec.from_file(str(project_path(config["model_path"])))
    stand = config["stand_test"]
    spec.worldbody.add_geom(
        name="phase1_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5.0, 5.0, 0.1],
        friction=stand["ground_friction"],
        condim=3,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, config["home_keyframe"])
    mujoco.mj_resetDataKeyframe(model, data, home_id)

    names = reference["joint_names"]
    qpos_ids, dof_ids, actuator_ids = joint_mapping(model, names)
    joint = reference["joint"]
    target = np.asarray(joint["default_position"], dtype=np.float64)
    gain = float(stand["gain_multiplier"])
    kp = gain * np.asarray(joint["stiffness"], dtype=np.float64)
    kd = np.sqrt(gain) * np.asarray(joint["damping"], dtype=np.float64)
    effort = np.asarray(joint["effort_limit"], dtype=np.float64)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["base_body"])

    min_height = np.inf
    max_tilt = 0.0
    max_joint_error = 0.0
    max_torque = 0.0
    max_contacts = 0
    steps = round(float(stand["duration"]) / model.opt.timestep)
    for _ in range(steps):
        torque = kp * (target - data.qpos[qpos_ids]) - kd * data.qvel[dof_ids]
        torque = np.clip(torque, -effort, effort)
        data.ctrl[actuator_ids] = torque
        mujoco.mj_step(model, data)
        quaternion = data.xquat[base_id]
        tilt = 2.0 * np.arccos(np.clip(abs(quaternion[0]), 0.0, 1.0))
        min_height = min(min_height, float(data.xpos[base_id, 2]))
        max_tilt = max(max_tilt, float(tilt))
        max_joint_error = max(max_joint_error, float(np.max(np.abs(target - data.qpos[qpos_ids]))))
        max_torque = max(max_torque, float(np.max(np.abs(torque))))
        max_contacts = max(max_contacts, data.ncon)

    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    assert min_height >= float(stand["minimum_base_height"])
    assert max_tilt <= float(stand["maximum_tilt_radians"])
    assert max_joint_error <= float(stand["maximum_joint_error_radians"])
    return {
        "duration": steps * model.opt.timestep,
        "gain_multiplier": gain,
        "minimum_base_height": min_height,
        "maximum_tilt_radians": max_tilt,
        "maximum_joint_error_radians": max_joint_error,
        "maximum_abs_torque": max_torque,
        "maximum_contacts": max_contacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-stand", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reference = json.loads(project_path(config["isaac_reference_path"]).read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(project_path(config["model_path"])))
    report = {"parity": check_structure(config, reference, model)}
    if not args.skip_stand:
        report["stand_test"] = run_stand_test(config, reference)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
