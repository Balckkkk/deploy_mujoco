#!/usr/bin/env python3
"""Build the torso-root G1 MJCF used by the perceptive policy.

The URDF supplies kinematics, visuals, and collision shapes.  The JSON
captured from the running Isaac Lab task supplies the post-fixed-joint-merge
mass properties and actuator parameters.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = Path(
    "/home/black/Code/Locomotion/depth/InstinctLab/source/instinctlab/instinctlab/"
    "assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf"
)
DEFAULT_REFERENCE = PROJECT_ROOT / "config/g1_perceptive_isaac_reference.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "assets/g1/mjcf/g1_29dof_perceptive.xml"

# Final values resolved from the target run's params/env.yaml. The training
# quaternion uses the robotics/world camera convention (+X forward, +Z up).
DEPTH_CAMERA_POS = (0.04812725697526883, 0.015, 0.4370303957245772)
DEPTH_CAMERA_PITCH = math.radians(34.0)
WORLD_TO_OPENGL_CAMERA = ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def vector(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None:
        return default
    return tuple(float(value) for value in text.split())


def fmt(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def matmul(a: tuple[tuple[float, ...], ...], b: tuple[tuple[float, ...], ...]):
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def matvec(matrix: tuple[tuple[float, ...], ...], value: tuple[float, ...]):
    return tuple(sum(matrix[row][k] * value[k] for k in range(3)) for row in range(3))


def rpy_matrix(rpy: tuple[float, ...]):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def compose(first, second):
    first_pos, first_rot = first
    second_pos, second_rot = second
    rotated = matvec(first_rot, second_pos)
    return (
        tuple(first_pos[index] + rotated[index] for index in range(3)),
        matmul(first_rot, second_rot),
    )


def origin(element: ET.Element | None):
    if element is None:
        return ((0.0, 0.0, 0.0), IDENTITY)
    return (
        vector(element.get("xyz"), (0.0, 0.0, 0.0)),
        rpy_matrix(vector(element.get("rpy"), (0.0, 0.0, 0.0))),
    )


def matrix_quaternion(matrix: tuple[tuple[float, ...], ...]):
    """Return a normalized MuJoCo-order quaternion (w, x, y, z)."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = (
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        )
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        quat = (
            (matrix[2][1] - matrix[1][2]) / scale,
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
        )
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        quat = (
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        quat = (
            (matrix[1][0] - matrix[0][1]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            0.25 * scale,
        )
    norm = math.sqrt(sum(value * value for value in quat))
    result = tuple(value / norm for value in quat)
    return tuple(-value for value in result) if result[0] < 0.0 else result


def add_pose(attributes: dict[str, str], transform) -> None:
    position, rotation = transform
    attributes["pos"] = fmt(position)
    quaternion = matrix_quaternion(rotation)
    if max(abs(quaternion[index]) for index in range(1, 4)) > 1e-12:
        attributes["quat"] = fmt(quaternion)


class Builder:
    def __init__(self, urdf_path: Path, reference_path: Path) -> None:
        self.urdf_path = urdf_path
        self.reference = json.loads(reference_path.read_text(encoding="utf-8"))
        urdf_root = ET.parse(urdf_path).getroot()
        self.links = {element.get("name"): element for element in urdf_root.findall("link")}
        self.joints = {element.get("name"): element for element in urdf_root.findall("joint")}
        self.children: dict[str, list[ET.Element]] = {}
        child_links = set()
        for joint in self.joints.values():
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            self.children.setdefault(parent, []).append(joint)
            child_links.add(child)
        roots = set(self.links) - child_links
        if roots != {"torso_link"}:
            raise ValueError(f"Expected torso_link as the only URDF root, got {sorted(roots)}")
        self.body_reference = {
            name: index for index, name in enumerate(self.reference["body_names"])
        }
        self.joint_reference = {
            name: index for index, name in enumerate(self.reference["joint_names"])
        }
        self.mesh_files: dict[str, str] = {}
        self.collision_counts: dict[str, int] = {}
        self.mujoco_joint_order: list[str] = []

    def build(self) -> ET.Element:
        root = ET.Element("mujoco", {"model": "g1_29dof_perceptive"})
        ET.SubElement(
            root,
            "compiler",
            {"angle": "radian", "meshdir": "../meshes", "autolimits": "true"},
        )
        ET.SubElement(
            root,
            "option",
            {"timestep": "0.005", "gravity": "0 0 -9.81", "integrator": "implicitfast"},
        )
        defaults = ET.SubElement(root, "default")
        ET.SubElement(defaults, "joint", {"limited": "true", "frictionloss": "0"})
        visual = ET.SubElement(defaults, "default", {"class": "visual"})
        ET.SubElement(
            visual,
            "geom",
            {"type": "mesh", "density": "0", "contype": "0", "conaffinity": "0", "group": "2"},
        )
        collision = ET.SubElement(defaults, "default", {"class": "collision"})
        ET.SubElement(
            collision,
            "geom",
            {"density": "0", "priority": "1", "condim": "6", "group": "3"},
        )
        ET.SubElement(defaults, "site", {"size": "0.01", "rgba": "1 0 0 1", "group": "5"})

        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "material", {"name": "white", "rgba": "0.7 0.7 0.7 1"})
        ET.SubElement(asset, "material", {"name": "dark", "rgba": "0.2 0.2 0.2 1"})
        worldbody = ET.SubElement(root, "worldbody")
        torso = ET.SubElement(worldbody, "body", {"name": "torso_link", "pos": "0 0 0.82"})
        ET.SubElement(torso, "freejoint", {"name": "floating_base_joint"})
        self.add_inertial(torso, "torso_link")
        camera_rotation_world = rpy_matrix((0.0, DEPTH_CAMERA_PITCH, 0.0))
        camera_rotation_opengl = matmul(camera_rotation_world, WORLD_TO_OPENGL_CAMERA)
        ET.SubElement(
            torso,
            "camera",
            {
                "name": "depth_camera",
                "mode": "fixed",
                "pos": fmt(DEPTH_CAMERA_POS),
                "quat": fmt(matrix_quaternion(camera_rotation_opengl)),
                "fovy": "70",
            },
        )
        ET.SubElement(
            torso,
            "site",
            {
                "name": "depth_camera_frame",
                "pos": fmt(DEPTH_CAMERA_POS),
                "quat": fmt(matrix_quaternion(camera_rotation_world)),
                "rgba": "0 0.6 1 1",
            },
        )
        self.walk_link("torso_link", torso, ((0.0, 0.0, 0.0), IDENTITY))

        for mesh_name, filename in sorted(self.mesh_files.items()):
            ET.SubElement(asset, "mesh", {"name": mesh_name, "file": filename})

        actuator = ET.SubElement(root, "actuator")
        joint_data = self.reference["joint"]
        for index, name in enumerate(self.reference["joint_names"]):
            effort = float(joint_data["effort_limit"][index])
            ET.SubElement(
                actuator,
                "motor",
                {
                    "name": name,
                    "joint": name,
                    "ctrllimited": "true",
                    "ctrlrange": fmt((-effort, effort)),
                },
            )
        sensor = ET.SubElement(root, "sensor")
        ET.SubElement(sensor, "gyro", {"name": "imu_ang_vel", "site": "imu_in_torso"})
        ET.SubElement(sensor, "velocimeter", {"name": "imu_lin_vel", "site": "imu_in_torso"})
        ET.SubElement(sensor, "accelerometer", {"name": "imu_lin_acc", "site": "imu_in_torso"})
        ET.SubElement(sensor, "framequat", {"name": "torso_quat", "objtype": "body", "objname": "torso_link"})
        keyframe = ET.SubElement(root, "keyframe")
        default_by_name = dict(
            zip(
                self.reference["joint_names"],
                self.reference["joint"]["default_position"],
                strict=True,
            )
        )
        home_qpos = [0.0, 0.0, 0.82, 1.0, 0.0, 0.0, 0.0]
        home_qpos.extend(default_by_name[name] for name in self.mujoco_joint_order)
        ET.SubElement(keyframe, "key", {"name": "home", "qpos": fmt(home_qpos)})
        return root

    def add_inertial(self, body: ET.Element, name: str) -> None:
        index = self.body_reference[name]
        body_data = self.reference["body"]
        inertia = body_data["inertia"][index]
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": fmt(body_data["com_pose_wxyz"][index][:3]),
                "mass": f"{body_data['mass'][index]:.10g}",
                "fullinertia": fmt(
                    (inertia[0], inertia[4], inertia[8], inertia[1], inertia[2], inertia[5])
                ),
            },
        )

    def walk_link(self, link_name: str, owner: ET.Element, owner_to_link) -> None:
        link = self.links[link_name]
        self.add_visuals(link, owner, owner_to_link)
        self.add_collisions(link, owner, owner_to_link)
        if link_name in {"imu_in_torso", "imu_in_pelvis", "LL_FOOT", "LR_FOOT"}:
            ET.SubElement(owner, "site", {"name": link_name, "pos": fmt(owner_to_link[0])})
        for joint in self.children.get(link_name, []):
            joint_transform = compose(owner_to_link, origin(joint.find("origin")))
            child_name = joint.find("child").get("link")
            if joint.get("type") == "fixed":
                self.walk_link(child_name, owner, joint_transform)
            else:
                child = ET.SubElement(owner, "body", {"name": child_name})
                add_pose(child.attrib, joint_transform)
                self.add_inertial(child, child_name)
                reference_index = self.joint_reference[joint.get("name")]
                limits = self.reference["joint"]["position_limit"][reference_index]
                ET.SubElement(
                    child,
                    "joint",
                    {
                        "name": joint.get("name"),
                        "axis": joint.find("axis").get("xyz"),
                        "range": fmt(limits),
                        "armature": f"{self.reference['joint']['armature'][reference_index]:.10g}",
                        "damping": "0",
                    },
                )
                self.mujoco_joint_order.append(joint.get("name"))
                self.walk_link(child_name, child, ((0.0, 0.0, 0.0), IDENTITY))

    def add_visuals(self, link: ET.Element, owner: ET.Element, owner_to_link) -> None:
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = Path(mesh.get("filename")).name
            mesh_name = Path(filename).stem
            self.mesh_files[mesh_name] = filename
            attributes = {"class": "visual", "mesh": mesh_name}
            material = visual.find("material")
            if material is not None and material.get("name") in {"white", "dark"}:
                attributes["material"] = material.get("name")
            add_pose(attributes, compose(owner_to_link, origin(visual.find("origin"))))
            ET.SubElement(owner, "geom", attributes)

    def add_collisions(self, link: ET.Element, owner: ET.Element, owner_to_link) -> None:
        owner_name = owner.get("name")
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            collision_transform = compose(owner_to_link, origin(collision.find("origin")))
            count = self.collision_counts.get(owner_name, 0) + 1
            self.collision_counts[owner_name] = count
            attributes = {
                "name": f"{owner_name}_collision_{count}",
                "class": "collision",
            }
            sphere = geometry.find("sphere")
            cylinder = geometry.find("cylinder")
            box = geometry.find("box")
            if sphere is not None:
                attributes.update({"type": "sphere", "size": sphere.get("radius")})
                add_pose(attributes, collision_transform)
            elif cylinder is not None:
                radius = float(cylinder.get("radius"))
                half_length = 0.5 * float(cylinder.get("length"))
                center, rotation = collision_transform
                offset = matvec(rotation, (0.0, 0.0, half_length))
                first = tuple(center[index] - offset[index] for index in range(3))
                second = tuple(center[index] + offset[index] for index in range(3))
                attributes.update(
                    {"type": "capsule", "size": f"{radius:.10g}", "fromto": fmt(first + second)}
                )
            elif box is not None:
                size = tuple(0.5 * value for value in vector(box.get("size"), (0.0, 0.0, 0.0)))
                attributes.update({"type": "box", "size": fmt(size)})
                add_pose(attributes, collision_transform)
            else:
                raise NotImplementedError(f"Unsupported collision geometry in {link.get('name')}")
            ET.SubElement(owner, "geom", attributes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Builder(args.urdf.resolve(), args.reference.resolve()).build()
    ET.indent(root, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(args.output, encoding="unicode", xml_declaration=False)
    with args.output.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
