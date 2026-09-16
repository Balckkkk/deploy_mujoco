from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import json
import mujoco
import yaml

from tools.build_g1_perceptive_mjcf import Builder
from tools.check_g1_perceptive_model import check_structure, project_path, run_stand_test


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/g1_perceptive.yaml"


def load_contract():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    reference = json.loads(project_path(config["isaac_reference_path"]).read_text(encoding="utf-8"))
    return config, reference


class G1PerceptiveModelTest(unittest.TestCase):
    def test_generated_mjcf_is_current(self):
        config, _ = load_contract()
        root = Builder(
            project_path(config["source_urdf_path"]),
            project_path(config["isaac_reference_path"]),
        ).build()
        ET.indent(root, space="  ")
        generated = ET.tostring(root, encoding="unicode") + "\n"
        committed = project_path(config["model_path"]).read_text(encoding="utf-8")
        self.assertEqual(generated, committed)

    def test_urdf_isaac_mujoco_parity(self):
        config, reference = load_contract()
        model = mujoco.MjModel.from_xml_path(str(project_path(config["model_path"])))
        report = check_structure(config, reference, model)
        self.assertEqual(report["collisions"], 29)
        self.assertEqual(report["bodies"], 30)

    def test_pd_standing_smoke(self):
        config, reference = load_contract()
        report = run_stand_test(config, reference)
        self.assertEqual(report["duration"], config["stand_test"]["duration"])


if __name__ == "__main__":
    unittest.main()
