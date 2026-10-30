import json
import tempfile
import unittest
from pathlib import Path

from sam3_shelf.config import load_recipe, read_config_tree
from sam3_shelf.methods import get_method
from sam3_shelf.training.sampling import weighted_rows


class ConfigTests(unittest.TestCase):
    def test_reference_configs_extend_and_override(self):
        root = Path(__file__).resolve().parents[1]
        a = load_recipe(root / "configs/experiments/c7a_reference.json")
        b = load_recipe(root / "configs/experiments/delivery_joint.json")
        self.assertEqual(a["training"]["patience"], 8)
        self.assertEqual(b["training"]["patience"], 12)
        self.assertEqual(b["training"]["minimum_updates"], 2261)
        self.assertEqual(b["optimizer"], a["optimizer"])
        self.assertEqual(Path(a["model"]), (root / "checkpoints/base").resolve())

    def test_inheritance_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.json"
            path.write_text(json.dumps(dict(extends="a.json")))
            with self.assertRaisesRegex(ValueError, "cycle"):
                read_config_tree(path)

    def test_sampler_can_resume_at_any_exposure(self):
        rows = [dict(id="m", sampling_group="multi"), dict(id="s", sampling_group="single")]
        first = weighted_rows(rows, 0, 35)
        self.assertEqual(first, weighted_rows(rows, 0, 11) + weighted_rows(rows, 11, 35))
        cycle = weighted_rows(rows, 0, 9)
        self.assertEqual(sum(r["id"] == "m" for r in cycle), 8)

    def test_new_plugin_does_not_require_a_new_trainer(self):
        method = get_method(
            dict(method="score_only_focal_control", method_plugin="sam3_shelf.methods.example")
        )
        self.assertTrue(method.needs_teacher)
        self.assertEqual(type(method).__name__, "ScoreOnlyFocalControl")


if __name__ == "__main__":
    unittest.main()
