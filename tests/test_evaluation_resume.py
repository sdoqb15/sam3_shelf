import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from sam3_shelf.evaluation import evaluate


class EvaluationResumeTests(unittest.TestCase):
    def test_resume_reuses_predictions_and_detects_changed_annotation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"test-model-identity")
            Image.new("RGB", (4, 4), (3, 4, 5)).save(root / "image.png")
            Image.new("L", (4, 4), 255).save(root / "mask.png")
            row = dict(id="one", group="one", image="image.png", instances=[dict(mask="mask.png")])
            manifest = root / "data.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            output = root / "evaluation"
            with (
                patch("sam3_shelf.model.load_model", return_value=(None, None)),
                patch(
                    "sam3_shelf.evaluation.predict",
                    return_value=(np.ones((1, 4, 4), bool), np.array([0.8])),
                ) as prediction,
            ):
                first = evaluate(model, manifest, output, device="cpu")
                self.assertEqual(first["metrics"]["complete_gt"], 1)
                self.assertEqual(prediction.call_count, 1)
                (output / "summary.json").unlink()
                second = evaluate(model, manifest, output, device="cpu", resume=True)
                self.assertEqual(prediction.call_count, 1)
                self.assertEqual(first["metrics"], second["metrics"])
                self.assertEqual(second["resources"]["reused_images"], 1)
                mask = Image.new("L", (4, 4), 255)
                mask.putpixel((0, 0), 0)
                mask.save(root / "mask.png")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    evaluate(model, manifest, output, device="cpu", resume=True)


if __name__ == "__main__":
    unittest.main()
