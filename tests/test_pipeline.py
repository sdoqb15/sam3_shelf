import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from sam3_shelf.data import read_manifest, validate_manifest
from sam3_shelf.io import read_json
from sam3_shelf.pipeline import run_experiment


class PipelineTests(unittest.TestCase):
    def test_baseline_precedes_training_and_shared_review_is_resumable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "base"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"base")
            Image.new("RGB", (4, 4), (2, 3, 4)).save(root / "train.png")
            Image.new("RGB", (4, 4), (5, 6, 7)).save(root / "test.png")
            Image.new("L", (4, 4), 255).save(root / "mask.png")
            for name in ("train", "test"):
                (root / f"{name}.jsonl").write_text(
                    json.dumps(
                        dict(
                            id=name,
                            group=name,
                            image=f"{name}.png",
                            instances=[dict(mask="mask.png")],
                        )
                    )
                    + "\n"
                )
            train_rows = read_manifest(root / "train.jsonl")
            validate_manifest(train_rows)
            out = root / "run"
            config = dict(
                model=str(model),
                output_dir=str(out),
                method="c7a_joint",
                experiment=dict(name="Control"),
                data=dict(root=None),
                evaluations=[
                    dict(
                        name="test",
                        manifest=str(root / "test.jsonl"),
                        scope="held_out_test",
                        compare_baseline=True,
                    )
                ],
            )
            events = []

            def load(path, device):
                events.append("inference")
                return None, None

            def train(*args):
                events.append("training")
                (out / "best").mkdir(exist_ok=True)
                (out / "best/model.safetensors").write_bytes(b"trained")
                return dict(status="complete", best_step=1)

            with (
                patch("sam3_shelf.config.load_recipe", return_value=config),
                patch("sam3_shelf.pipeline.preflight", return_value=(config, train_rows, [], {})),
                patch("sam3_shelf.training.trainer.train", side_effect=train),
                patch("sam3_shelf.model.load_model", side_effect=load),
                patch(
                    "sam3_shelf.evaluation.predict",
                    return_value=(np.ones((1, 4, 4), bool), np.array([0.8])),
                ) as prediction,
            ):
                first = run_experiment(root / "config.json", device="cpu")
                self.assertEqual(events, ["inference", "training", "inference"])
                self.assertEqual(first["status"], "complete")
                review = read_json(out / "reviews/test/manifest.json")
                self.assertEqual(
                    [m["label"] for m in review["models"]], ["Original SAM3", "Control"]
                )
                self.assertEqual(prediction.call_count, 2)
                run_experiment(root / "config.json", device="cpu", resume=True)
                self.assertEqual(prediction.call_count, 2)


if __name__ == "__main__":
    unittest.main()
