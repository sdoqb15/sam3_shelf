import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "download_models", Path(__file__).resolve().parents[1] / "scripts/download_models.py"
)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadTests(unittest.TestCase):
    def example(self, root, revision="a" * 40):
        path = root / "configs/checkpoints/huggingface.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                dict(
                    models=dict(
                        c7a=dict(
                            repo_id="owner/model",
                            revision=revision,
                            weights_sha256=hashlib.sha256(b"verified weights").hexdigest(),
                        )
                    )
                )
            )
        )

    def test_pins_revision_and_checks_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.example(root)
            calls = []

            def fetch(repo_id, **kwargs):
                calls.append((repo_id, kwargs))
                (kwargs["local_dir"] / "model.safetensors").write_bytes(b"verified weights")

            with (
                patch.object(downloader, "ROOT", root),
                patch.dict(
                    sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=fetch)}
                ),
            ):
                downloader.download("c7a", root / "out")
            self.assertEqual(calls[0][1]["revision"], "a" * 40)
            self.assertEqual(
                json.loads((root / "out/download_receipt.json").read_text())["status"], "verified"
            )

    def test_corrupt_weights_are_not_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.example(root)

            def fetch(repo_id, **kwargs):
                (kwargs["local_dir"] / "model.safetensors").write_bytes(b"corrupt")

            with (
                patch.object(downloader, "ROOT", root),
                patch.dict(
                    sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=fetch)}
                ),
            ):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    downloader.download("c7a", root / "out")
            self.assertNotEqual(
                json.loads((root / "out/download_receipt.json").read_text())["status"], "verified"
            )

    def test_refuses_unpinned_release_and_unrelated_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.example(root, None)
            with (
                patch.object(downloader, "ROOT", root),
                patch.dict(
                    sys.modules,
                    {
                        "huggingface_hub": types.SimpleNamespace(
                            snapshot_download=lambda *a, **kw: None
                        )
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, "not yet published"):
                    downloader.download("c7a", root / "out")
                self.example(root)
                (root / "out").mkdir()
                (root / "out/model.safetensors").write_bytes(b"keep")
                with self.assertRaisesRegex(ValueError, "empty destination"):
                    downloader.download("c7a", root / "out")
                self.assertEqual((root / "out/model.safetensors").read_bytes(), b"keep")
