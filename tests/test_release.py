import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).resolve().parents[1] / "scripts/check_release.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ReleaseTests(unittest.TestCase):
    def test_nested_checkpoint_configuration_is_audited(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "configs/checkpoints/models.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"path":"/' + 'home/private/model"}')
            result = checker.audit(root)
            self.assertEqual(result["files_checked"], 1)
            self.assertIn("private absolute machine path", result["problems"][0])

    def test_external_runtime_links_are_not_a_public_source_file(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as external:
            root = Path(folder)
            (root / "checkpoints").symlink_to(external, target_is_directory=True)
            (Path(external) / "model.safetensors").write_bytes(b"private weights")
            (root / "README.md").write_text("source")
            self.assertEqual(checker.audit(root)["problems"], [])
