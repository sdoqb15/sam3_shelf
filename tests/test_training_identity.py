"""Code fingerprints must be independent of symlinked import paths."""
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sam3_shelf.methods import get_method
from sam3_shelf.training.trainer import code_identity


class TrainingIdentityTests(unittest.TestCase):
    def test_method_sources_imported_through_symlink(self):
        method = get_method({"method": "c7a_joint"})
        expected = code_identity(method)
        real_getfile = inspect.getfile
        package = Path(real_getfile(type(method))).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "linked_package"
            alias.symlink_to(package, target_is_directory=True)

            def aliased_source(cls):
                return str(alias / Path(real_getfile(cls)).resolve().relative_to(package))

            with patch("inspect.getfile", side_effect=aliased_source):
                self.assertEqual(code_identity(method), expected)


if __name__ == "__main__":
    unittest.main()
