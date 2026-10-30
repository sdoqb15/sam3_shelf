import json
import tempfile
import unittest
from pathlib import Path

from sam3_shelf.report import build_report


class ReportTests(unittest.TestCase):
    def test_compare_requires_same_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in (1, 2):
                p = root / str(i)
                p.mkdir()
                (p / "summary.json").write_text(
                    json.dumps(
                        dict(samples=[dict(id=str(i))], metrics={}, top_n_metrics={}, has_gt=False)
                    )
                )
            with self.assertRaisesRegex(ValueError, "same ordered"):
                build_report([("A", root / "1"), ("B", root / "2")], root / "report")


if __name__ == "__main__":
    unittest.main()
