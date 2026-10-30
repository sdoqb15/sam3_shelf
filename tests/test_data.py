import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_shelf.data import assert_disjoint, load_mask, read_manifest, validate_manifest
from sam3_shelf.rle import decode_rle, encode_rle


class DataTests(unittest.TestCase):
    def test_alpha_and_disconnected_components(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.png"
            values = np.full((6, 6, 4), 255, dtype=np.uint8)
            values[..., 3] = 0
            values[0, 0, 3] = 255
            values[5, 5, 3] = 255
            Image.fromarray(values).save(path)
            mask = load_mask(path)
            self.assertEqual(int(mask.sum()), 2)
            self.assertTrue(np.array_equal(decode_rle(encode_rle(mask), 6, 6), mask))

    def test_relative_paths_and_rgb_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (6, 6), (1, 2, 3)).save(root / "a.png")
            Image.new("RGB", (6, 6), (1, 2, 3)).save(root / "b.png")
            for name in ("a", "b"):
                row = dict(id=name, group=name, image=f"{name}.png", instances=[])
                (root / f"{name}.jsonl").write_text(json.dumps(row) + "\n")
            a = read_manifest(root / "a.jsonl")
            b = read_manifest(root / "b.jsonl")
            validate_manifest(a)
            validate_manifest(b)
            self.assertEqual(a[0]["image"], str(root / "a.png"))
            with self.assertRaisesRegex(ValueError, "RGB overlap"):
                assert_disjoint(a, b)

    def test_rle_rejects_wrong_dimensions(self):
        with self.assertRaises(ValueError):
            decode_rle([1, 2], 10, 10)

    def test_single_pixel_masks_accept_0_1(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "mask.png"
            a = np.zeros((3, 3), dtype=np.uint8)
            a[1, 1] = 1
            Image.fromarray(a).save(p)
            self.assertEqual(int(load_mask(p).sum()), 1)


if __name__ == "__main__":
    unittest.main()
