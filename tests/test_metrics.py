import unittest
import numpy as np

from sam3_shelf.evaluation import score_masks
from sam3_shelf.metrics import match_pairs
from sam3_shelf.postprocessing import mask_nms


class MetricTests(unittest.TestCase):
    def test_split_package_does_not_count_as_complete(self):
        gt = np.ones((4, 4), dtype=bool)
        parts = np.zeros((2, 4, 4), dtype=bool)
        parts[0, :, :2] = True
        parts[1, :, 2:] = True
        m, _, _ = score_masks([gt], parts, np.array([0.9, 0.8]))
        self.assertEqual(m["complete_gt"], 0)

    def test_merged_front_and_back_is_not_complete(self):
        gt = np.zeros((4, 4), dtype=bool)
        gt[:2] = True
        m, _, _ = score_masks([gt], np.ones((1, 4, 4), dtype=bool), np.array([0.99]))
        self.assertEqual(m["complete_gt"], 0)

    def test_cardinality_before_iou_sum(self):
        g, p = match_pairs(np.array([[0.9, 0.6], [0.6, 0.49]]))
        self.assertEqual(len(g), 2)

    def test_duplicate_prediction_does_not_recall_two_gt(self):
        gt = np.ones((4, 4), dtype=bool)
        m, _, _ = score_masks([gt, gt], np.array([gt]), np.array([0.9]))
        self.assertEqual(m["complete_gt"], 1)

    def test_nms_suppresses_duplicates_not_contained_fragments(self):
        whole = np.ones((6, 6), dtype=bool)
        part = whole.copy()
        part[3:] = False
        masks = np.array([part, whole, whole])
        keep = mask_nms(masks, np.array([0.99, 0.9, 0.8]), 0.98)
        self.assertEqual(keep.tolist(), [0, 1])

    def test_topn_and_threshold_are_actual_outputs(self):
        gt = np.zeros((4, 4), dtype=bool)
        gt[:2] = True
        wrong = ~gt
        m, top, ranks = score_masks([gt], np.array([wrong, gt]), np.array([0.9, 0.1]))
        self.assertEqual(m["complete_gt"], 0)
        self.assertEqual(top["3"]["complete_gt"], 0)


if __name__ == "__main__":
    unittest.main()
