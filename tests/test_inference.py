import unittest

import numpy as np
import torch
from PIL import Image

from sam3_shelf import DEFAULT_PROMPT
from sam3_shelf.inference import predict
from sam3_shelf.postprocessing import postprocess


class Processor:
    def __init__(self):
        self.prompt = None

    def __call__(self, images, text, return_tensors):
        self.prompt = text

        class Batch(dict):
            def to(self, device):
                return self

        return Batch()

    def post_process_instance_segmentation(self, outputs, threshold, mask_threshold, target_sizes):
        scores = torch.tensor([0.099609375, 0.10009765625, 0.9, 0.8], dtype=torch.bfloat16)
        masks = torch.ones((4, 4, 4), dtype=torch.bool)
        masks[0, :2] = False
        masks[1, 2:] = False
        masks[2] = False
        masks[3] = masks[1]
        keep = scores > threshold
        return [dict(scores=scores[keep], masks=masks[keep])]


class Model:
    device = torch.device("cpu")

    def eval(self):
        return self

    def __call__(self, **inputs):
        return None


class InferenceTests(unittest.TestCase):
    def test_fp32_filter_retains_just_above_point_one(self):
        result = postprocess(Processor(), None, 0.1, [(4, 4)])[0]
        scores = result["scores"].float().tolist()
        self.assertIn(0.10009765625, scores)
        self.assertNotIn(0.099609375, scores)

    def test_inference_fixed_prompt_empty_removal_and_nms(self):
        processor = Processor()
        masks, scores = predict(Model(), processor, Image.new("RGB", (4, 4)))
        self.assertEqual(processor.prompt, DEFAULT_PROMPT)
        self.assertEqual(len(masks), 1)
        self.assertGreater(float(scores[0]), 0.1)
        self.assertTrue(np.all(masks[0, :2]))

    def test_threshold_below_floor_rejected(self):
        with self.assertRaises(ValueError):
            predict(Model(), Processor(), Image.new("RGB", (4, 4)), 0.01)


if __name__ == "__main__":
    unittest.main()
