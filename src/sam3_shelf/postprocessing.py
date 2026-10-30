"""FP32 threshold comparisons and greedy mask near-duplicate suppression."""

import numpy as np


def postprocess(processor, outputs, threshold, target_sizes):
    results = processor.post_process_instance_segmentation(
        outputs, threshold=max(0.0, threshold - 0.01), mask_threshold=0.5, target_sizes=target_sizes
    )
    filtered = []
    for result in results:
        keep = result["scores"].float() > threshold
        filtered.append({k: v[keep] for (k, v) in result.items()})
    return filtered


def mask_nms(masks, scores, threshold=0.98):
    if threshold >= 1:
        return np.argsort(-np.asarray(scores), kind="stable")
    areas = []
    boxes = []
    for mask in masks:
        (y, x) = np.where(mask)
        areas.append(len(x))
        boxes.append(
            (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
            if len(x)
            else (0, 0, 0, 0)
        )
    order = np.argsort(-np.asarray(scores), kind="stable")
    kept = []
    for candidate in order:
        suppress = False
        for previous in kept:
            if (
                min(areas[candidate], areas[previous]) / max(areas[candidate], areas[previous], 1)
                <= threshold
            ):
                continue
            (a, b) = (boxes[candidate], boxes[previous])
            (x0, y0) = (max(a[0], b[0]), max(a[1], b[1]))
            (x1, y1) = (min(a[2], b[2]), min(a[3], b[3]))
            if x1 <= x0 or y1 <= y0:
                continue
            intersection = np.count_nonzero(
                masks[candidate][y0:y1, x0:x1] & masks[previous][y0:y1, x0:x1]
            )
            iou = intersection / max(areas[candidate] + areas[previous] - intersection, 1)
            if iou > threshold:
                suppress = True
                break
        if not suppress:
            kept.append(int(candidate))
    return np.asarray(kept, dtype=np.int64)
