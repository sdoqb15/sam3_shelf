"""Fixed-prompt inference without GT, SKU text, or private project state."""

from contextlib import nullcontext
from functools import wraps

import numpy as np

from . import DEFAULT_PROMPT
from .postprocessing import mask_nms, postprocess


def deployment_defaults(function):
    """Use the same deployment kernel settings before and after a training call."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        import torch

        previous = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        )
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        try:
            return function(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
            torch.backends.cudnn.deterministic = previous[2]
            torch.backends.cudnn.benchmark = previous[3]

    return wrapped


def predict(model, processor, image, score_threshold=0.1, nms_iou=0.98):
    import torch

    if not 0.1 <= score_threshold <= 1:
        raise ValueError("score_threshold must be between 0.1 and 1")
    if not 0 <= nms_iou <= 1:
        raise ValueError("nms_iou must be between 0 and 1")
    image = image.convert("RGB")
    device = model.device
    inputs = processor(images=image, text=DEFAULT_PROMPT, return_tensors="pt").to(device)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    model.eval()
    with torch.inference_mode(), autocast:
        outputs = model(**inputs)
        result = postprocess(processor, outputs, score_threshold, [(image.height, image.width)])[0]
    masks = result["masks"].bool().cpu().numpy()
    scores = result["scores"].float().cpu().numpy()
    if len(masks):
        nonempty = masks.reshape(len(masks), -1).any(1)
        masks, scores = masks[nonempty], scores[nonempty]
    if not len(masks):
        masks = np.zeros((0, image.height, image.width), dtype=bool)
    keep = mask_nms(masks, scores, nms_iou)
    return masks[keep], scores[keep]
