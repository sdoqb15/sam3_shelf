"""Geometry and local-ROI losses extracted from the validated joint recipe."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def generalized_iou(a, b):
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = (rb - lt).clamp(min=0).prod(-1)
    aa = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    ba = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    union = (aa[:, None] + ba[None, :] - intersection).clamp(min=1e-06)
    clt = torch.minimum(a[:, None, :2], b[None, :, :2])
    crb = torch.maximum(a[:, None, 2:], b[None, :, 2:])
    enclosure = (crb - clt).clamp(min=0).prod(-1).clamp(min=1e-06)
    return intersection / union - (enclosure - union) / enclosure


def target_box(m):
    (y, x) = torch.where(m > 0.5)
    (h, w) = m.shape
    return torch.stack([x.min() / w, y.min() / h, (x.max() + 1) / w, (y.max() + 1) / h])


def dice(logits, target):
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    return (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()


def mask_first_cost(cost_bce, cost_dice, class_probability, box_l1, box_giou):
    return (
        2 * cost_bce
        + 10 * cost_dice
        - 0.25 * class_probability[:, None]
        + 2 * box_l1
        - 2 * box_giou
    )


def local_mask_loss(logits, targets, boxes, size=128, margin=0.2, cpu_backward=False):
    """Balanced BCE in native-GT-derived crops, differentiable in mask logits.

    Coordinates cover the GT box plus a margin. Bilinear sampling uses the same
    normalized image coordinates for the low-resolution logits and native GT.
    Normalizing positives separately prevents small or thin packaging from
    being overwhelmed by the rest of the shelf. Nearby negatives penalize spill.
    """
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0.0001)
    lo = (boxes[:, :2] - margin * wh).clamp(0, 1)
    hi = (boxes[:, 2:] + margin * wh).clamp(0, 1)
    axis = (torch.arange(size, device=logits.device, dtype=torch.float32) + 0.5) / size
    xx = lo[:, 0, None] + (hi[:, 0] - lo[:, 0])[:, None] * axis
    yy = lo[:, 1, None] + (hi[:, 1] - lo[:, 1])[:, None] * axis
    grid = (
        torch.stack(
            (xx[:, None, :].expand(-1, size, -1), yy[:, :, None].expand(-1, -1, size)), dim=-1
        )
        * 2
        - 1
    )
    if cpu_backward:
        pl = F.grid_sample(logits[:, None].cpu(), grid.cpu(), align_corners=False)[:, 0].to(
            logits.device
        )
    else:
        pl = F.grid_sample(logits[:, None], grid, align_corners=False)[:, 0]
    gt = F.grid_sample(targets[:, None], grid, align_corners=False)[:, 0].clamp(0, 1)
    positive = (F.softplus(-pl) * gt).sum((1, 2)) / gt.sum((1, 2)).clamp(min=1)
    negative = (F.softplus(pl) * (1 - gt)).sum((1, 2)) / (1 - gt).sum((1, 2)).clamp(min=1)
    return (positive.mean(), negative.mean())


def sampled_cost(logits, target, points):
    p = logits.flatten(1)[:, points]
    t = target.flatten(1)[:, points]
    s = p.sigmoid()
    bce = (F.softplus(p).sum(1)[:, None] - p @ t.T) / max(len(points), 1)
    dl = 1 - (2 * s @ t.T + 1) / (s.sum(1)[:, None] + t.sum(1)[None] + 1)
    return (bce, dl)


def masks_iou(a, b):
    with torch.autocast(a.device.type, enabled=False):
        a = a.flatten(1).float()
        b = b.flatten(1).float()
        inter = a @ b.T
        return inter / (a.sum(1)[:, None] + b.sum(1)[None] - inter).clamp(min=1)


def geometric_pairs(iou, minimum):
    if not iou.numel():
        return (np.array([], dtype=int), np.array([], dtype=int))
    a = iou.detach().cpu().numpy()
    valid = a >= minimum
    (rows, cols) = linear_sum_assignment(-(valid.astype(float) * (min(a.shape) + 1) + a))
    keep = valid[rows, cols]
    return (rows[keep], cols[keep])
