"""Joint instance supervision with GT-safe teacher/unknown exemptions.

The arithmetic is preserved from the validated C7-A joint objective.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .geometry import (
    target_box,
    generalized_iou,
    dice,
    mask_first_cost,
    local_mask_loss,
    sampled_cost,
    masks_iou,
    geometric_pairs,
)


def focal_with_unknown(z, positive, ignored, ng):
    target = torch.zeros_like(z)
    target[positive] = 1
    active = torch.ones_like(z, dtype=torch.bool)
    active[ignored] = False
    active[positive] = True
    prob = z.sigmoid()
    pt = prob * target + (1 - prob) * (1 - target)
    alpha = 0.25 * target + 0.75 * (1 - target)
    loss = (
        alpha * (1 - pt).square() * F.binary_cross_entropy_with_logits(z, target, reduction="none")
    )[active].sum() / max(ng, 1)
    return loss


def joint_loss(output, targets, teacher, c):
    pm = output.pred_masks[0].float()
    z = output.pred_logits[0].float()
    boxes = output.pred_boxes[0].float()
    device = pm.device
    gt = torch.stack([m.to(device) for m in targets])
    ng = len(gt)
    assert 0 < ng <= len(pm)
    low = F.interpolate(gt[:, None], size=pm.shape[-2:], mode="area")[:, 0]
    gb = torch.stack([target_box(m) for m in gt])
    with torch.no_grad():
        points = torch.randperm(pm.shape[-2] * pm.shape[-1], device=device)[:4096]
        (cb, cd) = sampled_cost(pm, low, points)
        cost = mask_first_cost(
            cb, cd, z.sigmoid(), torch.cdist(boxes, gb, p=1), generalized_iou(boxes, gb)
        )
        (qi, gi) = linear_sum_assignment(cost.cpu().numpy())
    qi = torch.as_tensor(qi, device=device)
    gi = torch.as_tensor(gi, device=device)
    bce = F.binary_cross_entropy_with_logits(pm[qi], low[gi])
    dl = dice(pm[qi], low[gi])
    (lp, ln) = local_mask_loss(pm[qi], gt[gi], gb[gi], cpu_backward=True)
    box = F.l1_loss(boxes[qi], gb[gi])
    giou = (1 - generalized_iou(boxes[qi], gb[gi]).diag()).mean()
    presence = F.binary_cross_entropy_with_logits(
        output.presence_logits[0].float(), torch.ones_like(output.presence_logits[0].float())
    )
    ignored = []
    protected_q = []
    protected_t = []
    if len(teacher["class_logits"]):
        soft = teacher["mask_logits"].to(device).float().sigmoid()
        protected = teacher["protected"].to(device).bool()
        with torch.no_grad():
            available = torch.ones(len(pm), dtype=torch.bool, device=device)
            available[qi] = False
            for is_protected, group in ((True, protected), (False, ~protected)):
                refs = group.nonzero().flatten()
                candidates = available.nonzero().flatten()
                if not len(refs) or not len(candidates):
                    continue
                overlap = masks_iou(soft[refs] > 0.5, pm[candidates].sigmoid() > 0.5)
                possible = (overlap >= c["teacher_policy"]["geometry_match_min_iou"]).any(0)
                candidates = candidates[possible]
                overlap = overlap[:, possible]
                if not len(candidates):
                    continue
                native = (
                    F.interpolate(
                        output.pred_masks[0, candidates].detach().sigmoid()[None],
                        size=gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    > 0.5
                )
                with torch.autocast(device_type=device.type, enabled=False):
                    inter = gt.flatten(1).float() @ native.flatten(1).float().T
                candidate_limit = c["teacher_policy"].get(
                    "max_union_gt_intersection_over_teacher_area", 0.02
                )
                gt_limit = c["teacher_policy"].get(
                    "max_teacher_intersection_over_each_gt_area", 0.02
                )
                safe = (inter.sum(0) / native.sum((1, 2)).clamp(min=1) <= candidate_limit) & (
                    inter / gt.sum((1, 2))[:, None].clamp(min=1) <= gt_limit
                ).all(0)
                candidates = candidates[safe]
                overlap = overlap[:, safe]
                if not len(candidates):
                    continue
                (a, b) = geometric_pairs(overlap, c["teacher_policy"]["geometry_match_min_iou"])
                tq = refs[torch.as_tensor(a, device=device)]
                sq = candidates[torch.as_tensor(b, device=device)]
                available[sq] = False
                ignored.extend(sq.tolist())
                if is_protected:
                    protected_q.extend(sq.tolist())
                    protected_t.extend(tq.tolist())
    focal = focal_with_unknown(z, qi, ignored, ng)
    loss = 2 * bce + 2 * dl + 2 * focal + lp + ln + 5 * box + 2 * giou + presence
    keep = z.sum() * 0
    if protected_q:
        sq = torch.as_tensor(protected_q, device=device)
        tq = torch.as_tensor(protected_t, device=device)
        tb = teacher["boxes"].to(device).float()[tq]
        tz = teacher["class_logits"].to(device).float()[tq]
        keep = (
            2 * F.binary_cross_entropy_with_logits(pm[sq], soft[tq])
            + 2 * dice(pm[sq], soft[tq])
            + 5 * F.l1_loss(boxes[sq], tb)
            + 2 * (1 - generalized_iou(boxes[sq], tb).diag()).mean()
            + F.binary_cross_entropy_with_logits(z[sq], tz.sigmoid())
        )
        loss = loss + c["teacher_policy"]["loss_weight"] * keep
    return (
        loss,
        dict(
            gt=ng,
            focal=float(focal.detach()),
            mask_bce=float(bce.detach()),
            mask_dice=float(dl.detach()),
            teacher_available=int(teacher["protected"].sum()),
            teacher_matched=len(protected_q),
            unknown_matched=len(ignored) - len(protected_q),
            automatic_negative_queries=len(z) - ng - len(ignored),
            teacher_loss=float(keep.detach()),
        ),
    )
