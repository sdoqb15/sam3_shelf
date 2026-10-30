"""One-to-one complete-instance recall, coverage, purity, and score-prefix metrics."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_pairs(iou, threshold=0.5):
    (g, p) = iou.shape
    if g and p:
        reward = (iou >= threshold).astype(float) + iou / (min(g, p) + 1)
        (gi, pi) = linear_sum_assignment(-reward)
        valid = iou[gi, pi] >= threshold
        return (gi[valid], pi[valid])
    else:
        return (np.array([], dtype=int), np.array([], dtype=int))


def matching_metrics(iou, threshold=0.5):
    (g, p) = iou.shape
    (gi, pi) = match_pairs(iou, threshold)
    matched = iou[gi, pi]
    tp = len(matched)
    (fp, fn) = (p - tp, g - tp)
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        gt=g,
        pred=p,
        matched_iou_sum=float(matched.sum()),
        gt_normalized_iou=float(matched.sum() / g) if g else None,
        count_abs_error=abs(p - g),
    )


def candidate_matrices(gt, pred):
    ga = np.array([m.sum() for m in gt], dtype=np.float64)
    pa = (
        pred.reshape(len(pred), -1).sum(1, dtype=np.int64).astype(np.float64)
        if len(pred)
        else np.zeros(0)
    )
    inter = np.array([pred[:, m].sum(1, dtype=np.int64) for m in gt], dtype=np.float64).reshape(
        len(gt), len(pred)
    )
    coverage = inter / np.maximum(ga[:, None], 1)
    purity = inter / np.maximum(pa[None, :], 1)
    iou = inter / np.maximum(ga[:, None] + pa[None, :] - inter, 1)
    return (iou, coverage, purity)


def at_threshold(iou, coverage, purity, scores, threshold, quality=0.95):
    keep = np.asarray(scores) > threshold
    valid = (coverage[:, keep] >= quality) & (purity[:, keep] >= quality)
    matched = matching_metrics(np.where(valid, iou[:, keep], 0))
    return dict(
        complete_gt=matched["tp"],
        gt=iou.shape[0],
        candidates=int(keep.sum()),
        image_success=matched["tp"] == iou.shape[0],
    )


def aggregate(items):
    gt = sum((x["gt"] for x in items))
    tp = sum((x["complete_gt"] for x in items))
    counts = [x["candidates"] for x in items]
    return dict(
        complete_gt=tp,
        gt=gt,
        complete_recall=tp / max(gt, 1),
        successful_images=sum((x["image_success"] for x in items)),
        images=len(items),
        image_success_rate=sum((x["image_success"] for x in items)) / max(len(items), 1),
        mean_candidates=float(np.mean(counts)) if counts else 0.0,
        p95_candidates=float(np.quantile(counts, 0.95)) if counts else 0.0,
        max_candidates=max(counts, default=0),
    )
