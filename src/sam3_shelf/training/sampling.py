"""Weighted deterministic sampling; the stream is indexed by image exposure."""

import random


def weighted_rows(rows, start, end, seed=42, namespace="c2", multi_weight=8, single_weight=1):
    pool = [r for r in rows if r["sampling_group"] == "multi"] * multi_weight + [
        r for r in rows if r["sampling_group"] == "single"
    ] * single_weight
    if not pool or start < 0 or end < start:
        raise ValueError("Invalid weighted pool or exposure interval")
    if start == end:
        return []
    result = []
    for cycle in range(start // len(pool), (end - 1) // len(pool) + 1):
        block = pool.copy()
        random.Random(f"{seed}:{namespace}:{cycle}").shuffle(block)
        result.extend(
            block[max(0, start - cycle * len(pool)) : min(len(pool), end - cycle * len(pool))]
        )
    return result


def validation_value(metrics):
    return (
        metrics["top_n_metrics"]["5"]["complete_gt"]
        + metrics["top_n_metrics"]["10"]["complete_gt"],
        metrics["complete_gt"],
        metrics["successful_images"],
        -metrics["mean_candidates"],
    )
