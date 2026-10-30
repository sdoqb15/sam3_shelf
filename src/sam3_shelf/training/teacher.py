"""Frozen original-model references, cached by checkpoint, policy, RGB and GT."""

import fcntl
import time
from pathlib import Path

from ..io import object_sha256, write_json


def cache_directory(config, audit):
    base = Path(config.get("teacher_cache") or Path(config["output_dir"]) / "teacher")
    namespace = object_sha256(
        dict(
            model=audit["identity"].get("teacher_sha256", audit["identity"]["model_sha256"]),
            policy=config["teacher_policy"],
        )
    )
    return base / namespace


def empty_teacher():
    import torch

    return dict(
        class_logits=torch.empty(0),
        mask_logits=torch.empty((0, 1, 1)),
        boxes=torch.empty((0, 4)),
        protected=torch.empty(0, dtype=torch.bool),
    )


def prepare_teacher(config, rows, audit, device="cuda:0"):
    import numpy as np
    import torch
    from PIL import Image
    from .. import DEFAULT_PROMPT
    from ..data import load_mask
    from ..metrics import candidate_matrices
    from ..model import load_model
    from ..postprocessing import mask_nms, postprocess

    if not str(device).startswith("cuda"):
        raise ValueError("The reference teacher extraction requires a CUDA device")
    torch.cuda.set_device(device)
    cache = cache_directory(config, audit)
    cache.mkdir(parents=True, exist_ok=True)
    policy = config["teacher_policy"]
    with (cache / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing = [r for r in rows if not (cache / (r["_cache_key"] + ".pt")).exists()]
        if not missing:
            return dict(cache=str(cache), reused=len(rows), extracted=0)
        model, processor = load_model(config.get("teacher_model") or config["model"], device)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for i, row in enumerate(missing):
            with Image.open(row["image"]) as image:
                inputs = processor(
                    images=image.convert("RGB"), text=DEFAULT_PROMPT, return_tensors="pt"
                ).to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**inputs)
                native = postprocess(processor, output, 0.0, [(row["height"], row["width"])])[0]
            masks = native["masks"].bool().cpu().numpy()
            scores = native["scores"].float().cpu().numpy()
            if len(scores) != output.pred_masks.shape[1]:
                raise RuntimeError("Teacher extraction requires all raw query scores to be nonzero")
            _, coverage, purity = candidate_matrices(
                [load_mask(x["mask"]) for x in row["instances"]], masks
            )
            safe = (purity.sum(0) <= policy["max_union_gt_intersection_over_teacher_area"]) & (
                coverage.max(0) <= policy["max_teacher_intersection_over_each_gt_area"]
            )
            selected = np.flatnonzero(
                safe
                & (scores >= policy["unknown_score_min"])
                & masks.reshape(len(masks), -1).any(1)
            )
            selected = selected[mask_nms(masks[selected], scores[selected], policy["mask_nms_iou"])]
            protected = scores[selected] >= policy["selection_score_min"]
            protected[np.flatnonzero(protected)[policy["max_training_teacher_candidates"] :]] = (
                False
            )
            record = dict(
                id=row["id"],
                cache_key=row["_cache_key"],
                class_logits=output.pred_logits[0, selected].float().cpu(),
                mask_logits=output.pred_masks[0, selected].cpu(),
                boxes=output.pred_boxes[0, selected].float().cpu(),
                scores=torch.from_numpy(scores[selected]),
                protected=torch.from_numpy(protected),
            )
            path = cache / (row["_cache_key"] + ".pt")
            temporary = path.with_suffix(".tmp")
            torch.save(record, temporary)
            temporary.replace(path)
            print(f"Teacher {i + 1}/{len(missing)}: {row['id']}", flush=True)
            del inputs, output, native, record, masks
        resources = dict(
            device=device,
            gpu_count=1,
            extracted=len(missing),
            reused=len(rows) - len(missing),
            wall_seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        write_json(Path(config["output_dir"]) / "teacher_resources.json", resources)
        return resources
