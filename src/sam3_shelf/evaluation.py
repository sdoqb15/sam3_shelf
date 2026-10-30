"""Evaluate complete instances and export real candidates for the static viewer."""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import DEFAULT_PROMPT
from .data import load_mask, read_manifest, validate_manifest
from .inference import predict, deployment_defaults
from .io import append_jsonl, file_sha256, write_json, read_json, object_sha256
from .metrics import aggregate, at_threshold, candidate_matrices
from .rle import encode_rle
from .visualization import overlay

TOP_NS = (3, 5, 10, 50)


def score_masks(gt_masks, masks, scores, threshold=0.1):
    iou, coverage, purity = candidate_matrices(gt_masks, masks)
    metrics = at_threshold(iou, coverage, purity, scores, threshold)
    top = {
        str(n): at_threshold(iou[:, :n], coverage[:, :n], purity[:, :n], scores[:n], threshold)
        for n in TOP_NS
    }
    complete = (coverage >= 0.95) & (purity >= 0.95) & (np.asarray(scores) > threshold)[None, :]
    ranks = [int(np.flatnonzero(row)[0]) + 1 if row.any() else None for row in complete]
    return metrics, top, ranks


def evaluate_model(model, processor, rows, threshold=0.1, nms_iou=0.98):
    """In-memory validation with the same inference path as deployment."""
    samples, top = [], {str(n): [] for n in TOP_NS}
    for row in rows:
        with Image.open(row["image"]) as image:
            masks, scores = predict(model, processor, image, threshold, nms_iou)
        gt = [load_mask(item["mask"]) for item in row["instances"]]
        metrics, prefixes, ranks = score_masks(gt, masks, scores, threshold)
        for n in top:
            top[n].append(prefixes[n])
        samples.append(dict(id=row["id"], first_complete_ranks=ranks, **metrics))
    return dict(
        **aggregate(samples),
        top_n_metrics={n: aggregate(v) for n, v in top.items()},
        samples=samples,
    )


def prediction_packet(image, masks, scores, threshold=0.1, nms_iou=0.98, gt=None):
    packet = dict(
        width=image.width,
        height=image.height,
        prompt=DEFAULT_PROMPT,
        score_threshold=threshold,
        mask_threshold=0.5,
        mask_nms_iou=nms_iou,
        scores=scores.tolist(),
        masks=[encode_rle(mask) for mask in masks],
    )
    if gt is not None:
        packet["gt_masks"] = [encode_rle(mask) for mask in gt]
    return packet


@deployment_defaults
def infer_image(
    model_path, image_path, output, device="cuda:0", threshold=0.1, nms_iou=0.98, save_masks=False
):
    from .model import load_model

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(model_path, device)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    masks, scores = predict(model, processor, image, threshold, nms_iou)
    packet = prediction_packet(image, masks, scores, threshold, nms_iou)
    write_json(output / "candidates.json", packet)
    image.save(output / "image.png")
    overlay(image, masks).save(output / "overlay.png")
    if save_masks:
        for i, mask in enumerate(masks):
            Image.fromarray(mask.astype(np.uint8) * 255).save(output / f"mask_{i:03d}.png")
    return dict(output=str(output), candidates=len(scores), prompt=DEFAULT_PROMPT)


def evaluation_identity(model_path, manifest, rows, threshold=0.1, nms_iou=0.98):
    package = Path(__file__).parent
    files = (
        "model.py",
        "inference.py",
        "postprocessing.py",
        "metrics.py",
        "evaluation.py",
        "data.py",
    )
    return dict(
        model_sha256=file_sha256(Path(model_path) / "model.safetensors"),
        manifest_sha256=file_sha256(manifest),
        data_fingerprint=object_sha256(
            [
                dict(
                    id=r["id"],
                    rgb=r["image_fingerprint"],
                    masks=[file_sha256(x["mask"]) for x in r.get("instances", [])],
                )
                for r in rows
            ]
        ),
        algorithm_sha256={name: file_sha256(package / name) for name in files},
        prompt=DEFAULT_PROMPT,
        score_threshold=threshold,
        mask_threshold=0.5,
        mask_nms_iou=nms_iou,
        runtime="deployment_defaults: deterministic=False, cudnn_benchmark=False",
    )


@deployment_defaults
def evaluate(
    model_path,
    manifest,
    output,
    device="cuda:0",
    threshold=0.1,
    nms_iou=0.98,
    data_root=None,
    label=None,
    resume=False,
):
    import torch
    from .model import load_model

    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Choose an empty evaluation output directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest, data_root)
    validate_manifest(rows)
    has_gt = ["instances" in row for row in rows]
    if any(has_gt) != all(has_gt):
        raise ValueError("Keep annotated and unannotated images in separate manifests")
    has_gt = all(has_gt)
    identity = evaluation_identity(model_path, manifest, rows, threshold, nms_iou)
    if (
        resume
        and output.exists()
        and any(output.iterdir())
        and not (output / "identity.json").exists()
    ):
        raise ValueError("Existing evaluation has no identity; choose a new directory")
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("Evaluation identity changed: model, manifest or postprocessing differs")
    if resume and (output / "summary.json").exists():
        return read_json(output / "summary.json")
    write_json(output / "identity.json", identity)
    model, processor = load_model(model_path, device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    samples, top = [], {str(n): [] for n in TOP_NS}
    reused = 0
    for index, row in enumerate(rows):
        destination = output / f"{index:05d}"
        if resume and (destination / "sample.json").exists():
            sample = read_json(destination / "sample.json")
            if sample["id"] != row["id"]:
                raise ValueError("Cached sample ordering changed")
            for name in ("image.png", "candidates.json", "overlay.png"):
                if not (destination / name).exists():
                    raise ValueError(f"Cached sample is incomplete: {destination / name}")
            if has_gt:
                for n in top:
                    top[n].append(sample["top_n_metrics"][n])
            samples.append(sample)
            reused += 1
            continue
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        masks, scores = predict(model, processor, image, threshold, nms_iou)
        gt = [load_mask(item["mask"]) for item in row["instances"]] if has_gt else None
        sample = dict(id=row["id"], index=index, candidates=len(scores))
        if has_gt:
            metrics, prefixes, ranks = score_masks(gt, masks, scores, threshold)
            sample.update(metrics, first_complete_ranks=ranks, top_n_metrics=prefixes)
            for n in top:
                top[n].append(prefixes[n])
        destination.mkdir(exist_ok=True)
        write_json(
            destination / "candidates.json",
            prediction_packet(image, masks, scores, threshold, nms_iou, gt),
        )
        image.save(destination / "image.png")
        overlay(image, masks).save(destination / "overlay.png")
        if has_gt:
            overlay(image, gt).save(destination / "gt.png")
        write_json(destination / "sample.json", sample)
        append_jsonl(output / "samples.jsonl", sample)
        samples.append(sample)
        print(f"{index + 1}/{len(rows)} {row['id']}: {len(scores)} candidates", flush=True)
    resources = dict(
        device=str(device),
        images=len(rows),
        reused_images=reused,
        newly_predicted_images=len(rows) - reused,
        wall_seconds=time.perf_counter() - start,
        scope="Inference, GT metrics, RLE and PNG export; not isolated inference latency",
    )
    if str(device).startswith("cuda"):
        resources.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    counts = [sample["candidates"] for sample in samples]
    summary = dict(
        label=label or Path(model_path).name,
        identity=identity,
        has_gt=has_gt,
        resources=resources,
        samples=samples,
        metrics=aggregate(samples)
        if has_gt
        else dict(images=len(rows), mean_candidates=float(np.mean(counts))),
        top_n_metrics={n: aggregate(v) for n, v in top.items()} if has_gt else None,
        metric_note="One-to-one complete-instance coverage/purity >=0.95; extra unlabelled instances are not penalized; not COCO AP",
    )
    write_json(output / "summary.json", summary)
    return summary
