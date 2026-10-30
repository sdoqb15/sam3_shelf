"""Decoded-image inference timing, separate from GT, PNG/RLE export and model load."""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .data import read_manifest
from .inference import deployment_defaults
from . import DEFAULT_PROMPT
from .postprocessing import postprocess, mask_nms
from .io import file_sha256, write_json
from .model import load_model


@deployment_defaults
def benchmark(model_path, manifest, output, device="cuda:0", data_root=None, warmup=5):
    import torch

    if not str(device).startswith("cuda"):
        raise ValueError("CUDA is required for GPU-event timing")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    rows = read_manifest(manifest, data_root)
    images = []
    for row in rows:
        with Image.open(row["image"]) as image:
            images.append(image.convert("RGB"))
    model, processor = load_model(model_path, device)

    @torch.inference_mode()
    def infer(image):
        start = time.perf_counter()
        inputs = processor(images=image, text=DEFAULT_PROMPT, return_tensors="pt").to(device)
        torch.cuda.synchronize()
        pre = time.perf_counter()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs)
        end.record()
        torch.cuda.synchronize()
        forward = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = postprocess(processor, output, 0.1, [(image.height, image.width)])[0]
        masks = result["masks"].bool().cpu().numpy()
        scores = result["scores"].float().cpu().numpy()
        if len(masks):
            keep = masks.reshape(len(masks), -1).any(1)
            masks, scores = masks[keep], scores[keep]
        retained = mask_nms(masks, scores, 0.98)
        torch.cuda.synchronize()
        finish = time.perf_counter()
        return dict(
            preprocess_h2d_ms=(pre - start) * 1000,
            forward_gpu_ms=begin.elapsed_time(end),
            postprocess_d2h_nms_ms=(finish - forward) * 1000,
            end_to_end_ms=(finish - start) * 1000,
            candidates=len(retained),
        )

    for i in range(warmup):
        infer(images[i % len(images)])
    torch.cuda.reset_peak_memory_stats()
    records = []
    for row, image in zip(rows, images):
        records.append(dict(id=row["id"], **infer(image)))
    timings = [r["end_to_end_ms"] for r in records]
    stages = {}
    for key in ("preprocess_h2d_ms", "forward_gpu_ms", "postprocess_d2h_nms_ms", "end_to_end_ms"):
        values = [r[key] for r in records]
        stages[key] = dict(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            p95=float(np.quantile(values, 0.95)),
        )
    result = dict(
        device=device,
        gpu_count=1,
        gpu=torch.cuda.get_device_name(device),
        batch_size=1,
        warmup=warmup,
        model_sha256=file_sha256(Path(model_path) / "model.safetensors"),
        manifest_sha256=file_sha256(manifest),
        images=len(rows),
        latency=stages,
        latency_ms=dict(
            mean=float(np.mean(timings)),
            median=float(np.median(timings)),
            p95=float(np.quantile(timings, 0.95)),
        ),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        per_image=records,
        scope="Decoded RGB to final masks: preprocessing, forward, restore, D2H, NMS; excludes IO, load, warmup, GT and export",
    )
    write_json(output, result)
    return result
