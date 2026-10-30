"""Strict local Hugging Face checkpoint loading and explicit training scopes."""

import copy
from pathlib import Path

from .io import read_json


def load_model(path, device="cuda:0", dtype=None):
    import torch
    from transformers import Sam3Model, Sam3Processor, Sam3VideoModel

    path = Path(path)
    config = read_json(path / "config.json")
    video = "Sam3VideoModel" in config.get("architectures", [])
    cls = Sam3VideoModel if video else Sam3Model
    dtype = dtype or (torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
    checkpoint, loading = cls.from_pretrained(
        str(path), dtype=dtype, local_files_only=True, output_loading_info=True
    )
    errors = {
        k: loading[k]
        for k in ("missing_keys", "mismatched_keys", "unexpected_keys")
        if loading.get(k)
    }
    if errors:
        raise RuntimeError(f"Checkpoint did not load exactly: {errors}")
    model = checkpoint.detector_model if video else checkpoint
    model = model.to(device).eval()
    processor = Sam3Processor.from_pretrained(str(path), local_files_only=True)
    return model, processor


def configure_joint(model):
    """Freeze vision/text/geometry; update DETR, mask decoder, and score head."""
    names = ("detr_encoder.", "detr_decoder.", "mask_decoder.", "dot_product_scoring.")
    return configure_modules(model, names, expected_parameters=24_529_462)


def configure_modules(model, prefixes, expected_parameters=None):
    """Reusable scope helper for new method plugins; no implicit LoRA injection."""
    import torch

    trainable = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(tuple(prefixes))
        parameter.requires_grad_(enabled)
        if enabled:
            parameter.data = parameter.data.to(torch.float32)
            trainable.append(name)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not count or (expected_parameters is not None and count != expected_parameters):
        raise ValueError(
            f"Unexpected trainable parameter count: {count}; expected {expected_parameters}"
        )
    model.eval()
    return dict(parameters=count, names=trainable)


def deployment_copy(model):
    """Cast floating parameters only, preserving FP32 positional-encoding buffers."""
    import torch

    result = copy.deepcopy(model)
    if hasattr(result, "merge_and_unload"):
        result = result.merge_and_unload(safe_merge=True)
    for parameter in result.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(torch.bfloat16)
    return result.eval()
