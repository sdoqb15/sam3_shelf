"""Validate experiments and dataset identities before allocating CUDA memory."""

import math
from pathlib import Path

from ..config import load_recipe
from ..data import assert_disjoint, read_manifest, validate_manifest
from ..io import file_sha256, object_sha256


def preflight(config_path):
    from ..methods import get_method

    config = load_recipe(config_path)
    method = get_method(config)
    data = config["data"]
    train = read_manifest(data["train"], data.get("root"))
    validation = read_manifest(data["validation"], data.get("root"))
    train_audit = validate_manifest(train, require_gt=True, require_group=True)
    val_audit = validate_manifest(validation, require_gt=True, require_group=True)
    assert_disjoint(train, validation)
    if any(not row["instances"] for row in train):
        raise ValueError("The reference joint loss requires at least one GT per training image")
    weighted = sum(config["sampling"][row["sampling_group"] + "_weight"] for row in train)
    if not weighted:
        raise ValueError("Weighted training pool is empty")
    settings = config["training"]
    interval = settings["validation_interval"]
    cycle_end = math.ceil(weighted / settings["gradient_accumulation"] / interval) * interval
    settings["effective_minimum_updates"] = (max(settings["minimum_updates"], cycle_end)
        if settings.get("require_full_cycle", True) else settings["minimum_updates"])
    if (
        settings["max_updates"] is not None
        and settings["max_updates"] < settings["effective_minimum_updates"]
    ):
        raise ValueError(
            "max_updates is smaller than the configured effective minimum"
        )
    model = Path(config["model"])
    required = (
        "config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete local model checkpoint: {missing}")
    fingerprints = {}
    for label, rows in (("train", train), ("validation", validation)):
        identities = []
        for row in rows:
            entry = dict(
                id=row["id"],
                group=row["group"],
                sampling_group=row["sampling_group"],
                rgb=row["image_fingerprint"],
                masks=[file_sha256(x["mask"]) for x in row["instances"]],
            )
            row["_cache_key"] = object_sha256(entry)
            identities.append(entry)
        fingerprints[label] = object_sha256(identities)
    portable_recipe = {
        k: v
        for k, v in config.items()
        if k not in ("model", "teacher_model", "output_dir", "data", "teacher_cache", "evaluations")
    }
    identity = dict(
        recipe=portable_recipe,
        model_sha256=file_sha256(model / "model.safetensors"),
        data_fingerprints=fingerprints,
    )
    if method.needs_teacher:
        teacher = Path(config.get("teacher_model") or config["model"])
        missing = [name for name in required if not (teacher / name).is_file()]
        if missing:
            raise ValueError(f"Incomplete teacher checkpoint: {missing}")
        identity["teacher_sha256"] = (
            file_sha256(teacher / "model.safetensors")
            if teacher != model
            else identity["model_sha256"]
        )
    audit = dict(
        train=train_audit,
        validation=val_audit,
        weighted_cycle_exposures=weighted,
        require_full_cycle=settings.get("require_full_cycle", True),
        effective_minimum_updates=settings["effective_minimum_updates"],
        identity=identity,
    )
    return config, train, validation, audit
