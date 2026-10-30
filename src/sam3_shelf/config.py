"""JSON recipes with explicit paths and no hidden machine-specific defaults."""

from pathlib import Path

from . import DEFAULT_PROMPT
from .io import read_json


def load_recipe(path):
    path = Path(path).resolve()
    config = read_config_tree(path)
    if config.get("schema_version") != 1 or not isinstance(config.get("method"), str):
        raise ValueError("Expected schema_version=1 and a registered method name")
    for key in ("model", "output_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    if config.get("teacher_cache"):
        config["teacher_cache"] = str((path.parent / config["teacher_cache"]).resolve())
    if config.get("teacher_model"):
        config["teacher_model"] = str((path.parent / config["teacher_model"]).resolve())
    for item in config.get("evaluations", []):
        item["manifest"] = str((path.parent / item["manifest"]).resolve())
        if item["scope"] not in ("validation", "training_fit", "held_out_test", "unlabeled"):
            raise ValueError("Unknown evaluation scope")
    for key in ("multi_weight", "single_weight"):
        if not isinstance(config["sampling"][key], int) or config["sampling"][key] < 1:
            raise ValueError("Sampling weights must be positive integers")
    for key in ("train", "validation"):
        config["data"][key] = str((path.parent / config["data"][key]).resolve())
    if config["data"].get("root"):
        config["data"]["root"] = str((path.parent / config["data"]["root"]).resolve())
    if config.get("prompt", DEFAULT_PROMPT) != DEFAULT_PROMPT:
        raise ValueError(f"This recipe uses the fixed prompt: {DEFAULT_PROMPT}")
    train = config["training"]
    positive = ("gradient_accumulation", "validation_interval", "minimum_updates", "patience")
    if any(not isinstance(train[k], int) or train[k] < 1 for k in positive):
        raise ValueError(f"Positive integer training settings are required: {positive}")
    if train["max_updates"] is not None and train["max_updates"] < train["minimum_updates"]:
        raise ValueError("max_updates cannot be smaller than minimum_updates")
    teacher = config["teacher_policy"]
    if not 0 <= teacher["unknown_score_min"] <= teacher["selection_score_min"] <= 1:
        raise ValueError("Invalid teacher confidence bands")
    if config["inference"] != dict(score_threshold=0.1, mask_threshold=0.5, mask_nms_iou=0.98):
        raise ValueError("Reference recipes use score>0.1, mask0.5 and mask NMS0.98")
    return config


def read_config_tree(path, visited=()):
    """Inherit JSON settings with explicit deep overrides and cycle detection.

    Runtime paths are resolved against the experiment file, including inherited
    values. Keep base method files about methods and experiment files about data.
    """
    path = Path(path).resolve()
    if path in visited:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    child = read_json(path)
    parent = child.pop("extends", None)
    if parent is None:
        return child
    base = read_config_tree(path.parent / parent, (*visited, path))
    return deep_merge(base, child)


def deep_merge(base, overrides):
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
