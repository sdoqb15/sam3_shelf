"""Portable JSONL manifests, alpha-aware masks, and split-overlap checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path):
    with Image.open(path) as image:
        if "A" in image.getbands() or "transparency" in image.info:
            return np.asarray(image.convert("RGBA"))[..., 3] > 127
        values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., :3].max(-1)
    return values > 0


def read_manifest(path, data_root=None):
    """Resolve relative image/mask paths against data_root or the JSONL directory."""
    path = Path(path).resolve()
    root = Path(data_root).resolve() if data_root else path.parent
    rows = []
    identifiers = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"{path}:{number}: a nonempty string id is required")
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        identifiers.add(row["id"])
        if not isinstance(row.get("image"), str) or not row["image"]:
            raise ValueError(f"{row['id']}: a nonempty image path is required")
        if "instances" in row and not isinstance(row["instances"], list):
            raise ValueError(
                f"{row['id']}: instances must be a list, or omitted for unannotated data"
            )
        row["image"] = str((root / row["image"]).resolve())
        if "instances" in row:
            row["instances"] = [
                dict(item, mask=str((root / item["mask"]).resolve())) for item in row["instances"]
            ]
        row["group"] = row.get("group") or row.get("split_group") or row.get("scene_group")
        if row["group"] is not None and not isinstance(row["group"], str):
            raise ValueError(f"{row['id']}: group must be a string")
        row["sampling_group"] = row.get("sampling_group") or (
            "single" if len(row.get("instances", [])) == 1 else "multi"
        )
        if row["sampling_group"] not in ("single", "multi"):
            raise ValueError(f"{row['id']}: sampling_group must be single or multi")
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def image_fingerprint(image):
    values = np.asarray(image.convert("RGB"))
    digest = hashlib.sha256(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def validate_manifest(rows, require_gt=False, require_group=False):
    """Validate native pixels without altering annotations or splitting components."""
    images = set()
    groups = set()
    gt_count = 0
    for row in rows:
        with Image.open(row["image"]) as image:
            width, height = image.size
            fingerprint = image_fingerprint(image)
        if (row.get("width", width), row.get("height", height)) != (width, height):
            raise ValueError(f"{row['id']}: declared and actual image dimensions differ")
        row.update(width=width, height=height, image_fingerprint=fingerprint)
        if require_gt and "instances" not in row:
            raise ValueError(f"{row['id']}: instances are required; [] means annotated empty")
        if require_group and not row["group"]:
            raise ValueError(f"{row['id']}: group is required for train/validation separation")
        for instance in row.get("instances", []):
            mask = load_mask(instance["mask"])
            if mask.shape != (height, width) or not mask.any():
                raise ValueError(f"{row['id']}: mask must be nonempty and match native dimensions")
        gt_count += len(row.get("instances", []))
        images.add(fingerprint)
        if row["group"]:
            groups.add(row["group"])
    return dict(
        images=len(rows),
        gt=gt_count,
        unique_rgb=len(images),
        groups=len(groups),
        missing_groups=sum(not r["group"] for r in rows),
    )


def assert_disjoint(first, second):
    """Reject shared IDs, decoded RGBs, and declared capture groups."""
    for name, left, right in (
        ("id", {r["id"] for r in first}, {r["id"] for r in second}),
        ("RGB", {r["image_fingerprint"] for r in first}, {r["image_fingerprint"] for r in second}),
        (
            "group",
            {r["group"] for r in first if r["group"]},
            {r["group"] for r in second if r["group"]},
        ),
    ):
        overlap = left & right
        if overlap:
            raise ValueError(f"Train/validation {name} overlap: {sorted(overlap)[:5]}")
