"""Generate a small static multi-model viewer from evaluation artifacts."""

import re
import shutil
from importlib.resources import files
from pathlib import Path

from .io import read_json, write_json


def build_report(evaluations, output, copy_assets=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets = files("sam3_shelf") / "assets"
    for name in ("index.html", "viewer.js", "viewer.css"):
        (output / name).write_text((assets / name).read_text(encoding="utf-8"), encoding="utf-8")
    target = output / "models"
    target.mkdir(exist_ok=True)
    models = []
    identifiers = None
    for index, (label, path) in enumerate(evaluations):
        path = Path(path).resolve()
        summary = read_json(path / "summary.json")
        ids = [s["id"] for s in summary["samples"]]
        if identifiers is None:
            identifiers = ids
        elif ids != identifiers:
            raise ValueError("Compare models on the same ordered evaluation image IDs")
        name = f"{index:02d}-" + re.sub(r"[^a-zA-Z0-9_-]", "-", label).strip("-")[:40]
        destination = target / name
        if destination.exists() or destination.is_symlink():
            raise ValueError("Choose a new report directory")
        if copy_assets:
            shutil.copytree(path, destination)
        else:
            destination.symlink_to(path, target_is_directory=True)
        models.append(
            dict(
                label=label,
                path=f"models/{name}",
                metrics=summary["metrics"],
                top_n_metrics=summary["top_n_metrics"],
                has_gt=summary["has_gt"],
                samples=summary["samples"],
            )
        )
    write_json(output / "manifest.json", dict(models=models, ids=identifiers or []))
    return dict(
        output=str(output),
        models=len(models),
        images=len(identifiers or []),
        note="Serve this directory over HTTP; use --copy-assets for a standalone export",
    )
