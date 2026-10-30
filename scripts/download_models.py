"""Explicit, revision-pinned download of the associated HF checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "operating_point.json",
    "LICENSE",
    "README.md",
    "training_summary.json",
    "release_manifest.json",
    "SHA256SUMS",
]


def download(name, destination):
    from huggingface_hub import snapshot_download

    registry = json.loads((ROOT / "configs/checkpoints/huggingface.json").read_text())
    spec = registry["models"][name]
    if not spec["revision"]:
        raise ValueError("This release is not yet published and pinned")
    destination = Path(destination).resolve()
    identity = {k: spec[k] for k in ("repo_id", "revision", "weights_sha256")}
    receipt = destination / "download_receipt.json"
    if destination.exists() and any(destination.iterdir()):
        if not receipt.exists() or json.loads(receipt.read_text()).get("identity") != identity:
            raise ValueError("Choose an empty destination or resume the same registered download")
    destination.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(dict(status="downloading", identity=identity), indent=2) + "\n")
    snapshot_download(
        spec["repo_id"], revision=spec["revision"], local_dir=destination, allow_patterns=FILES
    )
    digest = hashlib.sha256()
    with (destination / "model.safetensors").open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != spec["weights_sha256"]:
        raise ValueError("Downloaded model SHA256 does not match the registered release")
    receipt.write_text(json.dumps(dict(status="verified", identity=identity), indent=2) + "\n")
    print(f"Verified {name}: {destination}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("c7a", "d2-2261"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    download(args.model, args.output)
