"""Audit the public tree; this script never publishes or uploads anything."""

import argparse
import re
from pathlib import Path

IGNORE = {
    ".git",
    ".venv",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    "build",
    "dist",
    "data",
    "runs",
    "outputs",
    "reports",
    "cache",
    "checkpoints",
}
FORBIDDEN = {".pt", ".pth", ".safetensors", ".zip", ".gz", ".tar", ".jpg", ".jpeg", ".png", ".mp4"}
PATHS = re.compile(r"/" + r"(?:home|mnt|SharedFolder)/")
ADDRESS = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
TOKEN = re.compile(r"(?:hf_|ghp_)[A-Za-z0-9]{20,}")


def audit(root):
    problems = []
    checked = 0
    for path in Path(root).rglob("*"):
        relative = path.relative_to(root)
        if any(part in IGNORE or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            problems.append(f"{relative}: source symlink is not a standalone public file")
            continue
        if not path.is_file():
            continue
        checked += 1
        if path.suffix.lower() in FORBIDDEN:
            problems.append(f"{relative}: generated data/model/archive file")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{relative}: unexpected binary file")
            continue
        if PATHS.search(text):
            problems.append(f"{relative}: private absolute machine path")
        if TOKEN.search(text):
            problems.append(f"{relative}: credential-shaped token")
        for address in ADDRESS.findall(text):
            if address not in {"127.0.0.1", "0.0.0.0"}:
                problems.append(f"{relative}: non-loopback numeric address {address}")
    return dict(files_checked=checked, problems=problems, license_status="pending")


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["problems"]:
        raise SystemExit(1)
