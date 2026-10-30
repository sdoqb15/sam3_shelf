"""Public command line; heavyweight ML dependencies load only for ML commands."""

import argparse
import importlib.metadata
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(prog="sam3-shelf", description="SAM3 shelf-instance experiments")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Show dependency and CUDA availability")
    data = sub.add_parser("validate-data", help="Check masks, dimensions, IDs and split overlap")
    data.add_argument("--manifest", type=Path, required=True)
    data.add_argument("--compare", type=Path)
    data.add_argument("--data-root", type=Path)
    for name in ("train", "prepare-teacher", "run"):
        command = sub.add_parser(
            name,
            help={
                "train": "Train one method with shared infrastructure",
                "prepare-teacher": "Cache GT-safe original-model references",
                "run": "Train and evaluate configured datasets",
            }[name],
        )
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument(
            "--check-only",
            action="store_true",
            help="Validate config and data without model loading or training",
        )
        if name != "prepare-teacher":
            command.add_argument("--resume", action="store_true")
    for name in ("infer", "evaluate", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
        if name == "infer":
            command.add_argument("--image", type=Path, required=True)
            command.add_argument("--save-masks", action="store_true")
        else:
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--data-root", type=Path)
        if name != "benchmark":
            command.add_argument("--score-threshold", type=float, default=0.1)
            command.add_argument("--nms-iou", type=float, default=0.98)
        if name == "evaluate":
            command.add_argument("--resume", action="store_true")
    review = sub.add_parser("review", help="Build a static actual-mask/TOP-N comparison page")
    review.add_argument("--evaluation", action="append", required=True, metavar="LABEL=DIR")
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--copy-assets", action="store_true")
    serve = sub.add_parser("serve", help="Serve a report directory")
    serve.add_argument("--directory", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8836)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        result = {}
        for name in ("torch", "transformers", "numpy", "scipy", "Pillow", "safetensors"):
            try:
                result[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result[name] = "not installed"
        try:
            import torch

            result["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            result["cuda_available"] = False
    elif args.command == "validate-data":
        from .data import read_manifest, validate_manifest, assert_disjoint

        rows = read_manifest(args.manifest, args.data_root)
        result = validate_manifest(rows)
        if args.compare:
            other = read_manifest(args.compare, args.data_root)
            validate_manifest(other)
            assert_disjoint(rows, other)
            result["comparison"] = "disjoint IDs, RGB and declared groups"
    elif args.command in ("train", "prepare-teacher", "run"):
        from .training.preflight import preflight

        config, rows, _, audit = preflight(args.config)
        if args.check_only:
            result = audit
        elif args.command == "prepare-teacher":
            from .methods import get_method

            method = get_method(config)
            result = (
                method.prepare_teacher(config, rows, audit, args.device)
                if method.needs_teacher
                else dict(teacher_required=False)
            )
        elif args.command == "train":
            from .training.trainer import train

            result = train(args.config, args.device, args.resume)
        else:
            from .pipeline import run_experiment

            result = run_experiment(args.config, args.device, args.resume)
    elif args.command == "infer":
        from .evaluation import infer_image

        result = infer_image(
            args.model,
            args.image,
            args.output,
            args.device,
            args.score_threshold,
            args.nms_iou,
            args.save_masks,
        )
    elif args.command == "evaluate":
        from .evaluation import evaluate

        result = evaluate(
            args.model,
            args.manifest,
            args.output,
            args.device,
            args.score_threshold,
            args.nms_iou,
            args.data_root,
            resume=args.resume,
        )
        result = {k: v for k, v in result.items() if k != "samples"}
    elif args.command == "benchmark":
        from .benchmark import benchmark

        result = benchmark(args.model, args.manifest, args.output, args.device, args.data_root)
        result = {k: v for k, v in result.items() if k != "per_image"}
    elif args.command == "review":
        from .report import build_report

        entries = []
        for value in args.evaluation:
            label, separator, path = value.partition("=")
            if not separator or not label or not path:
                raise ValueError("Use --evaluation LABEL=DIRECTORY")
            entries.append((label, Path(path)))
        result = build_report(entries, args.output, args.copy_assets)
    else:
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        handler = partial(SimpleHTTPRequestHandler, directory=str(args.directory.resolve()))
        print(f"Report: http://{args.host}:{args.port}/", flush=True)
        ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))
