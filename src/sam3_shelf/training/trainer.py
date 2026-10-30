"""One resumable trainer shared by experiment configs and method plugins."""

import gc
import fcntl
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from .. import DEFAULT_PROMPT
from ..data import load_mask
from ..io import append_jsonl, file_sha256, read_json, write_json
from ..methods import get_method
from ..model import deployment_copy, load_model
from .preflight import preflight
from .sampling import weighted_rows


def training_rows(config, rows, start, end):
    return weighted_rows(
        rows,
        start,
        end,
        config["training"]["seed"],
        config["sampling"]["seed_namespace"],
        config["sampling"]["multi_weight"],
        config["sampling"]["single_weight"],
    )


def learning_rate(config, step):
    settings = config["optimizer"]
    duration = settings["schedule_updates"]
    if step >= duration:
        return settings["tail_lr"]
    warmup = min((step + 1) / max(duration * settings["warmup_fraction"], 1), 1)
    return settings["lr"] * warmup * 0.5 * (1 + math.cos(math.pi * step / max(duration, 1)))


def save_link(path, target):
    temporary = path.with_name(path.name + ".next")
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)
    temporary.replace(path)


def rng_state():
    import torch

    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def code_identity(method):
    import inspect

    package = Path(__file__).resolve().parents[1]
    names = (
        "training/trainer.py",
        "training/losses.py",
        "training/geometry.py",
        "training/sampling.py",
        "training/teacher.py",
        "training/preflight.py",
        "inference.py",
        "postprocessing.py",
        "metrics.py",
        "model.py",
        "data.py",
    )
    paths = {package / name for name in names}
    paths.update(
        Path(inspect.getfile(cls))
        for cls in type(method).__mro__
        if cls.__module__.startswith("sam3_shelf.")
    )
    return {str(p.relative_to(package)): file_sha256(p) for p in sorted(paths)}


def _train(config_path, device="cuda:0", resume=False):
    import torch
    from torch.utils.data import Dataset, DataLoader

    if not str(device).startswith("cuda"):
        raise ValueError(
            "The reference trainer requires CUDA; use train --check-only for CPU preflight"
        )
    config, rows, validation, audit = preflight(config_path)
    method = get_method(config)
    out = Path(config["output_dir"])
    identity = dict(**audit["identity"], code_sha256=code_identity(method))
    if config.get("method_plugin"):
        import importlib, inspect

        identity["plugin_sha256"] = file_sha256(
            inspect.getfile(importlib.import_module(config["method_plugin"]))
        )
    if (out / "identity.json").exists():
        if not resume:
            raise ValueError("Output already belongs to a run; use --resume or a new directory")
        if read_json(out / "identity.json") != identity:
            raise ValueError("Resume identity differs: config, code, model or data changed")
    else:
        if resume:
            raise ValueError("No existing run identity to resume")
        if out.exists():
            allowed = {
                "baseline",
                "pipeline.json",
                "teacher_resources.json",
                "teacher",
                ".train.lock",
                ".pipeline.lock",
            }
            unexpected = {p.name for p in out.iterdir()} - allowed
            if unexpected:
                raise ValueError(f"Output contains unrelated files: {sorted(unexpected)}")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "identity.json", identity)
    if (out / "complete.json").exists():
        if resume:
            return read_json(out / "complete.json")
        raise ValueError("Run already complete; choose a new experiment/output directory")
    torch.cuda.set_device(device)
    write_json(out / "resolved_config.json", config)
    write_json(out / "data_audit.json", audit)
    if method.needs_teacher:
        method.prepare_teacher(config, rows, audit, device)
    settings = config["training"]
    seed = settings["seed"]
    accumulation = settings["gradient_accumulation"]
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(settings["deterministic"], warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = settings["deterministic"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, processor = load_model(config["model"], device)
    model, scope = method.configure(model, config)
    write_json(out / "trainable_parameters.json", scope)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"]
    )

    class Images(Dataset):
        def __init__(self, selected):
            self.rows = selected

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            from PIL import Image

            row = self.rows[index]
            with Image.open(row["image"]) as im:
                inputs = processor(
                    images=im.convert("RGB"), text=DEFAULT_PROMPT, return_tensors="pt"
                )
            targets = [
                torch.from_numpy(load_mask(x["mask"]).copy()).float() for x in row["instances"]
            ]
            teacher = method.load_teacher(row, config, audit)
            return inputs, targets, teacher

    def collate(samples):
        return samples[0]

    step = stale = 0
    best = None
    resources = (
        read_json(out / "resources.json")
        if (out / "resources.json").exists()
        else dict(
            device=device,
            gpu_count=1,
            gpu=torch.cuda.get_device_name(device),
            cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
            active_training_seconds=0.0,
            validation_seconds=0.0,
            checkpoint_seconds=0.0,
            sessions=[],
            train_peak_allocated_bytes=0,
            train_peak_reserved_bytes=0,
            validation_peak_allocated_bytes=0,
            validation_peak_reserved_bytes=0,
        )
    )
    if resources["sessions"] and "finished" not in resources["sessions"][-1]:
        resources["sessions"][-1].update(
            finished=resources.get("last_updated", time.time()), interrupted=True
        )
    resources["sessions"].append(dict(started=time.time()))
    restored = resume and (out / "resume.pt").exists()
    if restored:
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step, stale, best = saved["step"], saved["stale"], saved["best"]
        restore_rng(saved["rng"])
        del saved

    def record_resources():
        resources.update(
            last_updated=time.time(), updates=step, image_exposures=step * accumulation
        )
        write_json(out / "resources.json", resources)

    def checkpoint():
        started = time.perf_counter()
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(),
                step=step,
                stale=stale,
                best=best,
            ),
            out / "resume.next.pt",
        )
        (out / "resume.next.pt").replace(out / "resume.pt")
        resources["checkpoint_seconds"] += time.perf_counter() - started
        record_resources()

    def validate():
        nonlocal best, stale
        random_state = rng_state()
        destination = out / "checkpoints" / f"step_{step:05d}"
        started = time.perf_counter()
        copied = deployment_copy(model)
        copied.save_pretrained(destination)
        processor.save_pretrained(destination)
        write_json(
            destination / "operating_point.json", dict(config["inference"], prompt=DEFAULT_PROMPT)
        )
        del copied
        gc.collect()
        torch.cuda.empty_cache()
        resources["checkpoint_seconds"] += time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loaded, proc = load_model(destination, device)
        metrics = method.validate(loaded, proc, validation, config)
        resources["validation_seconds"] += time.perf_counter() - started
        for kind, function in (
            ("allocated", torch.cuda.max_memory_allocated),
            ("reserved", torch.cuda.max_memory_reserved),
        ):
            key = f"validation_peak_{kind}_bytes"
            resources[key] = max(resources[key], function())
        del loaded, proc
        gc.collect()
        torch.cuda.empty_cache()
        restore_rng(random_state)
        improved = best is None or method.selection_value(metrics) > method.selection_value(best)
        if improved:
            best = dict(metrics, step=step)
            stale = 0
            write_json(out / "best_metrics.json", best)
            save_link(out / "best", destination)
        else:
            stale += 1
        record = dict(
            metrics,
            step=step,
            best_step=best["step"],
            improved=improved,
            non_improving_checks=stale,
        )
        write_json(out / f"validation_{step:05d}.json", record)
        append_jsonl(out / "validation.jsonl", record)
        write_json(
            out / "progress.json",
            dict(step=step, best_step=best["step"], non_improving_checks=stale),
        )
        print(
            f"Validation step={step}, best={best['step']}, TOP5/10="
            f"{metrics['top_n_metrics']['5']['complete_gt']}/{metrics['top_n_metrics']['10']['complete_gt']}",
            flush=True,
        )
        record_resources()

    if not restored:
        validate()
        checkpoint()
    limit = settings["max_updates"]
    while step < settings["effective_minimum_updates"] or stale < settings["patience"]:
        if limit is not None and step >= limit:
            break
        end = (step // settings["validation_interval"] + 1) * settings["validation_interval"]
        if limit is not None:
            end = min(end, limit)
        selected = training_rows(config, rows, step * accumulation, end * accumulation)
        loader = DataLoader(
            Images(selected),
            batch_size=1,
            num_workers=settings["workers"],
            shuffle=False,
            collate_fn=collate,
        )
        optimizer.zero_grad(set_to_none=True)
        micro = 0
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for inputs, targets, teacher in loader:
            lr = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
                if k in ("pixel_values", "input_ids", "attention_mask")
            }
            with torch.random.fork_rng(devices=[torch.device(device)]):
                torch.manual_seed(seed + step * accumulation + micro)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(**inputs)
                    loss, details = method.loss(output, targets, teacher, config)
                (loss / accumulation).backward()
            if not torch.isfinite(loss):
                raise FloatingPointError(details)
            micro += 1
            del output, inputs
            if micro < accumulation:
                continue
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, config["optimizer"]["clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            micro = 0
            if step % settings["log_interval"] == 0:
                append_jsonl(
                    out / "train.jsonl",
                    dict(
                        step=step,
                        loss=float(loss.detach()),
                        lr=lr,
                        gradient_norm=float(norm),
                        details=details,
                    ),
                )
        if step != end or micro:
            raise RuntimeError("Incomplete accumulated update interval")
        torch.cuda.synchronize()
        resources["active_training_seconds"] += time.perf_counter() - started
        for kind, function in (
            ("allocated", torch.cuda.max_memory_allocated),
            ("reserved", torch.cuda.max_memory_reserved),
        ):
            key = f"train_peak_{kind}_bytes"
            resources[key] = max(resources[key], function())
        del loader, loss
        gc.collect()
        torch.cuda.empty_cache()
        validate()
        checkpoint()
    save_link(out / "final", out / "checkpoints" / f"step_{step:05d}")
    resources["sessions"][-1]["finished"] = time.time()
    resources["session_wall_seconds"] = sum(
        s["finished"] - s["started"] for s in resources["sessions"]
    )
    record_resources()
    result = dict(
        status="complete",
        trained_updates=step,
        best_step=best["step"],
        best=str((out / "best").resolve()),
        final=str((out / "final").resolve()),
        non_improving_checks=stale,
        stop_reason="max_updates" if limit is not None and step >= limit else "validation_plateau",
    )
    write_json(out / "complete.json", result)
    return result


def train(config_path, device="cuda:0", resume=False):
    from ..config import load_recipe

    output = Path(load_recipe(config_path)["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another trainer is using this output directory") from None
        return _train(config_path, device, resume)
