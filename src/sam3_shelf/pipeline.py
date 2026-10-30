"""The same experiment pipeline for new configs: train, evaluate, benchmark, review."""

from pathlib import Path
import fcntl

from .data import assert_disjoint, read_manifest, validate_manifest
from .io import read_json, write_json
from .training.preflight import preflight


def _run_experiment(config_path, device="cuda:0", resume=False):
    from .benchmark import benchmark
    from .evaluation import evaluate, evaluation_identity
    from .report import build_report
    from .training.trainer import train

    config, train_rows, _, audit = preflight(config_path)
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    state = (
        read_json(out / "pipeline.json")
        if (out / "pipeline.json").exists()
        else dict(status="running", completed=[])
    )

    def ensure_evaluation(model, item, destination, label):
        rows = read_manifest(item["manifest"], config["data"].get("root"))
        validate_manifest(rows)
        if (destination / "summary.json").exists():
            existing = read_json(destination / "summary.json")
            if existing["identity"] != evaluation_identity(model, item["manifest"], rows):
                raise ValueError(f"Evaluation changed; use a new evaluation name: {item['name']}")
            return existing
        return evaluate(
            model,
            item["manifest"],
            destination,
            device,
            data_root=config["data"].get("root"),
            label=label,
            resume=resume,
        )

    for item in config.get("evaluations", []):
        rows = read_manifest(item["manifest"], config["data"].get("root"))
        validate_manifest(rows)
        if item["scope"] == "held_out_test":
            assert_disjoint(train_rows, rows)
        if item.get("compare_baseline", False):
            destination = out / "baseline" / item["name"]
            ensure_evaluation(config["model"], item, destination, "Original SAM3")
    trained = train(config_path, device, resume and (out / "identity.json").exists())
    state["training"] = trained
    write_json(out / "pipeline.json", state)
    checkpoint = config.get("evaluation_checkpoint", "best")
    if checkpoint not in ("best", "final"):
        raise ValueError("evaluation_checkpoint must be best or final")
    model = out / checkpoint
    for item in config.get("evaluations", []):
        name = item["name"]
        if not name or Path(name).name != name:
            raise ValueError("Evaluation names must be single directory names")
        rows = read_manifest(item["manifest"], config["data"].get("root"))
        validate_manifest(rows)
        if item["scope"] == "held_out_test":
            assert_disjoint(train_rows, rows)
        destination = out / "evaluations" / name
        summary = ensure_evaluation(model, item, destination, name)
        summary["evaluation_scope"] = item["scope"]
        write_json(destination / "summary.json", summary)
        if name not in state["completed"]:
            state["completed"].append(name)
        write_json(out / "pipeline.json", state)
        if item.get("benchmark") and not (destination / "benchmark.json").exists():
            benchmark(
                model,
                item["manifest"],
                destination / "benchmark.json",
                device,
                config["data"].get("root"),
            )
        report = out / "reviews" / name
        if not (report / "manifest.json").exists():
            comparisons = []
            if item.get("compare_baseline", False):
                comparisons.append(("Original SAM3", out / "baseline" / name))
            comparisons.append(
                (config.get("experiment", {}).get("name", config["method"]), destination)
            )
            build_report(comparisons, report)
    state.update(status="complete", checkpoint=str(model.resolve()), data_audit=audit)
    write_json(out / "pipeline.json", state)
    return state


def run_experiment(config_path, device="cuda:0", resume=False):
    from .config import load_recipe

    output = Path(load_recipe(config_path)["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pipeline is using this output directory") from None
        try:
            return _run_experiment(config_path, device, resume)
        except Exception as error:
            write_json(
                output / "pipeline_error.json", dict(status="needs_attention", error=repr(error))
            )
            raise
