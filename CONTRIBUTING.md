# Development / 开发约定

- Start with [architecture](docs/architecture.md) and [method development](docs/experiments.md).
- Add a config for a data/LR/budget control; add a method plugin for a new loss or trainable scope.
- Keep dataset export and machine scheduling outside the training objective.
- Preserve mask geometry, alpha handling, score precision and the documented metric definitions.
- Record data fingerprints, validation selection, resources and unsuccessful outcomes.
- Add behavioral or numerical tests for changed training/inference semantics.
- Run `python -m unittest discover -s tests -v` and `python scripts/check_release.py`.
- Keep generated outputs, private annotations, checkpoints and credentials out of source control.
- Code licensing is pending; publication/contribution licensing will be documented after selection.
