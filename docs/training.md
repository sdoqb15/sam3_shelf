# Training / 训练

## Reference recipe / 参考训练路线

`c7a_joint` starts from a local original SAM3 checkpoint, freezes visual/text/
geometry encoders, and updates DETR, mask decoder, scoring and presence-related
parameters. The verified trainable count is 24,529,462. There is no augmentation
or built-in LoRA in this reference method.

The objective contains mask-first Hungarian matching, mask BCE/Dice, local ROI
positive/negative terms, box L1/GIoU, classification focal, presence and GT-safe
teacher preservation. See [methods](methods.md) for the exact policy.

The descriptive method ID is `gt_safe_joint`; `c7a_joint` is a compatibility
alias. `teacher_model` optionally selects a frozen teacher independently of the
initial student. Without it, the initial model is also the frozen teacher.

Defaults: AdamW, LR 1e-5, weight decay 1e-4, gradient clip 1, physical batch 1 and
accumulation 4, seed 42. The historical warmup/cosine formula is retained over
2380 updates, followed by LR 1e-6. The sampling namespace is part of the random
stream; the reference config keeps `c2` for historical stream compatibility.

## Run / 执行

```bash
sam3-shelf train --config configs/experiments/c7a_reference.json --check-only
sam3-shelf prepare-teacher --config configs/experiments/c7a_reference.json
sam3-shelf train --config configs/experiments/c7a_reference.json
sam3-shelf train --config configs/experiments/c7a_reference.json --resume
```

`run` executes the shared baseline/train/evaluation/review workflow; `train`
executes optimization only. Teacher references are content-addressed by the
original weight hash, policy, image and GT fingerprints, and can be reused across
compatible experiments. Annotation changes produce new cache entries.

## Checkpoint selection / 选模

Every119 updates the trainer saves BF16 parameters, reloads that deployment
checkpoint and validates it. The ordering is TOP-5+TOP-10 complete GT count,
complete GT, whole-image success, then fewer candidates. Exact ties retain the
earlier checkpoint. The minimum budget is at least one full weighted cycle;
patience counts consecutive validation checks, not raw optimization steps.

`best` is selected on validation; `final` is the last checkpoint. The delivery
example raises minimum updates and patience and evaluates `final` explicitly.
It does not rename a final checkpoint as validation BEST.

## Reproducibility limits / 复现边界

These are portable implementations of the validated objective and recipe, not
binary copies of a historical run. Old experiments changed some validation
rules during execution and used private data. The clean workflow applies its
declared rule from the beginning. Reproducing exact historical checkpoints also
requires those data snapshots, ordering, source versions and execution history.

The trainer records deterministic settings and uses warn-only CUDA determinism;
some attention/backward kernels can remain nondeterministic. BF16 boundary
pixels/candidate counts may differ across runtime settings. No bit-identical
training guarantee is made. Benchmark and evaluation settings must be reported
alongside any comparison.
