# Reference method / 参考方法

## Supervision priority / 监督优先级

Human GT is matched first. Remaining queries use the frozen original-model
references and geometric correspondence, not equal query indices.

| Candidate | Classification | Geometry |
|---|---|---|
| Matched human GT | Positive | Human mask, box and local boundary |
| Teacher score >=0.9, GT-safe student correspondence | Exempt from background; soft teacher target | Teacher mask/box, total weight0.25 |
| Teacher0.85<=score<0.9, GT-safe correspondence | Unknown: no direct class target | No teacher geometry loss |
| Other unmatched queries | Focal background negative | No whole-image background-mask target |

GT-safe requires intersection/teacher area <=2% and intersection/each GT area
<=2%. The student mask is checked again so a GT-conflicting merge cannot borrow
the exemption. At most32 references receive teacher loss; safe overflow remains
unknown. Teachers are absent during inference.

## Joint loss / 联合损失

```text
2*mask_BCE + 2*Dice + 2*focal
+ local_positive + local_negative
+ 5*box_L1 + 2*box_GIoU + presence_BCE
+ 0.25*teacher_preservation
```

Mask-first assignment uses sampled mask BCE/Dice, a bounded classification term
and box costs. See `training/geometry.py` and `training/losses.py`; the numerical
tests check GT precedence, unknown gradients and conflicting teacher matches.

`supervised_joint` uses the same trainable scope and joint losses with empty
teacher references. It is a control, not a claim of better accuracy.

## What this method does not imply / 方法边界

Unmatched queries outside the exemption bands are still negative, including
some unannotated real objects. This is not the policy "all unlabelled rear is
unknown". The score is query confidence multiplied by presence probability,
not a direct estimate of mask completeness. An experiment that calibrates
quality must explicitly define its targets and negative/unknown treatment.

The framework retains SAM3's architecture. It does not implement an independent
MaskIoU quality head or reproduce the complete official SAM3 training system.

### Frozen TOP-N scoring

`topn_score` (plugin `sam3_shelf.methods.topn_score`) uses the shared trainer and
freezes all modules except `dot_product_scoring`. Start from a completed parent
checkpoint and provide its SHA256 through `score_calibration.parent_weights_sha256`.
The recipe template is `configs/methods/topn_score.json`.

The method promotes one complete candidate per annotated GT only when its parent
rank misses a reachable TOP-N target (including a newly displaced current candidate). Competitor cutoffs are detached; unannotated
candidates do not receive background labels. Confirmed merges/fragments have
separate normalized losses. Other complete and unknown candidates retain explicit
score anchors. Parent geometry and scores are cached on the server without raw
mask tensors. Validation checks all raw mask/box/presence geometry, retained GT
identities, confirmed error counts/ranks and protected candidate scores before
selecting a checkpoint. A parent-equivalent step-zero checkpoint remains a valid
fallback; this method does not guarantee an improvement on unseen images.

Training and validation use the deployment inference backend to avoid attaching
cached geometry labels to a different numerical forward. Checkpoints retain the
standard SAM3 architecture and normal inference API. Geometry freezing alone does
not preserve the threshold/NMS output set; inspect actual-output regressions.

### GT-supported quality ranking and sampling controls

`quality_rank` (`sam3_shelf.methods.quality_rank`) derives from the frozen scoring
method and reuses its geometry references. Complete/confirmed-error pairs keep
both score gradients, so the ranking term is invariant to a shared additive
score bias. It mines at most three threatening errors per GT. A separate weak
ranking group covers clearly GT-related intermediate-quality masks without
assigning those masks a background target. Independent unknown instances retain
score anchors. Full geometry and actual-output validation guards are inherited.

The template `configs/methods/quality_rank.json` defines a fixed scoring pilot.
`sampling.strategy=weighted` reproduces the original stream;
`quality_stratified` draws a hard multi image, a hard single image, a high-score
confirmed-error image and a normal replay image per four-exposure update. Mining
uses training references only and is saved with model/data/target fingerprints.

The shared trainer accepts the optional method `sample_rows` hook. Its default
sampling and full-cycle minimum are unchanged. Set
`training.require_full_cycle=false` explicitly for a fixed-budget scoring pilot;
the available pool is then distinct from the samples actually visited. Training
logs aggregate all accumulated microbatches. An already-validated preflight can
be passed internally to avoid decoding the same frozen dataset twice; recipe
identity is checked before reuse.
