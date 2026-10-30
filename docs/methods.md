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
