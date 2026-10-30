# Evaluation and review / 评测与查看

## Complete-instance recall / 完整实例召回

For a prediction P and an annotated unit G:

```text
coverage = |P intersect G| / |G|
purity   = |P intersect G| / |P|
complete = coverage >=0.95 and purity >=0.95
```

Use maximum-cardinality one-to-one matching among complete pairs, with IoU as a
secondary preference. A GT or prediction cannot be counted twice. TOP-N uses
the highest-score retained candidates after thresholding, empty-mask removal
and near-duplicate NMS. Extra unannotated rear instances are not penalized as FP
by this completeness objective. It is **not COCO AP**.

标注只覆盖前排时，未标注后排不能简单计作误检。完整召回高也不能证明所有额外候选都好，
仍应查看合并、残缺、冗余、候选数量及下游匹配表现。

## Commands / 命令

```bash
sam3-shelf evaluate --model checkpoints/finetuned --manifest data/test.jsonl --output outputs/test
sam3-shelf benchmark --model checkpoints/finetuned --manifest data/test.jsonl --output outputs/timing.json
sam3-shelf review --evaluation Model=outputs/test --output reports/test
sam3-shelf serve --directory reports/test
```

Each evaluation exports `summary.json`, per-image original/overlay PNGs,
`candidates.json` RLE, raw scores and optional GT. RLE is uncompressed **row-major**
alternating zero/one counts, not COCO RLE. Inference never reads GT; GT is used
only after prediction for metrics and the explicit GT display layer.

Evaluate timing includes GT and PNG/RLE work and must not be called pure
inference latency. The separate benchmark excludes load/IO/GT/export/warmup and
records the actual device and per-process CUDA memory. Compare latency on the
same device/runtime; parameter-efficient training does not necessarily improve
deployment latency.

Do not tune on a final test set and then present it as blind evaluation. Name
validation selection, training fit and held-out results separately. For no-GT
manifests the viewer displays predictions without fabricated quality scores.
