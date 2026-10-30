# Data format / 数据格式

One UTF-8 JSON object per line. Relative image and mask paths resolve against the
manifest directory, or the explicit `--data-root`/recipe `data.root`.

```json
{"id":"scene-001/view-1","group":"scene-001","image":"images/001.png","width":1280,"height":720,"sampling_group":"multi","instances":[{"mask":"masks/001_0.png"},{"mask":"masks/001_1.png"}]}
```

| Field | Meaning |
|---|---|
| `id` | Unique image identity |
| `group` | Same shelf/capture group, shared across nearby views |
| `image` | Original RGB image path |
| `width`, `height` | Native dimensions; optional until validation fills them |
| `sampling_group` | `multi` or `single`, based on the dataset source |
| `instances` | One mask per complete annotated sale unit |

`scene_group` and `split_group` from older manifests are accepted as group aliases.
Mark a multi-image source explicitly even when a particular image has only one
GT. If `sampling_group` is omitted, one GT defaults to `single`, otherwise `multi`.

## Mask semantics / Mask含义

- Transparent multipack: one complete package GT, not one GT per internal item.
- Repeated SKU: distinct visible units have distinct instance masks.
- Preserve disconnected regions and holes. Do not split by connected components.
- RGBA PNG uses alpha >127. Grayscale/RGB binary masks use nonzero foreground,
  supporting both 0/1 and 0/255.
- Native dimensions must match. Empty instance masks are rejected.
- Missing `instances` means unannotated; `instances: []` means annotated empty.
- The reference joint trainer requires nonempty training GT; empty-image training
  requires a method that explicitly supports its presence/negative objective.

## Split checks / 数据划分

```bash
sam3-shelf validate-data --manifest data/train.jsonl --compare data/validation.jsonl
```

Train/validation checks reject overlapping IDs, **decoded RGB**, and capture
groups. Group membership still needs your capture knowledge: a duplicate check
does not infer all near-duplicate shelf views. SKU overlap and scene overlap are
different evaluation questions.

Training requires explicit groups. A test set added to delivery training must
be called `training_fit`, not `held_out_test`, in the evaluation config. A partial
front-item GT set is not an exhaustive annotation of rear objects.

The synthetic generator demonstrates the schema and split checks without
distributing private images. It is not evidence of real shelf-model accuracy.
