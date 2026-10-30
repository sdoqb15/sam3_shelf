# Associated checkpoints / 配套权重

Code repository: <https://github.com/sdoqb15/sam3_shelf>.

| Model / 模型 | Hugging Face | Selected checkpoint / 权重 |
|---|---|---|
| C7-A | [sam3-shelf-c7a-2380](https://huggingface.co/sdoqb15/sam3-shelf-c7a-2380) | Joint-stage validation BEST, step 2380 |
| D2 | [sam3-shelf-d2-2261](https://huggingface.co/sdoqb15/sam3-shelf-d2-2261) | Step 2261, visually preferred delivery checkpoint |

Publication revisions are recorded in `configs/checkpoints/huggingface.json`
after remote verification. A null revision means publication is still pending;
the downloader refuses to use an unpinned release.

```bash
python scripts/download_models.py --model c7a --output checkpoints/c7a
python scripts/download_models.py --model d2-2261 --output checkpoints/d2-2261
sam3-shelf infer --model checkpoints/d2-2261 --image data/example.png --output outputs/example
```

These are complete Transformers `Sam3Model` checkpoints. Prompt:
`an object on the shelf`; score strictly `>0.1`, mask threshold `0.5`, mask NMS
IoU `0.98`, no candidate-count cap. Download is explicit; local inference does
not make network requests.

完整HF权重，可通过现有本地推理入口使用。D2的2261步与验证BEST 714步是不同权重；
这里明确发布2261步。C7-A未用原162张回归测试图训练；D2用过这些图片，其对应结果属于
训练回看。查看各模型卡中的完整数据与评测说明，不将两者混为独立盲测比较。

The associated dataset is private: `sdoqb15/sam3-shelf-data`. Access is controlled
by the dataset owner. Private data access and the SAM model license are separate.

The private snapshot contains C7-A's train1635/validation72/test162 images,
35 annotated supplemental images and165 unannotated supplemental views. The
annotation and unannotated supplemental lists can refer to the same scene/image.
Download its pinned version after authenticating with an authorized account:

```bash
hf download sdoqb15/sam3-shelf-data --repo-type dataset \
  --revision 882b760a80a748256f5020aa8673958b4a72f905 --local-dir data/private-snapshot
cd data/private-snapshot
sha256sum -c SHA256SUMS
for archive in archives/*.tar.gz; do tar -xzf "$archive"; done
```

Portable JSONL files are in `manifests/`. C7-A never trained on the regression
or supplemental data; D2 included the old regression and annotated supplemental
data. Unannotated images have no `instances` field and must not be treated as
fully negative training examples.

Weights retain the upstream SAM License supplied in their HF repositories.
The code license remains undecided; no new code-license grant is implied.
