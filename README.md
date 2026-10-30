# SAM3 Shelf

**Complete retail-instance masks with a fixed text prompt.**
**固定文本提示词下的货架完整商品实例分割。**

`an object on the shelf`

Adapt SAM3 to the annotated *sale unit*: a transparent multipack is one instance
when its complete package has one GT mask. Return all retained masks for downstream
matching, with explicit score filtering and near-duplicate suppression.

以标注的销售单元为目标：透明组合装标成一个GT时，学习输出完整的一包。
默认输出所有保留候选，供下游匹配；不依赖GT框、SKU名称或GT辅助筛选。

This repository contains the reusable core, not private datasets, model weights,
internal experiment history, server addresses, or one-off migration scripts.
The code license has **not yet been selected**; see [release status](LICENSE_PENDING.md).

本仓库只保留可复用核心。私有数据、权重和完整内部实验历史另行保存。
**代码许可证待确认，尚未添加开源授权条款。源代码可见不等于已授予开源许可。**

## Start here / 从这里开始

| You want to… / 目标 | Read / 文档 |
|---|---|
| Understand the architecture / 理解架构 | [Architecture](docs/architecture.md) |
| Add another method / 增加实验方法 | [Experiment development](docs/experiments.md) |
| Prepare data / 准备标注 | [Data format](docs/data.md) |
| Train or resume / 训练与恢复 | [Training](docs/training.md) |
| Evaluate and compare / 评测与可视化 | [Evaluation](docs/evaluation.md) |
| Download associated weights / 下载配套权重 | [Pretrained models](docs/pretrained_models.md) |
| Trace the method / 了解方法及依据 | [Methods](docs/methods.md) · [References](docs/references.md) |
| Prepare a release / 开源准备 | [Release](docs/release.md) |

## Install / 安装

The reference environment is Linux, Python 3.12, PyTorch 2.5.1 and Transformers
5.12.1. A CUDA GPU is required for the reference trainer. The repository does not
download weights automatically. Obtain a compatible **Hugging Face SAM3 model
directory** following the upstream model-access instructions.

参考环境为Linux、Python 3.12、PyTorch 2.5.1和Transformers 5.12.1。
训练使用CUDA；请自行按上游要求获取HF格式SAM3权重，不将原生`.pt`直接改名使用。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=68' wheel
# Choose the PyTorch build appropriate for your machine; CUDA 12.1 example:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
sam3-shelf doctor
```

[requirements-tested.txt](requirements-tested.txt) records tested library versions.
The shared environment of an existing installation does not need to be upgraded.

## Inference / 推理

```bash
sam3-shelf infer \
  --model checkpoints/finetuned \
  --image data/example.png \
  --output outputs/example --device cuda:0 --save-masks
```

Defaults: score **strictly > 0.1**, mask threshold **0.5**, mask near-duplicate
IoU **0.98**, no candidate-count cap. Outputs are native-size masks, row-major RLE,
scores and an overlay. The prompt is fixed, and GT is never used during inference.

默认严格`score > 0.1`、mask阈值0.5、mask近重复去重IoU 0.98，不限制候选数量。
输出原尺寸mask、RLE、评分和叠加图；推理不访问GT。

## One workflow for new experiments / 新实验共用同一流程

1. Put annotations in the [JSONL format](docs/data.md).
2. Copy a config in `configs/experiments/`; set local model/data/output paths.
3. Keep the method unchanged for a data/budget control, or register a new method.
4. Run preflight, then the same train/evaluate/review pipeline.

```bash
sam3-shelf run --config configs/experiments/c7a_reference.json --check-only
sam3-shelf run --config configs/experiments/c7a_reference.json --device cuda:0
# Resume the exact config/code/data:
sam3-shelf run --config configs/experiments/c7a_reference.json --device cuda:0 --resume
```

开启`compare_baseline`时，流程先评测原始模型，再训练。每个实验独立记录配置、
数据指纹、代码身份、训练范围、验证曲线、资源、BEST和final。`best`与`final`分开保存，
不会把末轮自动冒充验证BEST。

The reference methods are `c7a_joint` and its `supervised_joint` control. An
executable score-only plugin demonstrates extending the architecture; it is not
presented as the historical two-stage score-calibration method.

## Review / 查看真实候选与TOP-N

```bash
sam3-shelf evaluate --model checkpoints/base --manifest data/test.jsonl \
  --output outputs/baseline --device cuda:0
sam3-shelf evaluate --model checkpoints/finetuned --manifest data/test.jsonl \
  --output outputs/finetuned --device cuda:0
sam3-shelf review --evaluation Baseline=outputs/baseline \
  --evaluation Finetuned=outputs/finetuned --output reports/comparison
sam3-shelf serve --directory reports/comparison --port 8836
```

Open `http://127.0.0.1:8836/`. The page compares actual masks and supports TOP-N by
score. Cards show complete-instance and TOP-3/5/10/50 recall. Unannotated images
show predictions only, without invented recall values. Default report assets are
links; add `--copy-assets` to create a standalone export.

页面支持模型对比、真实mask叠加和按分数TOP-N，标注集展示TOP-3/5/10/50召回。
无GT时不计算召回。默认链接已有评测产物，避免重复占空间；分享独立文件夹时加`--copy-assets`。

## Layout / 目录

```text
configs/
  methods/           reusable method settings / 方法配置
  experiments/       data, budget and output controls / 实验配置
src/sam3_shelf/
  methods/           registered trainable scopes and objectives / 方法插件
  training/          trainer, teacher cache, sampling, preflight / 共享训练设施
  data.py            manifests and annotation checks
  inference.py       GT-free fixed-prompt inference
  metrics.py         one-to-one complete-instance metrics
  evaluation.py      real prediction artifacts
  pipeline.py        shared experiment workflow
  report.py, assets/ static comparison viewer
docs/                short topic documents / 按主题文档
examples/            synthetic data and usage examples
tests/               CPU tests and numerical contracts
scripts/             release checks, not one script per experiment
```

## Checks / 检查

```bash
python examples/make_demo_data.py --output data/demo
sam3-shelf validate-data --manifest data/demo/train.jsonl --compare data/demo/validation.jsonl
python -m unittest discover -s tests -v
python scripts/check_release.py
```

Synthetic rectangles verify format and software behavior; they are not a shelf
segmentation benchmark. Private historical scores are not advertised as public
held-out benchmark results. See [metric definitions](docs/evaluation.md) and
[method provenance](docs/references.md).
