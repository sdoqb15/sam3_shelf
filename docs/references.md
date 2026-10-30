# References and provenance / 方法依据与边界

- [SAM3 paper](https://arxiv.org/abs/2511.16719) and [official training interface](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md): base model and upstream training reference.
- [Transformers SAM3 documentation](https://huggingface.co/docs/transformers/model_doc/sam3): the model/processor API used by this implementation.
- [Mask2Former](https://arxiv.org/abs/2112.01527): set-based mask-supervision reference; this repository does not replace SAM3 with Mask2Former.
- [LoRA](https://arxiv.org/abs/2106.09685) and [PEFT](https://huggingface.co/docs/peft/main/package_reference/lora): parameter-efficient adaptation references for extensions, not enabled in the default joint recipe.
- [Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241): classification confidence and mask quality can differ. The default implementation does **not** add that paper's MaskIoU head.

The joint/score sequencing, GT-safe confidence bands, weighted data sampling,
local loss weights and completeness/TOP-N protocol were task-specific experiment
designs. Internal method names are not paper names. These combinations are not
claimed as academic novelties or exact reproductions of cited full systems.

本项目采用已有模型与监督技术，并按商品完整mask目标组合、调整训练策略。
引用某篇论文的问题分析，不代表已实现论文全部模块；历史内部实验和私有指标不随核心发布。
