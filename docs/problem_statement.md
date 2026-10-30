# Problem background and task / 问题背景与任务定义

## Shelf segmentation for downstream decisions / 面向下游决策的货架分割

In supermarket and warehouse scenes, instance segmentation is an input to SKU
recognition and robot picking. This project uses SAM3 with the fixed prompt
`an object on the shelf` to produce candidate masks and scores. A downstream
system embeds each masked crop, compares it against a SKU vector database,
selects a suitable mask, and passes that perception result to robot planning.

在超市和仓库场景中，实例分割的结果会继续用于商品识别和机器人抓取。本项目固定输入
`an object on the shelf`，返回候选mask及其评分。下游对mask覆盖的商品区域提取向量，
与SKU向量库匹配并挑选合适的mask，再将该感知结果用于机器人抓取规划。

```mermaid
flowchart LR
    A[RGB image + fixed prompt] --> B[Sale-unit masks + scores]
    B --> C[Masked-crop embeddings]
    C --> D[SKU vector matching and mask selection]
    D --> E[Robot perception and picking plans]
```

## Observed problems / 观察到的问题

Original SAM3 can segment many products in these scenes. Its generic object
boundaries and confidence ranking do not consistently match the sale units and
candidate priorities required by this application.

原始SAM3已经能分割出许多商品，但在本项目的货架场景里，商品单元的边界和候选排序仍有
以下问题。这些是本项目场景中的观察，不代表所有SAM3场景都会出现相同错误。

| Observation / 现象 | Downstream impact / 下游影响 |
|---|---|
| Several identical items are bound together by transparent film as one sale unit, but masks isolate individual contents or only part of the package. 透明薄膜绑在一起的多个相同商品属于一件组合商品，模型却拆分内部单品，或只覆盖包装的一部分。 | The embedding represents a fragment or a different sale unit, making SKU/vector matching and mask selection less reliable. 向量表征偏向局部或错误商品单元，干扰SKU匹配和mask挑选。 |
| A whole-package candidate is missing or incomplete. 没有返回完整包装候选，或外轮廓残缺。 | Downstream matching cannot recover a correct mask that was never proposed; the perception region used for picking is incomplete. 下游无法从候选中挑到不存在的正确mask，提供给抓取感知的区域也不完整。 |
| Confidence does not reliably prioritize a complete, currently exposed outermost item. 当前可见、最外侧完整商品的mask不一定排在前面。 | Good candidates can be delayed or displaced by less useful crops; a wrong selection can propagate into target/grasp-region estimation. 好候选可能被排后或挤掉，错误选择会传递到抓取目标及区域估计。 |
| A candidate merges front and rear products or adds unrelated shelf regions. 前后商品被合并，或mask额外包含无关区域。 | Mixed appearance can confuse vector matching, waste candidate processing and corrupt the selected instance boundary. 混合外观干扰向量匹配、增加候选处理开销，并使最终实例边界失真。 |

Some fine-tuning attempts also amplified the last problem: a messy mask could
score highly merely because it overlapped a labeled front item. The intended
score preference is for a **complete single sale unit**, including a complete
multipack, rather than any mask that contains part of that unit.

部分微调尝试还会放大最后一种问题：杂乱mask因为包含了已标注的前排商品区域而获得高分。
目标应是让**完整单个销售单元**的候选靠前；一个完整组合包装也算一个单元，单纯“包含了
前排商品的一部分面积”不能成为杂乱mask获得高分的理由。

## Desired behavior / 期望行为

- **Respect the sale unit.** Return a full mask for the annotated package. A
  transparent multipack is one instance when sold/annotated as one unit.
  **按销售单元分割。** 透明组合装按完整一包标注时，期望输出一包的完整mask。
- **Recall the usable complete candidate.** Each target should have a complete
  candidate available to downstream matching; extra valid rear-object candidates
  are acceptable. **优先完整召回。** 下游需要的每个目标都应有完整候选；额外正确的后排候选可接受。
- **Rank useful complete instances ahead of fragments and merges.** More masks
  alone do not make a better result. **改善排序。** 完整单件应优先于残缺或合并候选，候选数量增加不等于效果改善。
- **Follow the currently exposed front of each column.** Columns have different
  stock depths. After outer items are removed, a deeper item can become that
  column's current outermost item and still needs a complete, useful candidate.
  **面向每列当前最外侧商品。** 各列排数不同，拿走前排后露出的靠里商品也需要完整mask和合适排序。
- **Retain useful rear segmentation.** An unlabeled rear item is not automatically
  a negative. This matters when stock changes expose it later.
  **保留后排分割能力。** 未标注后排不等于真负例，库存变化后它可能成为当前最外侧商品。

The output contract is native-resolution masks and scores with the fixed prompt,
score strictly `>0.1`, mask threshold `0.5`, near-duplicate mask NMS IoU `0.98`
and no candidate-count cap. Inference does not use GT to choose candidates.
The downstream vector matcher remains responsible for SKU matching and final
mask selection. Mask confidences and vector-matching similarities are separate
quantities. These segmentation scores are not calibrated grasp-success
probabilities, depth estimates or reachability predictions.

输出采用原尺寸mask和score，固定提示词，严格`score >0.1`、mask阈值0.5、近重复去重
IoU0.98，不限制候选数量；推理时不借助GT挑选候选。下游继续负责SKU匹配和最终mask选择。
分割score与向量匹配相似度是两个不同量；它也不等于抓取成功概率，不直接表示深度或可达性。

## What evaluation establishes / 评测能说明什么

We inspect complete-instance recall, TOP-N recall, ranking and failure cases.
Completeness requires both GT coverage and prediction purity to reach0.95, with
one-to-one matching. The aim is useful full-unit proposals for downstream systems;
a segmentation metric alone does not establish improved robot grasp success.
Published checkpoints can still miss, split or merge products.

评测关注完整实例召回、TOP-N召回、排序和实际失败案例；完整匹配要求覆盖率与纯度均达到
0.95，并进行一对一匹配。分割指标不直接证明机器人抓取成功率提升，当前权重仍可能漏检、
拆分或合并商品。数据主要只标注最外侧商品，额外后排候选需结合实际mask检查。

See [evaluation](evaluation.md) for metric definitions and
[associated checkpoints](pretrained_models.md) for the C7-A/D2 data roles and model cards.
