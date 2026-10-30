# Refactor verification / 重构验证

The shared workflow is a new implementation. Numerical checks establish the
parts listed below; they do not establish that a complete historical training
run has been replayed.

统一流程是新的实现。已完成以下检查，尚未用整理后的训练器重新跑完整训练，不宣称
所有历史权重能够逐位重现。

- CPU tests cover configuration inheritance, RGBA mask decoding, RGB/group split
  overlap, strict score filtering, loss gradients, complete-instance metrics,
  evaluation resume and annotation changes, and baseline-before-training order.
- Four protected/unknown/conflicting teacher cases produce exactly the same loss,
  diagnostics and gradients as the original joint objective.
- The first 9,024 weighted image exposures match the original sampler exactly.
- A real trained SAM3 checkpoint strictly loads; the joint trainable scope is
  24,529,462 parameters, matching the original implementation.
- Two real dataset configurations pass CPU preflight, including training versus
  validation group/RGB separation, mask dimensions and effective minimum budget.
- The wheel builds with bundled viewer assets; the installed package CLI starts
  without importing the training stack. Viewer JavaScript passes syntax checking. Browser checks also confirm two-model
  overlays, TOP-1 score order, TOP-3/5/200, GT/no-GT layers and the original image
  when no candidates remain.

For a new method, repeat checks specific to its trainable scope, teacher/cache
contract, checkpoint save/reload and gradients before committing GPU time.
Runtime data and verification receipts remain outside the distributed source.
