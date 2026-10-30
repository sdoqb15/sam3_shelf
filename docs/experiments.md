# Adding an experiment or method / 新增实验与方法

## 1. Data, learning rate or budget changes / 只改变数据或预算

Copy an experiment config, inherit its method, change the intended variables and
use a new output directory. Keep evaluation data and postprocessing fixed when
comparing methods. Do not copy a trainer file or overwrite another run.

```json
{
  "extends": "c7a_reference.json",
  "experiment": {"name": "more_multi_images"},
  "output_dir": "../../runs/more_multi_images",
  "data": {"train": "../../data/larger_train.jsonl"}
}
```

## 2. New trainable modules or objective / 改变方法

A plugin implements trainable scope, loss and selection. It may also override
teacher preparation/loading and validation diagnostics:

```python
from sam3_shelf.methods import register_method
from sam3_shelf.methods.joint import TeacherProtectedJoint

@register_method("my_method")
class MyMethod(TeacherProtectedJoint):
    needs_teacher = True

    def configure(self, model, config):
        # Return the configured model and actual scope metadata.
        return super().configure(model, config)

    def loss(self, output, targets, teacher, config):
        loss, details = super().loss(output, targets, teacher, config)
        # Add a separately normalized term here; report it in details.
        return loss, details

    def selection_value(self, metrics):
        return super().selection_value(metrics)
```

Set `method` to `my_method` and `method_plugin` to the importable module containing
the plugin. The included `sam3_shelf.methods.example` demonstrates a real
score-head-only focal control. Its config is executable, but it is **not** the
historical complete-candidate score-calibration recipe.

如需PEFT，`configure`可返回包装后的模型；部署导出会合并支持`merge_and_unload`的适配器。
新增LoRA方案仍需自己的参数范围、保存重载和输出一致性测试，不能仅凭开关认为已生效。

## 3. Check the contract / 新方法检查

1. Run `--check-only`: verify data, grouping, checkpoint and effective minimum budget.
2. Verify actual trainable names/count and finite gradient directions on a small test.
3. Keep complete candidates protected in any GT-derived negative mining.
4. Export ordinary checkpoints and evaluate real score-filtered/NMS outputs.
5. Record failure cases as well as recall gains; an added candidate is not automatically a correct mask.
6. Compare loss or module changes with the same data and budget whenever possible.

The shared trainer handles gradient accumulation, RNG/AdamW resume, selection,
resource collection and artifacts. A plugin should not copy that infrastructure.

Optional hooks: `prepare_teacher(config, rows, audit, device)`,
`load_teacher(row, config, audit)` and `validate(model, processor, rows, config)`.
Use these for a different frozen teacher, a richer full-query score cache, or
additional quality diagnostics. The default hooks implement GT-safe joint
references; a new score-quality method need not squeeze its cache into that
format or change the shared trainer. `teacher_model` can explicitly select a
frozen checkpoint independently of initialization.

## 4. Resume versus a new experiment / 恢复与新实验

`--resume` requires matching method/configuration, algorithm source, model weights
and annotation fingerprints. New unrelated plugins do not change an existing
method's source identity. A new objective or changed training data is a new run,
not an undocumented resume. Evaluation sets are recorded independently and may
be added under new names without rewriting old predictions.
