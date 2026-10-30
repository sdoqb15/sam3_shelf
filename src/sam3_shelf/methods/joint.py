"""Reference joint methods: teacher-protected and fully supervised controls."""

from ..model import configure_joint
from ..training.sampling import validation_value
from .registry import register_method


@register_method("c7a_joint")
@register_method("gt_safe_joint")
class TeacherProtectedJoint:
    """Keep the validated joint objective and GT-safe teacher/unknown bands."""

    needs_teacher = True

    def configure(self, model, config):
        return model, configure_joint(model)

    def loss(self, output, targets, teacher, config):
        from ..training.losses import joint_loss

        return joint_loss(output, targets, teacher, config)

    def selection_value(self, metrics):
        return validation_value(metrics)

    def prepare_teacher(self, config, rows, audit, device):
        from ..training.teacher import prepare_teacher

        return prepare_teacher(config, rows, audit, device)

    def load_teacher(self, row, config, audit):
        from ..training.teacher import cache_directory, empty_teacher

        if not self.needs_teacher:
            return empty_teacher()
        import torch

        return torch.load(
            cache_directory(config, audit) / (row["_cache_key"] + ".pt"), weights_only=True
        )

    def validate(self, model, processor, rows, config):
        from ..evaluation import evaluate_model

        return evaluate_model(model, processor, rows)


@register_method("supervised_joint")
class SupervisedJoint(TeacherProtectedJoint):
    """Same trainable scope and losses, without teacher/unknown exemptions."""

    needs_teacher = False
