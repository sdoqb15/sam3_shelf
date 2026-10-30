"""A small executable plugin example, not a claimed reproduction of score calibration."""

from ..model import configure_modules
from .joint import TeacherProtectedJoint
from .registry import register_method


@register_method("score_only_focal_control")
class ScoreOnlyFocalControl(TeacherProtectedJoint):
    """Use the joint focal/teacher objective while updating only the score head."""

    def configure(self, model, config):
        scope = configure_modules(model, ("dot_product_scoring.",), expected_parameters=1_182_976)
        return model, scope
