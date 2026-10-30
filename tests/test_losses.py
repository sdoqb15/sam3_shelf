"""Gradient behavior for partial labels and GT-safe teacher exemptions."""

import unittest
from types import SimpleNamespace
import torch
from sam3_shelf.training.losses import joint_loss, focal_with_unknown


class TeacherLossTests(unittest.TestCase):
    def example(self, protected=False, conflict=False):
        m = torch.full((1, 4, 8, 8), -5.0, requires_grad=True)
        m.data[0, 0, :4, :4] = 5
        m.data[0, 1, 4:, 4:] = 5
        if conflict:
            m.data[0, 1, :4, :4] = 5
        z = torch.tensor([[2.0, -2.0, 1.0, 1.0]], requires_grad=True)
        boxes = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.5, 0.5],
                    [0.5, 0.5, 1.0, 1.0],
                    [0.0, 0.0, 0.1, 0.1],
                    [0.8, 0.0, 1.0, 0.1],
                ]
            ],
            requires_grad=True,
        )
        gt = torch.zeros(8, 8)
        gt[:4, :4] = 1
        output = SimpleNamespace(
            pred_masks=m,
            pred_logits=z,
            pred_boxes=boxes,
            presence_logits=torch.ones(1, 1, requires_grad=True),
        )
        mask = torch.full((1, 8, 8), -5.0)
        mask[:, 4:, 4:] = 5
        teacher = dict(
            class_logits=torch.tensor([3.0]),
            mask_logits=mask,
            boxes=torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
            protected=torch.tensor([protected]),
        )
        c = {"teacher_policy": dict(geometry_match_min_iou=0.5, loss_weight=0.25)}
        return (output, gt, teacher, c)

    def test_gt_safe_unknown_has_zero_score_gradient_and_others_are_negative(self):
        (o, g, t, c) = self.example()
        (loss, log) = joint_loss(o, [g], t, c)
        loss.backward()
        self.assertEqual(log["unknown_matched"], 1)
        self.assertEqual(log["teacher_matched"], 0)
        self.assertEqual(float(o.pred_logits.grad[0, 1]), 0)
        self.assertGreater(float(o.pred_logits.grad[0, 2]), 0)
        self.assertLess(float(o.pred_logits.grad[0, 0]), 0)

    def test_protected_teacher_receives_only_soft_target_not_background(self):
        (o, g, t, c) = self.example(protected=True)
        (loss, log) = joint_loss(o, [g], t, c)
        loss.backward()
        self.assertEqual(log["teacher_matched"], 1)
        self.assertEqual(log["unknown_matched"], 0)
        self.assertLess(float(o.pred_logits.grad[0, 1]), 0)
        self.assertGreater(float(o.pred_logits.grad[0, 2]), 0)

    def test_gt_conflicting_unknown_is_negative(self):
        (o, g, t, c) = self.example(conflict=True)
        (loss, log) = joint_loss(o, [g], t, c)
        loss.backward()
        self.assertEqual(log["unknown_matched"], 0)
        self.assertGreater(float(o.pred_logits.grad[0, 1]), 0)

    def test_gt_conflicting_high_teacher_cannot_shield_student_merge(self):
        (o, g, t, c) = self.example(protected=True, conflict=True)
        (loss, log) = joint_loss(o, [g], t, c)
        loss.backward()
        self.assertEqual(log["teacher_matched"], 0)
        self.assertGreater(float(o.pred_logits.grad[0, 1]), 0)

    def test_human_positive_overrides_ignore(self):
        z = torch.zeros(3, requires_grad=True)
        focal_with_unknown(z, [0], [0, 1], 1).backward()
        self.assertLess(float(z.grad[0]), 0)
        self.assertEqual(float(z.grad[1]), 0)
        self.assertGreater(float(z.grad[2]), 0)


if __name__ == "__main__":
    unittest.main()
