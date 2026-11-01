"""Gradient, reachability and partial-label safeguards for frozen TOP-N ranking."""
import copy
import unittest
import torch
from sam3_shelf.methods.topn_score import topn_loss,classify,eligible,FrozenTopNScore

SPEC=dict(top_ns=[5,10],rank_margin=.2,error_margin=1.,negative_score_ceiling=.05,
 complete_coverage=.95,complete_purity=.95,merge_coverage=.8,merge_purity_max=.7,
 fragment_coverage_min=.2,fragment_coverage_max=.8,fragment_purity=.98,safe_gt_overlap=.02,
 protected_min_score=.9,rank_weight=1.,error_rank_weight=.5,negative_weight=1.,
 complete_anchor_weight=2.,rear_anchor_weight=2.,unknown_anchor_weight=.5,
 maximum_rear_mean_score_drop=.02,maximum_rear_score_drop=.1)

class TopNScoreTests(unittest.TestCase):
 def example(self,gt=1):
  coverage=torch.zeros(gt,12);purity=torch.zeros(gt,12)
  for g in range(gt):coverage[g,g]=purity[g,g]=1
  z=torch.tensor([-2.,2.,1.5,1.,.8,.4,.2,0.,-.2,-.4,-.8,-1.])
  return dict(coverage=coverage,purity=purity,class_logits=z,scores=z.sigmoid(),presence=torch.tensor(1.),nonempty=torch.ones(12,dtype=torch.bool),nms_keep=torch.argsort(z,descending=True))
 def test_hard_positive_up_without_negative_gradient_for_unknown_cutoff(self):
  r=self.example();z=r['class_logits'].clone().requires_grad_();loss,detail=topn_loss(z,r,SPEC);loss.backward()
  self.assertLess(float(z.grad[0]),0);self.assertTrue(torch.equal(z.grad[1:],torch.zeros(11)))
  self.assertEqual(detail['hard_positives'],1)
 def test_correct_high_rank_is_anchored_without_boosting(self):
  r=self.example();r['class_logits'][0]=4;r['scores']=r['class_logits'].sigmoid();z=r['class_logits'].clone().requires_grad_()
  loss,detail=topn_loss(z,r,SPEC);loss.backward();self.assertTrue(torch.equal(z.grad,torch.zeros(12)));self.assertEqual(detail['hard_positives'],0)
 def test_newly_displaced_parent_success_is_also_restored(self):
  r=self.example();r['class_logits'][0]=4;r['scores']=r['class_logits'].sigmoid()
  z=r['class_logits'].clone();z[0]=-2;z.requires_grad_()
  _,detail=topn_loss(z,r,SPEC);self.assertEqual(detail['hard_positives'],1)
 def test_confirmed_errors_down_independent_rear_preserved(self):
  r=self.example();r['coverage'][0,2]=1;r['purity'][0,2]=.5;r['coverage'][0,3]=.5;r['purity'][0,3]=1
  z=r['class_logits'].clone().requires_grad_();topn_loss(z,r,SPEC)[0].backward()
  self.assertGreater(float(z.grad[2]),0);self.assertGreater(float(z.grad[3]),0);self.assertEqual(float(z.grad[1]),0)
 def test_positive_for_another_gt_never_becomes_negative(self):
  r=self.example(gt=2);r['coverage'][0,1]=1;r['purity'][0,1]=.5;t=classify(r,SPEC)
  self.assertTrue(t['good_any'][1]);self.assertFalse(t['bad'][:,1].any())
 def test_more_gt_than_rank_capacity_skips_unreachable_top5(self):
  r=self.example(gt=6);s=dict(SPEC,top_ns=[5]);z=r['class_logits'].clone().requires_grad_();loss,detail=topn_loss(z,r,s);loss.backward()
  self.assertEqual(detail['hard_positives'],0);self.assertEqual(detail['rank'],0.)
 def test_only_one_complete_candidate_per_gt_is_promoted(self):
  r=self.example();r['coverage'][0,11]=r['purity'][0,11]=1
  t=classify(r,SPEC);self.assertEqual(len(t['primary']),1);self.assertEqual(int(t['primary'][0,1]),11)
 def test_parent_with_presence_below_output_threshold_is_not_promised_recovery(self):
  r=self.example();r['presence']=torch.tensor(.05);r['scores']=r['class_logits'].sigmoid()*.05
  z=r['class_logits'].clone().requires_grad_();_,detail=topn_loss(z,r,SPEC);self.assertEqual(detail['hard_positives'],0)
 def test_rear_drift_is_pulled_back_without_background_label(self):
  r=self.example();z=r['class_logits'].clone();z[1]-=1;z.requires_grad_();topn_loss(z,r,SPEC)[0].backward();self.assertLess(float(z.grad[1]),0)
 def test_guard_rejects_pollution_or_lost_parent_success_even_with_topn_gain(self):
  base=dict(complete_gt_ids=['a'],protected_ids=['rear'],known_bad=1,bad_in_top5=0,bad_in_top10=1,quality_inversion_gt=0)
  now=dict(base,rear_score_drop_mean=0.,rear_score_drop_max=0.)
  self.assertTrue(eligible(now,base,SPEC));now['bad_in_top5']=1;self.assertFalse(eligible(now,base,SPEC))
  now=dict(base,complete_gt_ids=[],rear_score_drop_mean=0.,rear_score_drop_max=0.);self.assertFalse(eligible(now,base,SPEC))
 def test_export_audit_rejects_non_score_weight_change(self):
  import tempfile
  from pathlib import Path
  from safetensors.torch import save_file
  from sam3_shelf.methods.topn_score import audit_frozen_checkpoint
  with tempfile.TemporaryDirectory() as directory:
   parent=Path(directory)/'parent';child=Path(directory)/'child';parent.mkdir();child.mkdir()
   save_file({'dot_product_scoring.weight':torch.ones(2),'mask_decoder.weight':torch.ones(2)},str(parent/'model.safetensors'))
   save_file({'dot_product_scoring.weight':torch.zeros(2),'mask_decoder.weight':torch.ones(2)},str(child/'model.safetensors'))
   self.assertEqual(audit_frozen_checkpoint(parent,child)['status'],'passed')
   save_file({'dot_product_scoring.weight':torch.zeros(2),'mask_decoder.weight':torch.zeros(2)},str(child/'model.safetensors'))
   with self.assertRaises(ValueError):audit_frozen_checkpoint(parent,child)
 def test_ineligible_better_topn_never_selected(self):
  m=FrozenTopNScore();self.assertEqual(m.selection_value(dict(eligible=False)),(-1,))

if __name__=='__main__':unittest.main()
