"""Shared-parameter ranking invariance, partial labels, and controlled sampling."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from sam3_shelf.methods.quality_rank import quality_loss,quality_targets,stratified_rows,FrozenQualityRank
from sam3_shelf.training.sampling import weighted_rows

SCORE=dict(complete_coverage=.95,complete_purity=.95,merge_coverage=.8,merge_purity_max=.7,
 fragment_coverage_min=.2,fragment_coverage_max=.8,fragment_purity=.98,safe_gt_overlap=.02,
 protected_min_score=.9,negative_score_ceiling=.05)
SPEC=dict(weak_quality_max=.9,strong_margin=1.,weak_margin=.2,max_pairs_per_gt=3,positive_floor_score=.12,
 strong_rank_weight=1.,weak_rank_weight=.25,positive_floor_weight=1.,negative_weight=1.,
 complete_anchor_weight=2.,rear_anchor_weight=2.,unknown_anchor_weight=.5)

class QualityRankingTests(unittest.TestCase):
 def example(self):
  z=torch.tensor([-.5,1.,.5,-1.,3.,-2.,-3.])
  c=torch.tensor([[1.,1.,.5,.85,0.,.92,.99]])
  p=torch.tensor([[1.,.5,1.,.99,0.,.99,.90]])
  return dict(id='train',class_logits=z,scores=z.sigmoid(),presence=torch.tensor(1.),
   coverage=c,purity=p,nonempty=torch.ones(7,dtype=torch.bool),nms_keep=torch.argsort(z,descending=True))
 def test_shared_common_bias_has_zero_ranking_gradient(self):
  r=self.example();s=dict(SPEC)
  for k in ['positive_floor','negative','complete_anchor','rear_anchor','unknown_anchor']:s[k+'_weight']=0.
  bias=torch.nn.Parameter(torch.tensor(0.));loss,_=quality_loss(r['class_logits']+bias,r,SCORE,s);loss.backward()
  self.assertAlmostEqual(float(bias.grad),0.,places=6)
  values=[float(quality_loss(r['class_logits']+x,r,SCORE,s)[0]) for x in [-.01,0,.01]]
  self.assertLess(max(values)-min(values),1e-5)
 def test_strong_pairs_have_both_gradients_unknown_is_not_background(self):
  r=self.example();z=r['class_logits'].clone().requires_grad_();quality_loss(z,r,SCORE,SPEC)[0].backward()
  self.assertLess(float(z.grad[0]),0);self.assertGreater(float(z.grad[1]),0);self.assertGreater(float(z.grad[2]),0)
  self.assertEqual(float(z.grad[4]),0)
 def test_middle_quality_is_relative_only_not_target_zero(self):
  r=self.example();r['coverage'][:,1:3]=0;r['purity'][:,1:3]=0
  z=r['class_logits'].clone();z[3]=1.;z.requires_grad_();loss,detail=quality_loss(z,r,SCORE,SPEC);loss.backward()
  self.assertEqual(detail['negative'],0.);self.assertGreater(float(z.grad[3]),0);self.assertLess(float(z.grad[0]),0)
  self.assertGreater(detail['weak_pairs'],0)
 def test_ambiguous_gt_overlap_keeps_anchor(self):
  r=self.example();r['purity'][0,6]=.94;t=quality_targets(r,SCORE,SPEC)
  self.assertFalse(t['weak'][:,5].any());self.assertFalse(t['weak'][:,6].any())
  z=r['class_logits'].clone();z[5]-=1;z.requires_grad_();quality_loss(z,r,SCORE,SPEC)[0].backward();self.assertLess(float(z.grad[5]),0)
 def test_complete_for_any_gt_overrides_wrong_for_another(self):
  r=self.example();r['coverage']=torch.cat([r['coverage'],torch.tensor([[0.,1.,0.,0.,0.,0.,0.]])]);r['purity']=torch.cat([r['purity'],torch.tensor([[0.,1.,0.,0.,0.,0.,0.]])])
  t=quality_targets(r,SCORE,SPEC);self.assertFalse(t['bad'][:,1].any());self.assertFalse(t['weak'][:,1].any())
 def test_only_hardest_three_errors_keep_pair_gradients(self):
  r=self.example();r['coverage'][:]=1;r['purity'][:]=.5;r['purity'][0,0]=1
  s=dict(SPEC,strong_rank_weight=1.,weak_rank_weight=0.,negative_weight=0.,complete_anchor_weight=0.,rear_anchor_weight=0.,unknown_anchor_weight=0.)
  z=r['class_logits'].clone().requires_grad_();_,detail=quality_loss(z,r,SCORE,s);quality_loss(z,r,SCORE,s)[0].backward()
  self.assertEqual(detail['strong_pairs'],3);self.assertEqual(float(z.grad[6]),0)
 def test_floor_only_for_below_threshold_complete(self):
  r=self.example();r['coverage'][:,1:]=0;r['purity'][:,1:]=0;r['class_logits'][0]=-3;r['scores']=r['class_logits'].sigmoid()
  z=r['class_logits'].clone().requires_grad_();loss,d=quality_loss(z,r,SCORE,SPEC);loss.backward()
  self.assertEqual(d['floor_positives'],1);self.assertLess(float(z.grad[0]),0)
 def test_presence_below_point1_is_not_promised_recovery(self):
  r=self.example();r['presence']=torch.tensor(.05);r['scores']=r['class_logits'].sigmoid()*.05
  _,d=quality_loss(r['class_logits'],r,SCORE,SPEC);self.assertEqual(d['floor_positives'],0)

class SamplingTests(unittest.TestCase):
 def setup_pool(self):
  groups={'hard_multi':['m1','m2'],'hard_single':['s1','s2'],'negative':['m1','n1','n2'],
   'replay_multi':['r1','r2'],'replay_single':['r3','r4']}
  rows=[dict(id=sid,sampling_group='single' if sid.startswith('s') else 'multi') for sid in sorted({x for v in groups.values() for x in v})]
  c=dict(training=dict(seed=42,gradient_accumulation=4),sampling=dict(strategy='quality_stratified'))
  return rows,c,dict(identity=dict(pools=groups))
 def test_quota_distinct_training_only_and_resume_equivalence(self):
  rows,c,p=self.setup_pool();full=stratified_rows(rows,0,40,c,p)
  for i in range(0,40,4):
   b=full[i:i+4];self.assertEqual(len({r['id'] for r in b}),4)
   self.assertEqual([r['_sample_bucket'] for r in b],['hard_multi','hard_single','negative','replay'])
  split=stratified_rows(rows,0,13,c,p)+stratified_rows(rows,13,40,c,p);self.assertEqual(full,split)
  self.assertEqual(full,stratified_rows(rows,0,40,c,p));self.assertFalse(any('_sample_bucket' in r for r in rows))
 def test_validation_ids_are_rejected(self):
  rows,c,p=self.setup_pool();p['identity']['pools']['negative'].append('validation-only')
  with self.assertRaises(ValueError):stratified_rows(rows,0,4,c,p)
 def test_A_exactly_replays_C11_weighted_prefix(self):
  rows,c,p=self.setup_pool();c['sampling']=dict(strategy='weighted',multi_weight=8,single_weight=1,seed_namespace='c2')
  self.assertEqual(FrozenQualityRank().sample_rows(rows,0,400,c),weighted_rows(rows,0,400,42,'c2',8,1))

if __name__=='__main__':unittest.main()
