"""GT-supported quality ranking with shared-bias invariant pair gradients.

The C11 parent-reference format and extractor are deliberately unchanged so that
new score objectives reuse frozen geometry. Unknown independent instances never
receive background or pairwise-negative labels merely for lacking annotations.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .registry import register_method
from .topn_score import FrozenTopNScore, classify, reference_directory, row_key
from ..io import object_sha256, read_json, write_json
from ..training.sampling import weighted_rows


def quality_targets(record, score_spec, spec):
    target=classify(record,score_spec)
    c=record['coverage'].numpy();p=record['purity'].numpy()
    strong=target['bad']
    weak=(((c>=score_spec['merge_coverage']) & (p>score_spec['merge_purity_max']) &
           (p<=spec['weak_quality_max'])) |
          ((p>=score_spec['fragment_purity']) & (c>=score_spec['fragment_coverage_max']) &
           (c<=spec['weak_quality_max'])))
    weak &= ~target['good_any'][None,:] & ~strong.any(0)[None,:]
    weak &= record['nonempty'].numpy()[None,:]
    weak &= ~target['safe'][None,:]
    target['weak']=weak
    z=record['class_logits'].numpy().reshape(-1)
    learnable=np.zeros(len(z),dtype=bool);hard=False
    for g,q in target['primary']:
        strong_violation=strong[g] & (spec['strong_margin']+z-z[q]>0)
        weak_violation=weak[g] & (spec['weak_margin']+z-z[q]>0)
        floor=bool(float(record['presence'].reshape(-1)[0])>.1 and float(record['scores'][q])<=.1)
        if strong_violation.any() or weak_violation.any() or floor:
            learnable[q]=True;hard=True
    target.update(learnable=learnable,hard=hard,
        high_negative=bool((strong.any(0)&(record['scores'].numpy()>score_spec['negative_score_ceiling'])).any()))
    return target


def quality_loss(logits,record,score_spec,spec):
    z=logits.float().reshape(-1);device=z.device;zero=z.sum()*0
    old=record['class_logits'].to(device).float().reshape(-1)
    presence=record['presence'].to(device).float().reshape(-1)[0]
    t=quality_targets(record,score_spec,spec)
    strong=torch.as_tensor(t['bad'],device=device);weak=torch.as_tensor(t['weak'],device=device)
    good=torch.as_tensor(t['good_any'],device=device);protected=torch.as_tensor(t['protected'],device=device)
    learnable=torch.as_tensor(t['learnable'],device=device).clone()
    strong_losses=[];weak_losses=[];floor_losses=[];neg_losses=[];strong_gaps=[];weak_gaps=[]
    strong_pairs=weak_pairs=0
    def hardest(mask):
        idx=mask.nonzero().flatten()
        if not len(idx):return idx
        # Index mining is discrete; BOTH gathered score values keep gradients.
        return idx[torch.topk(z[idx].detach(),min(spec['max_pairs_per_gt'],len(idx))).indices]
    for g,q in t['primary']:
        for mask,margin,losses,gaps,is_strong in [(strong[g],spec['strong_margin'],strong_losses,strong_gaps,True),
                                                 (weak[g],spec['weak_margin'],weak_losses,weak_gaps,False)]:
            idx=hardest(mask)
            if len(idx):
                difference=z[q]-z[idx]
                loss=F.relu(margin-difference).square().mean()
                losses.append(loss);gaps.append(float(difference.detach().min()))
                if bool((difference.detach()<margin).any()):learnable[q]=True
                if is_strong:strong_pairs+=len(idx)
                else:weak_pairs+=len(idx)
        if float(presence)>.1 and (float(record['scores'][q])<=.1 or float(z[q].detach().sigmoid()*presence)<=.1):
            threshold=torch.logit((z.new_tensor(spec['positive_floor_score'])/presence).clamp(1e-5,.99))
            floor_losses.append(F.relu(threshold-z[q]).square());learnable[q]=True
    ceiling=torch.logit((z.new_tensor(score_spec['negative_score_ceiling'])/presence.clamp_min(1e-6)).clamp(1e-5,.99))
    for group in strong:
        idx=hardest(group)
        if len(idx):neg_losses.append(F.relu(z[idx]-ceiling).square().mean())
    def mean(items):return torch.stack(items).mean() if items else zero
    def anchor(mask):return F.smooth_l1_loss(z[mask],old[mask]) if mask.any() else zero
    unknown=~(good|strong.any(0)|weak.any(0)|protected)
    parts=dict(strong_rank=mean(strong_losses),weak_rank=mean(weak_losses),positive_floor=mean(floor_losses),
        negative=mean(neg_losses),complete_anchor=anchor(good&~learnable),rear_anchor=anchor(protected),unknown_anchor=anchor(unknown))
    loss=sum(spec[name+'_weight']*value for name,value in parts.items())
    details={name:float(value.detach()) for name,value in parts.items()}
    details.update(primary=len(t['primary']),hard_positives=int(learnable.sum()),strong_pairs=strong_pairs,
        weak_pairs=weak_pairs,floor_positives=len(floor_losses),confirmed_bad=int(strong.any(0).sum()),
        protected=int(protected.sum()),unknown=int(unknown.sum()),
        minimum_strong_gap=min(strong_gaps,default=0.),minimum_weak_gap=min(weak_gaps,default=0.),
        sample_bucket=record.get('sample_bucket','weighted'))
    return loss,details


def make_sampling_pools(rows,records,score_spec,spec,parent_sha,train_fingerprint):
    pools={k:[] for k in ('hard_multi','hard_single','negative','replay_multi','replay_single')}
    counts=dict(images=0,hard_images=0,negative_images=0)
    for row,record in zip(rows,records):
        if row['id']!=record['id']:raise ValueError('Pool reference identity mismatch')
        target=quality_targets(record,score_spec,spec);role=row['sampling_group'];sid=row['id']
        if target['hard']:pools['hard_'+role].append(sid)
        if target['high_negative']:pools['negative'].append(sid)
        if not target['hard'] and not target['high_negative']:pools['replay_'+role].append(sid)
        counts['images']+=1;counts['hard_images']+=int(target['hard']);counts['negative_images']+=int(target['high_negative'])
    pools={k:sorted(v) for k,v in pools.items()}
    identity=dict(parent_sha256=parent_sha,training_fingerprint=train_fingerprint,
        score_spec=score_spec,quality_spec=spec,pools=pools)
    return dict(status='passed',identity=identity,pool_sha256=object_sha256(identity),counts=counts,
        pool_sizes={k:len(v) for k,v in pools.items()},scope='Training references only. A and B share identical target pools; strategy alone changes draws.')


def stratified_rows(rows,start,end,config,pool):
    if config['training']['gradient_accumulation']!=4:
        raise ValueError('The defined 2:1:1 ablation uses four image exposures/update')
    if start<0 or end<start:raise ValueError('Invalid sampling interval')
    if start==end:return []
    index={r['id']:r for r in rows};p=pool['identity']['pools'];all_hard=p['hard_multi']+p['hard_single']
    if not all_hard or not p['negative'] or not (p['replay_multi'] or p['replay_single']):
        raise ValueError('Stratified pools must provide hard, negative and replay samples')
    if not all(sid in index for group in p.values() for sid in group):raise ValueError('Pool contains non-training IDs')
    result=[]
    for step in range(start//4,(end-1)//4+1):
        rng=random.Random(f"{config['training']['seed']}:quality_stratified:{step}");chosen=set();batch=[]
        replay=p['replay_multi'] if step%2==0 else p['replay_single']
        replay=replay or p['replay_multi'] or p['replay_single']
        groups=[('hard_multi',p['hard_multi'] or all_hard),('hard_single',p['hard_single'] or all_hard),
                ('negative',p['negative']),('replay',replay)]
        for name,group in groups:
            sid=None
            for _ in range(32):
                candidate=rng.choice(group)
                if candidate not in chosen:sid=candidate;break
            if sid is None:
                sid=next((x for x in group if x not in chosen),None)
            if sid is None:raise ValueError('Cannot build distinct four-image batch from pools')
            chosen.add(sid);batch.append(dict(index[sid],_sample_bucket=name))
        result.extend(batch[max(0,start-step*4):min(4,end-step*4)])
    return result


@register_method('quality_rank')
class FrozenQualityRank(FrozenTopNScore):
    def loss(self,output,targets,teacher,config):
        return quality_loss(output.pred_logits,teacher,config['score_calibration'],config['quality_rank'])

    def load_teacher(self,row,config,audit):
        record=super().load_teacher(row,config,audit)
        record['sample_bucket']=row.get('_sample_bucket','weighted')
        return record

    def prepare_teacher(self,config,rows,audit,device):
        started=time.time();resource=super().prepare_teacher(config,rows,audit,device)
        cache=reference_directory(config,audit)
        records=(torch.load(cache/(row_key(r)+'.pt'),weights_only=True) for r in rows)
        pool=make_sampling_pools(rows,records,config['score_calibration'],config['quality_rank'],
            audit['identity']['model_sha256'],audit['identity']['data_fingerprints']['train'])
        path=Path(config['output_dir'])/'sampling_pools.json'
        if path.exists() and read_json(path)['pool_sha256']!=pool['pool_sha256']:
            raise ValueError('Sampling pool identity changed')
        write_json(path,pool)
        resource=dict(resource,wall_seconds=time.time()-started,gpu_count=1 if resource.get('extracted',0) else 0,
            peak_allocated_bytes=resource.get('peak_allocated_bytes',0),peak_reserved_bytes=resource.get('peak_reserved_bytes',0),
            scope='Reuse unchanged C11 frozen C10 geometry/score references; regenerate training-only quality mining pools.')
        write_json(Path(config['output_dir'])/'teacher_resources.json',resource)
        return resource

    def sample_rows(self,rows,start,end,config):
        strategy=config['sampling'].get('strategy','weighted')
        if strategy=='weighted':
            return weighted_rows(rows,start,end,config['training']['seed'],config['sampling']['seed_namespace'],
                config['sampling']['multi_weight'],config['sampling']['single_weight'])
        if strategy!='quality_stratified':raise ValueError('Unknown quality sampling strategy')
        return stratified_rows(rows,start,end,config,read_json(Path(config['output_dir'])/'sampling_pools.json'))
