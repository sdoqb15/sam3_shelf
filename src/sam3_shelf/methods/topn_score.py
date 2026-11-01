"""Frozen-geometry, hard-positive TOP-N calibration with partial-label guards.

This method uses the shared trainer. GT labels define complete instances and a
conservative subset of errors; unannotated candidates never become background
merely because they lack a GT match. Runtime output remains a standard SAM3.
"""
import fcntl
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .registry import register_method
from .. import DEFAULT_PROMPT
from ..data import image_fingerprint, load_mask, read_manifest, validate_manifest
from ..inference import deployment_defaults
from ..io import file_sha256, object_sha256, write_json
from ..metrics import candidate_matrices, at_threshold, aggregate, match_pairs
from ..model import configure_modules, load_model
from ..postprocessing import mask_nms, postprocess


def geometry_digest(masks, boxes, presence_logits):
    digest=hashlib.sha256()
    for value in (np.asarray(masks,dtype=np.bool_),boxes.detach().float().cpu().numpy(),
                  presence_logits.detach().float().cpu().numpy()):
        digest.update(str((value.shape,str(value.dtype))).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def row_key(row):
    if row.get('_cache_key'):
        return row['_cache_key']
    return object_sha256(dict(id=row['id'],group=row['group'],sampling_group=row['sampling_group'],
        rgb=row['image_fingerprint'],masks=[file_sha256(x['mask']) for x in row['instances']]))


def reference_directory(config, audit=None):
    parent_sha=(audit or {}).get('identity',{}).get('model_sha256') or config['score_calibration']['parent_weights_sha256']
    return Path(config['teacher_cache'])/object_sha256(dict(model=parent_sha,
        extraction_source=file_sha256(__file__),format='native200_geometry_v1'))


def classify(record, spec):
    cov=record['coverage'].numpy();pur=record['purity'].numpy()
    nonempty=record['nonempty'].numpy()
    good=(cov>=spec['complete_coverage'])&(pur>=spec['complete_purity'])&nonempty[None,:]
    good_any=good.any(0)
    bad=(((cov>=spec['merge_coverage'])&(pur<=spec['merge_purity_max']))|
        ((cov>=spec['fragment_coverage_min'])&(cov<spec['fragment_coverage_max'])&
         (pur>=spec['fragment_purity'])))&~good_any[None,:]&nonempty[None,:]
    safe=(pur.sum(0)<=spec['safe_gt_overlap'])&(cov.max(0)<=spec['safe_gt_overlap'])&nonempty
    scores=record['scores'].numpy()
    g,q=linear_sum_assignment(-(good.astype(float)*(min(good.shape)+1)+good*scores[None,:]))
    primary=np.array([(int(a),int(b)) for a,b in zip(g,q) if good[a,b]],dtype=np.int64).reshape(-1,2)
    return dict(good=good,good_any=good_any,bad=bad,safe=safe,primary=primary,
                protected=safe&(scores>=spec['protected_min_score']))


def topn_loss(logits, record, spec):
    """Raise only parent-hard positives past detached, reachable rank cutoffs."""
    z=logits.float().reshape(-1);device=z.device;zero=z.sum()*0
    parent=record['class_logits'].to(device).float().reshape(-1)
    presence=record['presence'].to(device).float().reshape(-1)[0]
    t=classify(record,spec)
    good=torch.as_tensor(t['good_any'],device=device)
    bad=torch.as_tensor(t['bad'],device=device)
    protected=torch.as_tensor(t['protected'],device=device)
    keep=record['nms_keep'].to(device).long()
    parent_scores=record['scores'].to(device).float()
    active=torch.zeros_like(good);ranking=[];paired=[];negative=[]
    floor=torch.logit((z.new_tensor(.1)/presence.clamp_min(1e-6)).clamp(1e-5,1-1e-5))
    ng=record['coverage'].shape[0]
    for g,q in t['primary']:
        # Do not penalize complete duplicates of this same GT in the rank cutoff.
        same=torch.as_tensor(t['good'][g],device=device)
        others=keep[~same[keep]]
        for n in spec['top_ns']:
            if ng>n or float(presence)<=.1:
                continue
            parent_rank=1+int((parent_scores[others]>=parent_scores[q]).sum())
            parent_hard=parent_rank>n or float(parent_scores[q])<=.1
            current_rank=1+int((z[others].detach()>=z[q].detach()).sum())
            current_hard=current_rank>n or float((z[q].detach().sigmoid()*presence))<=.1
            if not (parent_hard or current_hard):
                continue
            active[q]=True
            cutoff=torch.topk(z[others],n).values[-1].detach() if len(others)>=n else floor
            cutoff=torch.maximum(cutoff,floor).detach()
            ranking.append(F.relu(cutoff+spec['rank_margin']-z[q]).square())
        wrong=bad[g]
        if wrong.any():
            paired.append(F.relu(spec['error_margin']+z[wrong]-z[q]).square().mean())
    ceiling=torch.logit((z.new_tensor(spec['negative_score_ceiling'])/presence.clamp_min(1e-6)).clamp(1e-5,1-1e-5))
    for wrong in bad:
        if wrong.any():
            negative.append(F.relu(z[wrong]-ceiling).square().mean())
    def mean(items):return torch.stack(items).mean() if items else zero
    def anchor(mask):return F.smooth_l1_loss(z[mask],parent[mask]) if mask.any() else zero
    unknown=~(good|bad.any(0)|protected)
    parts=dict(rank=mean(ranking),error_rank=mean(paired),negative=mean(negative),
               complete_anchor=anchor(good&~active),rear_anchor=anchor(protected),unknown_anchor=anchor(unknown))
    total=sum(spec[k+'_weight']*v for k,v in parts.items())
    return total,dict(**{k:float(v.detach()) for k,v in parts.items()},primary=len(t['primary']),
        hard_positives=int(active.sum()),confirmed_bad=int(bad.any(0).sum()),
        protected=int(protected.sum()),unknown=int(unknown.sum()))


def output_summary(row, record, masks, scores, spec):
    t=classify(record,spec)
    available=np.flatnonzero((scores>.1)&record['nonempty'].numpy())
    keep=available[mask_nms(masks[available],scores[available],.98)]
    a,c,p=(record[k].numpy()[:,keep] for k in ('iou','coverage','purity'))
    score=scores[keep];metric=at_threshold(a,c,p,score,.1)
    top={str(n):at_threshold(a[:,:n],c[:,:n],p[:,:n],score[:n],.1) for n in (3,5,10,50)}
    valid=(c>=spec['complete_coverage'])&(p>=spec['complete_purity'])
    gi,_=match_pairs(np.where(valid,a,0))
    wrong=t['bad'][:,keep];bad_indices=np.flatnonzero(wrong.any(0));inversions=0
    for g in range(len(valid)):
        if valid[g].any() and wrong[g].any():
            inversions+=int(score[wrong[g]].max()>=score[valid[g]].max())
    return dict(metric=metric,top=top,keep=keep,complete_ids=[f'{row["id"]}::gt{g}' for g in gi],
        known_bad=len(bad_indices),bad_in_top5=int((bad_indices<5).sum()),
        bad_in_top10=int((bad_indices<10).sum()),inversions=inversions)


def eligible(current, parent, spec):
    return (set(parent['complete_gt_ids'])<=set(current['complete_gt_ids']) and
        all(current[k]<=parent[k] for k in ('known_bad','bad_in_top5','bad_in_top10','quality_inversion_gt')) and
        set(parent['protected_ids'])<=set(current['protected_ids']) and
        current['rear_score_drop_mean']<=spec['maximum_rear_mean_score_drop'] and
        current['rear_score_drop_max']<=spec['maximum_rear_score_drop'])


@register_method('topn_score')
class FrozenTopNScore:
    needs_teacher=True

    def configure(self, model, config):
        scope=configure_modules(model,('dot_product_scoring.',),expected_parameters=1_182_976)
        return model,scope

    def loss(self, output, targets, teacher, config):
        return topn_loss(output.pred_logits,teacher,config['score_calibration'])

    @deployment_defaults
    def prepare_teacher(self, config, rows, audit, device):
        from PIL import Image
        spec=config['score_calibration'];parent=Path(config['model'])
        if file_sha256(parent/'model.safetensors')!=spec['parent_weights_sha256']:
            raise ValueError('Frozen parent checkpoint identity changed')
        validation=read_manifest(config['data']['validation'],config['data'].get('root'))
        validate_manifest(validation,require_gt=True,require_group=True)
        cache=reference_directory(config,audit);cache.mkdir(parents=True,exist_ok=True)
        with (cache/'prepare.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            all_rows={row_key(r):r for r in [*rows,*validation]}
            missing=[(k,r) for k,r in all_rows.items() if not (cache/(k+'.pt')).exists()]
            if not missing:
                return dict(extracted=0,reused=len(all_rows),cache=str(cache))
            model,processor=load_model(parent,device)
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            for i,(key,row) in enumerate(missing):
                with Image.open(row['image']) as image:
                    x=processor(images=image.convert('RGB'),text=DEFAULT_PROMPT,return_tensors='pt').to(device)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(**x)
                    native=postprocess(processor,output,0.,[(row['height'],row['width'])])[0]
                masks=native['masks'].bool().cpu().numpy();scores=native['scores'].float().cpu().numpy()
                if len(scores)!=output.pred_masks.shape[1]:
                    raise RuntimeError('Frozen references require every raw query in original query order')
                a,c,p=candidate_matrices([load_mask(item['mask']) for item in row['instances']],masks)
                nonempty=masks.reshape(len(masks),-1).any(1);idx=np.flatnonzero(nonempty)
                kept=idx[mask_nms(masks[idx],scores[idx],.98)]
                record=dict(id=row['id'],cache_key=key,class_logits=output.pred_logits[0].float().cpu(),
                    scores=torch.from_numpy(scores),presence=output.presence_logits.sigmoid().float().cpu(),
                    iou=torch.from_numpy(a),coverage=torch.from_numpy(c),purity=torch.from_numpy(p),
                    nonempty=torch.from_numpy(nonempty),nms_keep=torch.from_numpy(kept),
                    geometry_sha256=geometry_digest(masks,output.pred_boxes,output.presence_logits))
                temporary=cache/(key+'.next.pt');torch.save(record,temporary);temporary.replace(cache/(key+'.pt'))
                if i%25==0 or i+1==len(missing):
                    print(f'Frozen score reference {i+1}/{len(missing)}: {row["id"]}',flush=True)
                del x,output,native,masks,record
            resource=dict(device=str(device),gpu_count=1,extracted=len(missing),reused=len(all_rows)-len(missing),
                wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),cache=str(cache),
                scope='Frozen parent raw-query geometry and score references for train plus validation; no test labels; no raw mask tensors persisted.')
            write_json(Path(config['output_dir'])/'teacher_resources.json',resource)
            return resource

    def load_teacher(self,row,config,audit):
        record=torch.load(reference_directory(config,audit)/(row_key(row)+'.pt'),weights_only=True)
        if record['id']!=row['id']:raise ValueError('Reference row identity mismatch')
        return record

    @deployment_defaults
    @torch.inference_mode()
    def validate(self,model,processor,rows,config):
        from PIL import Image
        spec=config['score_calibration'];samples=[];tops={str(n):[] for n in (3,5,10,50)}
        current=dict(complete_gt_ids=[],protected_ids=[],known_bad=0,bad_in_top5=0,bad_in_top10=0,quality_inversion_gt=0)
        parent={k:[] if isinstance(v,list) else 0 for k,v in current.items()}
        drops=[];details=[];parent_samples=[];parent_tops={str(n):[] for n in (3,5,10,50)}
        for row in rows:
            record=self.load_teacher(row,config,{})
            with Image.open(row['image']) as image:
                x=processor(images=image.convert('RGB'),text=DEFAULT_PROMPT,return_tensors='pt').to(model.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(**x);native=postprocess(processor,output,0.,[(row['height'],row['width'])])[0]
            masks=native['masks'].bool().cpu().numpy();scores=native['scores'].float().cpu().numpy()
            if geometry_digest(masks,output.pred_boxes,output.presence_logits)!=record['geometry_sha256']:
                raise RuntimeError(f'Frozen mask/box/presence changed: {row["id"]}')
            now=output_summary(row,record,masks,scores,spec)
            old=output_summary(row,record,masks,record['scores'].numpy(),spec)
            samples.append(dict(id=row['id'],**now['metric']));parent_samples.append(old['metric'])
            for n in tops:tops[n].append(now['top'][n]);parent_tops[n].append(old['top'][n])
            for result,total in ((now,current),(old,parent)):
                total['complete_gt_ids']+=result['complete_ids']
                for key in ('known_bad','bad_in_top5','bad_in_top10'):total[key]+=result[key]
                total['quality_inversion_gt']+=result['inversions']
            t=classify(record,spec)
            refs=[int(q) for q in old['keep'] if t['protected'][q]]
            for q in refs:
                name=f'{row["id"]}::q{q}';parent['protected_ids'].append(name)
                retained=q in now['keep']
                if not retained and len(now['keep']):
                    other=masks[now['keep']];inter=(other&masks[q]).sum((1,2));union=(other|masks[q]).sum((1,2))
                    retained=bool((inter/np.maximum(union,1)>.98).any())
                if retained:current['protected_ids'].append(name)
            rear=[q for q in old['keep'] if t['safe'][q]]
            drops.extend(np.maximum(record['scores'].numpy()[rear]-scores[rear],0).tolist())
            details.append(dict(id=row['id'],known_bad=now['known_bad'],bad_in_top5=now['bad_in_top5'],
                bad_in_top10=now['bad_in_top10'],complete_gt=now['metric']['complete_gt']))
        current.update(**aggregate(samples),top_n_metrics={n:aggregate(v) for n,v in tops.items()},samples=samples,
            rear_score_drop_mean=float(np.mean(drops)) if drops else 0.,rear_score_drop_max=max(drops,default=0.),
            raw_geometry_verified_images=len(rows),quality_details=details)
        parent.update(**aggregate(parent_samples),top_n_metrics={n:aggregate(v) for n,v in parent_tops.items()})
        current['eligible']=eligible(current,parent,spec)
        current['lost_parent_gt']=sorted(set(parent['complete_gt_ids'])-set(current['complete_gt_ids']))
        current['parent_baseline']=parent
        return current

    def selection_value(self,metrics):
        if not metrics['eligible']:return (-1,)
        return (metrics['top_n_metrics']['5']['complete_gt']+metrics['top_n_metrics']['10']['complete_gt'],
            metrics['complete_gt'],-metrics['bad_in_top5'],-metrics['bad_in_top10'],-metrics['known_bad'],
            -metrics['quality_inversion_gt'],metrics['successful_images'],-metrics['mean_candidates'])


def audit_frozen_checkpoint(parent,child):
    """Verify exported standard checkpoints differ exclusively in scoring tensors."""
    from safetensors import safe_open
    changed=[];unchanged=0
    with safe_open(str(Path(parent)/'model.safetensors'),framework='pt',device='cpu') as a,\
         safe_open(str(Path(child)/'model.safetensors'),framework='pt',device='cpu') as b:
        if set(a.keys())!=set(b.keys()):raise ValueError('Checkpoint tensor keys changed')
        for key in a.keys():
            if torch.equal(a.get_tensor(key),b.get_tensor(key)):unchanged+=1
            elif key.startswith('dot_product_scoring.'):changed.append(key)
            else:raise ValueError(f'Frozen tensor changed: {key}')
    return dict(status='passed',unchanged_tensors=unchanged,changed_scoring_tensors=changed,
                parent_sha256=file_sha256(Path(parent)/'model.safetensors'),
                weights_sha256=file_sha256(Path(child)/'model.safetensors'))
