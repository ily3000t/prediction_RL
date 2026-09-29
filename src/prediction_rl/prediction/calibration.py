"""Frozen episode-max residual bands; not a collision or continuous-action guarantee."""
import math

import torch

from prediction_rl.data.collection_store import read_json, file_hash
from prediction_rl.data.frozen_dataset import inside
from .interface import PredictorConfig
from .supervision import align_supervision
from .epoch_training import subset

SETTINGS={'method':'episode_max_standardized_residual_v1','nominal_coverage':0.9,
          'disagreement_floor_scale':0.1,'scope':'observed_nonterminal_cells_only'}
CHANNEL_SCALES=[10.,3.,5.,2.]


def load_calibration_split(repo,review):
    """Separate consumer: never broaden the training loader's allowed splits."""
    if review['test_locked']is not True: raise ValueError('Need sealed test')
    histories=[]; packs=[]
    for row in review['roots']['calibration']:
        hp,lp=inside(repo,row['history']),inside(repo,row['labels'])
        if file_hash(hp)!=row['history_sha256'] or file_hash(lp)!=row['labels_sha256']:
            raise ValueError('Calibration data changed')
        h,p=read_json(hp),read_json(lp)
        if h['label_pack_sha256']!=file_hash(lp) or inside(repo,h['label_pack'])!=lp:
            raise ValueError('Calibration history/label join changed')
        histories.append(h); packs.append(p)
    return align_supervision(histories,packs,review['manifest'],
        PredictorConfig.from_dict(review['protocol']['model']),split='calibration')


def standardized_residuals(predictions,targets,mask):
    if (predictions.ndim!=6 or predictions.shape[0]!=3 or predictions.shape[1:]!=targets.shape
            or targets.ndim!=5 or targets.shape[-1]!=4 or mask.shape!=targets.shape[:-1]
            or predictions.dtype!=torch.float32 or targets.dtype!=torch.float32 or mask.dtype!=torch.bool
            or predictions.device!=targets.device or targets.device!=mask.device
            or not torch.isfinite(predictions).all() or not torch.isfinite(targets).all()):
        raise ValueError('Expected finite three-member masked physical trajectories')
    # Float64 accumulation avoids float32 score/quantile boundary roundoff.
    values=predictions.double(); mean=values.mean(0)
    scale=torch.maximum(values.std(0,unbiased=False),values.new_tensor(CHANNEL_SCALES)*SETTINGS['disagreement_floor_scale'])
    residual=(targets.double()-mean).abs()/scale
    return torch.where(mask[...,None],residual,0.),scale


def episode_scores(model,batch,episode_ids,*,batch_roots=16,purpose='formal_calibration'):
    expected='calibration' if purpose=='formal_calibration' else 'development'
    if purpose not in ('formal_calibration','development_smoke') or batch.split!=expected:
        raise ValueError('Calibration fitting cannot consume train/validation/test')
    if (type(batch_roots)is not int or batch_roots<1 or len(set(episode_ids))!=len(episode_ids)
            or not set(batch.episode_ids)<=set(episode_ids)): raise ValueError('Invalid episode roster')
    if model.training or torch.get_num_threads()!=1: raise ValueError('Use eval mode and one CPU thread')
    rows={e:{'episode_id':e,'score':None,'valid_actor_time_cells':0,'roots':0,'empty_roots':0} for e in episode_ids}
    root_rows=[]
    with torch.inference_mode():
        for start in range(0,len(batch.root_ids),batch_roots):
            b=subset(batch,list(range(start,min(start+batch_roots,len(batch.root_ids)))))
            predictions=model(b.observations,b.plans)['member_trajectories']
            score,_=standardized_residuals(predictions,b.targets,b.target_mask)
            for i,(root,episode) in enumerate(zip(b.root_ids,b.episode_ids)):
                cells=int(b.target_mask[i].sum()); value=float(score[i].max()) if cells else None
                r=rows[episode]; r['roots']+=1; r['valid_actor_time_cells']+=cells; r['empty_roots']+=int(not cells)
                if value is not None: r['score']=value if r['score']is None else max(r['score'],value)
                root_rows.append({'root_id':root,'episode_id':episode,'score':value,'valid_actor_time_cells':cells})
    return list(rows.values()),root_rows


def fit_episode_quantile(rows,settings):
    if settings!=SETTINGS: raise ValueError('Frozen calibration settings changed')
    if len({r['episode_id'] for r in rows})!=len(rows): raise ValueError('Duplicate calibration episodes')
    values=[r['score'] for r in rows if r['score']is not None]
    if not values or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError('No eligible or invalid calibration episodes')
    n=len(values); rank=math.ceil((n+1)*settings['nominal_coverage'])
    q=sorted(values)[rank-1] if rank<=n else None
    return {'settings':settings,'channel_scales':CHANNEL_SCALES,'eligible_episodes':n,'total_episodes':len(rows),
        'excluded_episode_ids':[r['episode_id'] for r in rows if r['score']is None],
        'order_statistic_one_based':rank,'q':q,'unbounded':q is None,
        'fitted_sample_coverage':sum(q is None or v<=q for v in values)/n,
        'coverage_is_in_sample_not_test':True,'empirical_test_coverage':None}


def band_diagnostics(model,batch,fit,*,batch_roots=16):
    """Descriptive in-fit width/coverage, NEVER held-out calibration evidence."""
    if (batch.split!='calibration' or fit['settings']!=SETTINGS or model.training or torch.get_num_threads()!=1
            or (fit['unbounded']!=(fit['q']is None))
            or (fit['q']is not None and (not math.isfinite(fit['q']) or fit['q']<0))):
        raise ValueError('Wrong calibration diagnostic scope/band')
    widths=torch.zeros(4,dtype=torch.float64); covered=total=0
    with torch.inference_mode():
        for start in range(0,len(batch.root_ids),batch_roots):
            b=subset(batch,list(range(start,min(start+batch_roots,len(batch.root_ids)))))
            predictions=model(b.observations,b.plans)['member_trajectories']
            scores,scale=standardized_residuals(predictions,b.targets,b.target_mask)
            selected=scale[b.target_mask]; total+=selected.shape[0]
            if fit['unbounded']: covered+=selected.numel()
            else:
                widths+=(2*fit['q']*selected).sum(0)
                covered+=int((scores[b.target_mask]<=fit['q']).sum())
    return {'scope':'calibration_fit_sample_only','valid_actor_time_cells':total,
        'coordinate_coverage':covered/(4*total) if total else None,
        'mean_band_width_by_channel':None if fit['unbounded'] or not total else (widths/total).tolist(),
        'width_unbounded':fit['unbounded'],'censored_tail_coverage':'not_identifiable'}
