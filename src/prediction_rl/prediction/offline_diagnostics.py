"""Descriptive validation metrics and frozen-band attribution, not action ranking."""
from collections import Counter
import itertools
import math

import torch

from .calibration import standardized_residuals,CHANNEL_SCALES

FIELDS=('scaled_mse','ade_m','last_observed_displacement_m','fde_5s_m','x_mae_m','y_mae_m',
        'speed_mae_mps','acceleration_mae_mps2','pairwise_response_delta_scaled_mse')


def average(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def root_metrics(prediction,batch):
    target=batch.targets;mask=batch.target_mask
    if (prediction.shape!=target.shape or prediction.dtype!=torch.float32
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError('Invalid point predictions/targets')
    delta=prediction.double()-target.double();scales=delta.new_tensor(CHANNEL_SCALES); rows=[]
    for i,root in enumerate(batch.root_ids):
        candidates=[]
        for c in range(5):
            valid=mask[i,c]
            if not valid.any(): continue
            err=delta[i,c]; dist=err[...,:2].square().sum(-1).sqrt()
            final=[]
            for k in range(valid.shape[0]):
                times=valid[k].nonzero().flatten()
                if len(times):final.append(float(dist[k,int(times[-1])]))
            mae=err[valid].abs().mean(0).tolist()
            candidates.append({'scaled_mse':float((err[valid]/scales).square().mean()),
                'ade_m':float(dist[valid].mean()),'last_observed_displacement_m':average(final),
                'fde_5s_m':float(dist[:,-1][valid[:,-1]].mean()) if valid[:,-1].any() else None,
                **dict(zip(FIELDS[4:8],mae))})
        contrasts=[]; contrast_cells=0
        for a,b in itertools.combinations(range(5),2):
            valid=mask[i,a]&mask[i,b]
            if valid.any():
                err=(prediction[i,a].double()-prediction[i,b].double())-(target[i,a].double()-target[i,b].double())
                contrasts.append(float((err[valid]/scales).square().mean()));contrast_cells+=int(valid.sum())
        rows.append({'root_id':root,'episode_id':batch.episode_ids[i],
            'valid_actor_time_cells':int(mask[i].sum()),'valid_candidate_pairs':len(contrasts),
            'pair_common_actor_time_cells':contrast_cells,'full_horizon_actor_branches':int(mask[i,:,:,-1].sum()),
            **{key:average([x[key] for x in candidates]) for key in FIELDS[:-1]},
            FIELDS[-1]:average(contrasts)})
    return rows


def summarize(rows,episode_roster):
    if len({r['root_id'] for r in rows})!=len(rows) or not set(r['episode_id'] for r in rows)<=set(episode_roster):
        raise ValueError('Duplicate roots or wrong episode roster')
    episode_rows=[{'episode_id':e,'roots':sum(r['episode_id']==e for r in rows),
        **{k:average([r[k] for r in rows if r['episode_id']==e]) for k in FIELDS}} for e in episode_roster]
    return {'root_equal':{k:average([r[k] for r in rows]) for k in FIELDS},
        'episode_equal':{k:average([r[k] for r in episode_rows]) for k in FIELDS},
        'episode_metrics':episode_rows,'roots':len(rows),'episodes':len(episode_roster),
        'empty_roots':sum(not r['valid_actor_time_cells'] for r in rows),
        'root_denominators':{k:sum(r[k]is not None for r in rows) for k in FIELDS},
        'valid_actor_time_cells':sum(r['valid_actor_time_cells'] for r in rows),
        'full_horizon_actor_branches':sum(r['full_horizon_actor_branches'] for r in rows)}


def paired_difference(left,right,episode_roster):
    if [(r['root_id'],r['episode_id']) for r in left]!=[(r['root_id'],r['episode_id']) for r in right]:
        raise ValueError('Paired root/episode identities differ')
    output={}
    for field in FIELDS:
        pairs=[(a,b) for a,b in zip(left,right) if a[field]is not None and b[field]is not None]
        per_episode=[average([b[field]-a[field] for a,b in pairs if a['episode_id']==e]) for e in episode_roster]
        output[field]={'paired_roots':len(pairs),'root_mean_right_minus_left':average([b[field]-a[field] for a,b in pairs]),
            'paired_episodes':sum(v is not None for v in per_episode),'episode_mean_right_minus_left':average(per_episode)}
    return output


def residual_attribution(predictions,batch,fit,actor_ids):
    residual,scale=standardized_residuals(predictions,batch.targets,batch.target_mask)
    mean=predictions.double().mean(0);spread=predictions.double().std(0,unbiased=False)
    floor=spread.new_tensor(CHANNEL_SCALES)*.1
    channel_names=('x','y','speed','acceleration'); rows=[]
    if fit['unbounded']!=(fit['q']is None) or (fit['q']is not None and (not math.isfinite(fit['q']) or fit['q']<0)):
        raise ValueError('Invalid frozen calibration quantile')
    for i,root in enumerate(batch.root_ids):
        valid=batch.target_mask[i];cells=int(valid.sum())
        row={'root_id':root,'episode_id':batch.episode_ids[i],'valid_actor_time_cells':cells,'maximum':None}
        if cells:
            # Argmax over valid cells only, including when every valid residual is zero.
            masked=residual[i].masked_fill(~valid[...,None],-1.)
            flat=int(masked.argmax());c,k,t,d=[int(x) for x in torch.unravel_index(torch.tensor(flat),masked.shape)]
            row.update(maximum={'candidate_id':c,'actor_slot':k,'actor_id':actor_ids[i][k],
                'time_index':t,'horizon_s':(t+1)*.2,'channel':channel_names[d],
                'standardized_residual':float(residual[i,c,k,t,d]),'truth':float(batch.targets[i,c,k,t,d]),
                'mean_prediction':float(mean[i,c,k,t,d]),'population_std':float(spread[i,c,k,t,d]),
                'scale':float(scale[i,c,k,t,d])},
                max_by_channel=residual[i][valid].max(0).values.tolist(),
                absolute_error_sum=(batch.targets[i].double()-mean[i]).abs()[valid].sum(0).tolist(),
                floor_active_counts=(spread[i][valid]<=floor).sum(0).tolist(),
                covered_coordinate_counts=([cells]*4 if fit['unbounded'] else (residual[i][valid]<=fit['q']).sum(0).tolist()),
                width_sum=(None if fit['unbounded'] else (2*fit['q']*scale[i][valid]).sum(0).tolist()))
        rows.append(row)
    return rows


def summarize_attribution(rows,roster,fit,split):
    eligible=[r for r in rows if r['maximum']is not None]; cells=sum(r['valid_actor_time_cells'] for r in rows)
    episodes=[]
    for e in roster:
        candidates=[r for r in eligible if r['episode_id']==e]
        worst=max(candidates,key=lambda r:r['maximum']['standardized_residual']) if candidates else None
        episodes.append({'episode_id':e,'worst_root':worst,
            'all_observed_coordinates_covered':None if worst is None else
                (fit['unbounded'] or worst['maximum']['standardized_residual']<=fit['q'])})
    def sum_vector(key):return [sum(r[key][k] for r in eligible) for k in range(4)]
    covered=[x['all_observed_coordinates_covered'] for x in episodes if x['all_observed_coordinates_covered']is not None]
    return {'scope':split+'_observed_nonterminal_only','independent_test_evidence':False,
        'episode_coverage':average(covered),'eligible_episodes':len(covered),'empty_episodes':len(roster)-len(covered),
        'worst_channel_episode_counts':dict(Counter(x['worst_root']['maximum']['channel'] for x in episodes if x['worst_root'])),
        'floor_active_fraction_by_channel':[x/cells for x in sum_vector('floor_active_counts')] if cells else None,
        'coordinate_coverage_by_channel':[x/cells for x in sum_vector('covered_coordinate_counts')] if cells else None,
        'mean_width_by_channel':None if fit['unbounded'] or not cells else [x/cells for x in sum_vector('width_sum')],
        'episode_worst_cases':episodes,'valid_actor_time_cells':cells,
        'horizon_worst_episode_counts':dict(Counter(str(x['worst_root']['maximum']['horizon_s']) for x in episodes if x['worst_root']))}
