"""Compare interface semantics on immutable saved means, not method effects."""
from collections import Counter

import numpy as np

from prediction_rl.evaluation.rcci_frozen_audit import MODES

VERSION = 'rcci02_v2_cached_prefix_audit_v1'
SETTINGS = {'version':VERSION,'design_contract':'docs/design/rcci02_adapter_contract_v2.md',
    'parent_request':'artifacts/rcci02_diag/rcci02_diag_v1/request.json',
    'parent_request_hash':'597cfcb0f7cea1bf976c8dd36792fb56185a2436af3fc13f21e39e36eafb6d0f',
    'parent_report_sha256':'213bbd86f043d642418619677bd34a00312748292c6b98b53da04cb52cc64a65',
    'parent_forecasts_sha256':'0a7a01d9f2df08c108f0e3a65128ead1d61248866b4eee6c0902098feb515128',
    'parent_diagnostics_sha256':'db98e913cdcbf464f2b27c080d4286e199520fc67e1e76d9a43fba0e572b1616',
    'scope':'same_six_cached_roots_no_expansion','splits':['train','validation'],
    'model_weight_loading':False,'model_forward_calls':0,'simulation':False,'training':False,
    'calibration_target_access':False,'test_target_access':False,
    'native_targets_used_for_v2_features':False,'adapter_version':'rcci02_a_supported_prefix_control_volume_v2'}


def compare(v1, v2):
    old = np.asarray(v1['features'],dtype=np.float32); new = np.asarray(v2['features'],dtype=np.float32)
    if old.shape != (149,) or new.shape != (149,) or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Need two finite five-probe interfaces')
    if not np.array_equal(old[-9:],new[-9:]): raise ValueError('Root metadata changed')
    a,b = old[:140].reshape(5,28),new[:140].reshape(5,28)
    if (not np.array_equal(a[:,4],b[:,6]) or not np.array_equal(a[:,10],b[:,13])
            or not np.array_equal(a[:,18],b[:,18]) or not np.array_equal(a[:,19],b[:,27])):
        raise ValueError('Do not turn partial support into full-5s coverage')
    for c in range(5):
        if a[c,4] and a[c,3] and not np.array_equal(a[c,:3],b[c,:3]):
            raise ValueError('Full-support front consequence changed unexpectedly')
        if a[c,10] and a[c,9] and not np.array_equal(a[c,6:9],b[c,7:10]):
            raise ValueError('Full-support rear consequence changed unexpectedly')
    counts = Counter(candidate_tables=5,full_follow_pool_5s=int(a[:,4].sum()),
        cross_applicable=int(a[:,18].sum()),full_cross_pool_5s=int(a[:,19].sum()),
        v1_front_min_available=int((a[:,3]*a[:,4]).sum()),v2_supported_front_min_available=int(b[:,4].sum()),
        v1_rear_min_available=int((a[:,9]*a[:,10]).sum()),v2_supported_rear_min_available=int(b[:,11].sum()),
        v1_complete_cross_overlap_available=int(a[:,26].sum()),
        v2_supported_overlap_available=int(b[:,26].sum()),
        v2_supported_positive_overlap=int(((b[:,16] > 0)&(b[:,26] > 0)).sum()),
        v2_front_available_with_unknown_suffix=int(((b[:,4] > 0)&(b[:,6] == 0)).sum()),
        v2_rear_available_with_unknown_suffix=int(((b[:,11] > 0)&(b[:,13] == 0)).sum()),
        v2_cross_available_with_unknown_suffix=int(((b[:,26] > 0)&(b[:,27] == 0)).sum()),
        v2_follow_root_only=int((b[:,5] == 0).sum()),
        v2_cross_root_only=int(((b[:,18] > 0)&(b[:,19] == 0)).sum()),
        v2_front_min_at_observed_root=int(((b[:,4] > 0)&(b[:,2] == 0)).sum()),
        v2_rear_min_at_observed_root=int(((b[:,11] > 0)&(b[:,9] == 0)).sum()),
        v2_selected_ego_entry_at_root=int(((b[:,20] > 0)&(b[:,22] > 0)).sum()),
        v2_selected_actor_entry_at_root=int(((b[:,21] > 0)&(b[:,23] > 0)).sum()),
        v2_selected_cross_actor_changed=sum(x['conflict']['selected_actor'] != y['conflict']['selected_actor']
                                            for x,y in zip(v1['candidates'],v2['candidates'])))
    return {'counts':dict(counts),'v2_features':new.tolist(),
        'full_5s_support_unchanged':True,'comparison_is_prediction_accuracy':False,
        'following_prefix_horizons_s':[x['front']['shared_prefix_horizon_s'] for x in v2['candidates']],
        'cross_prefix_horizons_s':[x['conflict']['shared_prefix_horizon_s'] for x in v2['candidates']
                                    if x['conflict']['selected_actor'] is not None],
        'v2_candidates':v2['candidates']}


def describe(values):
    x = np.asarray(values,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite prefix support statistic')
    return {'count':len(x),'min':None if not len(x) else float(x.min()),
            'median':None if not len(x) else float(np.median(x)),
            'max':None if not len(x) else float(x.max())}


def aggregate(rows):
    output = {}
    for split in SETTINGS['splits']:
        group = [r for r in rows if r['split'] == split]; methods = {}
        for mode in MODES:
            counts = Counter()
            for row in group: counts.update(row['methods'][mode]['counts'])
            methods[mode] = {'counts':dict(counts),
                'following_prefix_horizon_s':describe([v for r in group for v in r['methods'][mode]['following_prefix_horizons_s']]),
                'cross_prefix_horizon_s':describe([v for r in group for v in r['methods'][mode]['cross_prefix_horizons_s']])}
        output[split] = {'roots':len(group),'unique_episodes':len({r['episode_id'] for r in group}),
            'methods':methods,'independent_scenarios_not_candidate_actor_time':True}
    return output
