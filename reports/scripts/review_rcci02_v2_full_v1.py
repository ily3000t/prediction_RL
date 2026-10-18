"""Read-only reproduction of the 2026-10-18 full-cohort result review.

No run/retry, weights, inference, simulation, artifact edits or new cohorts.
The additional intersection is scalar availability, NOT seed/outcome selection.
"""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import validate_rcci_v2_cohort as audit

REQUEST_HASH = '68b9f264739f07fd77df81cb2315eea7f058b542d184cf4d5db798a45e47ec59'
REPORT_SHA256 = 'efe1c5eede5ad69552adacca3002f066ab93cdec18287161f7a201b88ab821f7'
INDEX_SHA256 = '4c1552e280ae66f724302241516dd958420d01717a94a63dc6d0034c9e336c92'


def require(value):
    if not value: raise ValueError('Immutable result-review identity check failed')


def paired_gap(rows,role,time=None):
    key = role+'_neighbor_only_min_gap_m'; pairs = []; episodes = defaultdict(lambda:[[],[]])
    for row in rows:
        for branch in row['branches']:
            record = branch['following'] if time is None else branch['fixed_horizons'][time]['methods']
            if not record: continue
            ordinary = record['ordinary']['errors'].get(key,[])
            conditional = record['conditional']['errors'].get(key,[])
            if len(ordinary) == len(conditional) == 1:
                values = abs(ordinary[0]),abs(conditional[0]); pairs.append(values)
                for i,value in enumerate(values): episodes[row['episode_id']][i].append(value)
    return {'paired_candidates':len(pairs),'episodes':len(episodes),
        'ordinary_mae_m':None if not pairs else float(np.mean([p[0] for p in pairs])),
        'conditional_mae_m':None if not pairs else float(np.mean([p[1] for p in pairs])),
        'ordinary_equal_episode_mae_m':None if not episodes else float(np.mean([np.mean(v[0]) for v in episodes.values()])),
        'conditional_equal_episode_mae_m':None if not episodes else float(np.mean([np.mean(v[1]) for v in episodes.values()]))}


def summarize_group(rows,summary):
    failures = {m:{'ego':Counter(),'actor':Counter()} for m in audit.MODES}
    selected = {m:Counter() for m in audit.MODES}
    for row in rows:
        for mode in audit.MODES:
            for candidate in row['coverage'][mode]['v2_candidates']:
                for role,items in (('ego',[candidate['ego_prefix']]),('actor',candidate['actor_prefixes'].values())):
                    for p in items:
                        if not p['full_5s']: failures[mode][role][p['first_unavailable_reason']] += 1
                cross = candidate['conflict']; actor = cross['selected_actor']
                if actor is None: continue
                pair = cross['pairs'][actor]; c = selected[mode]; c['selected'] += 1
                c['positive_overlap'] += int(pair['supported_overlap_s'] > 0)
                c['actor_at_root'] += int(pair['actor']['left_censored'])
                c['actor_entire_prefix'] += int(pair['actor']['left_censored'] and pair['actor']['right_censored'])
                c['first_equals_ego_entry'] += int(pair['first_overlap_s'] is not None
                    and abs(pair['first_overlap_s']-pair['ego']['entry_s']) <= 1e-8)
    return {'roots':len(rows),'episodes':len({r['episode_id'] for r in rows}),
        'paired_gaps':{role:{time or 'common_prefix':paired_gap(rows,role,time)
                             for time in (None,'1.0','3.0','5.0')} for role in ('front','rear')},
        'failure_attribution':failures,'selected_cross':selected,
        'frozen_report_summary':summary,'independent_confirmation':False}


def review():
    directory = ROOT/'artifacts/rcci02_v2_cohort/rcci02_v2_full_v1'
    q = audit.read(directory/'request.json'); report = audit.read(directory/'report.json')
    receipt = audit.read(directory/'receipt.json'); index = audit.read(directory/'index.json')
    require(not (directory/'writer.lock').exists())
    require(audit.digest(q) == REQUEST_HASH == report['request_hash'] == receipt['request_hash'])
    require(audit.sha(directory/'report.json') == REPORT_SHA256 == receipt['report_sha256'])
    require(audit.sha(directory/'index.json') == INDEX_SHA256 == receipt['index_sha256'] == report['index_sha256'])
    require(q['source_hashes'] == audit.shared.sources() and q['environment'] == audit.shared.environment())
    require(audit.sha(audit.inside(ROOT,q['config_path'])) == q['config_sha256'])
    require(audit.sha(audit.inside(ROOT,q['config']['design_contract'])) == q['design_contract_sha256'])
    with patch('torch.load',side_effect=RuntimeError('No weight loading in result review')):
        roots,data,hashes,parents,accepted,_,design = audit.prerequisites(q['config'],q['baseline_root'])
    require(roots == q['roots'] and hashes == q['input_hashes'] and parents == q['parent_artifact_hashes'])
    require(accepted == q['training_acceptance'] and design == q['design_episodes'])
    require(audit.digest(data['manifest']) == q['manifest_hash'])
    require(report['status'] == 'complete' and report['roots'] == 890 and report['episodes'] == 320)
    require(report['ensemble_forward_calls'] == 1780 and report['member_forward_calls'] == 5340)
    require(index['root_ids'] == [r['root_id'] for r in roots] and len(index['root_files']) == len(roots))
    rows = []
    for (name,h),source in zip(index['root_files'].items(),roots):
        file = audit.inside(directory,name); require(audit.sha(file) == h)
        row = audit.read(file)['diagnostics']
        require(all(row[k] == source[k] for k in ('root_id','episode_id','split')))
        require(row['design_episode_reuse'] == (row['episode_id'] in design)); rows.append(row)
    require(audit.aggregate(rows) == report['summaries'])
    start = audit.read(directory/'started.json')
    return {'version':'rcci02_v2_full_review_20261018','request_hash':REQUEST_HASH,'report_sha256':REPORT_SHA256,
        'exact_original_aggregation_verified':True,'verified_root_files':len(rows),'verified_input_files':len(hashes),
        'new_weight_loads':0,'new_forward_calls':0,'simulation_episodes':0,'training':False,
        'actual_original_elapsed_s':(datetime.fromisoformat(report['finished_at'])-datetime.fromisoformat(start['timestamp'])).total_seconds(),
        'groups':{name:summarize_group([r for r in rows if r['split'] == name.split('/')[0]
                    and r['design_episode_reuse'] == name.endswith('/design_episode_reuse')],summary)
                    for name,summary in report['summaries'].items()},
        'paired_gap_intersection':'same_candidate_time_support_and_both_scalars_available',
        'time_errors_are_available_event_conditional_not_always_identical_event_hit_sets':True,
        'method_effect_established':False,'independent_confirmation':False}


if __name__ == '__main__':
    print(json.dumps(review(),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
