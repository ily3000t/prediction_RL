"""Manual-start collection releases; formal test execution is deliberately absent."""
from copy import deepcopy

from .dataset_contract import digest
from .protocol_plan import build_plan, keys

SPLITS = ('train', 'validation', 'calibration')


def collection_plan(protocol):
    p = deepcopy(protocol)
    execution = p.pop('execution', None)
    keys(execution, 'contract allowed_splits worker_wall_limit_s require_hash_confirmation')
    if (execution['contract'] != 'response_collection_manual_start_v1'
            or execution['allowed_splits'] != list(SPLITS)
            or type(execution['worker_wall_limit_s']) is not int or execution['worker_wall_limit_s'] != 300
            or execution['require_hash_confirmation'] is not True):
        raise ValueError('Unsupported execution release; test must remain locked')
    if p['review_status'] != 'collection_manual_start_required':
        raise ValueError('An old draft is not an executable release')
    p['review_status'] = 'draft_not_approved'
    plan = build_plan(p)
    return {'schema_version': 1, 'protocol_hash': digest(protocol), 'resolved_protocol': deepcopy(protocol),
            'split_manifest': plan['split_manifest'], 'jobs': plan['collection_jobs'], 'budget': plan['budget'],
            'status': 'awaiting_manual_start', 'formal_training_ready': False, 'test_locked': True}


def selected_jobs(request, splits, confirmed_hash):
    if confirmed_hash != digest(request):
        raise ValueError('Explicit confirmation must match the exact immutable request hash')
    if not isinstance(splits, list) or not splits or len(set(splits)) != len(splits) or any(s not in SPLITS for s in splits):
        raise ValueError('Choose unique train/validation/calibration splits; test is locked')
    plan = collection_plan(request['protocol'])
    if request['plan'] != plan:
        raise ValueError('Request job list, seeds or split manifest changed')
    return [j for j in plan['jobs'] if j['split'] in splits]


def validate_job_metadata(job, metadata, target_ids):
    if (metadata['status'] != 'complete' or metadata['seed'] != job['simulator_seed']
            or metadata['episode_id'] != job['episode_id'] or metadata['split'] != job['split']):
        raise ValueError('Episode receipt identity mismatch')
    rows = metadata['roots']
    if [r['target'] for r in rows] != target_ids:
        raise ValueError('Missing/reordered root accounting')
    root_ids = []
    for row in rows:
        if row['status'] == 'complete':
            if row['candidate_count'] != 5 or row['branch_executions'] != 10 or not row['future_repeat_exact']:
                raise ValueError('Incomplete/unverified candidate family')
            root_ids.append(row['root_id'])
        elif row['status'] == 'unavailable':
            if not row.get('reason') or row['candidate_count'] != 0:
                raise ValueError('Unavailable root requires explicit reason and zero candidates')
        else:
            raise ValueError('Pending/failed root cannot be a completed episode')
    if len(root_ids) != len(set(root_ids)):
        raise ValueError('Duplicate reached root identity')


def summarize(jobs_and_metadata, targets):
    """All outcomes are data, never a performance acceptance gate."""
    locations = {t: {'reached': 0, 'unavailable': 0, 'reasons': {}} for t in targets}
    terms = {}
    roots = candidates = executions = frames = omitted = cells = valid_cells = empty_candidates = 0
    empty_episodes = 0
    for job, metadata in jobs_and_metadata:
        validate_job_metadata(job, metadata, targets)
        reached = 0
        for row in metadata['roots']:
            loc = locations[row['target']]
            if row['status'] == 'unavailable':
                loc['unavailable'] += 1
                loc['reasons'][row['reason']] = loc['reasons'].get(row['reason'], 0) + 1
                continue
            loc['reached'] += 1; reached += 1; roots += 1
            candidates += row['candidate_count']; executions += row['branch_executions']
            frames += row['future_observed_frames']; omitted += row['omitted_neighbors']
            cells += row['response']['paired_nonterminal_neighbor_cells']
            valid_cells += row['valid_selected_actor_time_cells']
            empty_candidates += row['empty_trajectory_candidates']
            for reason, count in row['termination_counts'].items():
                terms[reason] = terms.get(reason, 0) + count
        empty_episodes += reached == 0
    return {'episodes': len(jobs_and_metadata), 'episodes_without_reached_roots': empty_episodes,
            'reached_roots': roots, 'unique_candidates': candidates, 'branch_executions': executions,
            'observed_future_frames': frames, 'summed_root_omitted_neighbors': omitted,
            'paired_nonterminal_neighbor_cells': cells, 'locations': locations, 'termination_counts': terms,
            'valid_selected_actor_time_cells': valid_cells, 'empty_trajectory_candidates': empty_candidates,
            'formal_training_ready': False, 'coverage_review_required': True}
