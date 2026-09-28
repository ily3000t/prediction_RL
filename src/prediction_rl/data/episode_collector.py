"""One isolated fresh-seed episode; no dependence on historical label artifacts."""
from datetime import datetime, timezone
import math
from pathlib import Path
import time

from prediction_rl.envs.upstream import upstream_session
from .actor_adapter import build_actor_inputs
from .branching import ReplayBrancher, fingerprint
from .collection_store import write_once, file_hash, seal_episode
from .dataset_contract import build_branch_labels, digest, episode_id
from .merge_geometry import MergeGeometry
from .reference_policy import AuthorDDPGReference
from .response_collection import branch_with_geometry, root_accounting


def now():
    return datetime.now(timezone.utc).isoformat()


def response_magnitude(branches, root_actors):
    """Descriptive common-time, nonterminal, same-neighbor differences; no gate."""
    from itertools import combinations
    grids = {k: {f['info']['execution_audit']['after']['simulation_time_s']:
                 f['info']['execution_audit']['after']['vehicles'] for f in trace if not f['done']}
             for k, trace in branches.items()}
    position = speed = 0.0
    cells = 0
    for a, b in combinations(grids.values(), 2):
        for t in a.keys() & b.keys():
            for actor in a[t].keys() & b[t].keys() & (set(root_actors) - {'ego'}):
                cells += 1
                position = max(position, math.dist(a[t][actor]['position'], b[t][actor]['position']))
                speed = max(speed, abs(a[t][actor]['speed'] - b[t][actor]['speed']))
    return {'paired_nonterminal_neighbor_cells': cells,
            'max_neighbor_position_delta_m': position if cells else None,
            'max_neighbor_speed_delta_mps': speed if cells else None}


def collect_episode(repo, baseline, collection, network, job, output, binding, *, audit=None, before_seal=None):
    """Only a parent with a reviewed manifest should call this engine.

    audit is a development-only parity observer. It cannot provide inputs/labels.
    All data are newly constructed from replay, never copied from prior artifacts.
    """
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if not output.is_relative_to(repo):
        raise ValueError('Episode output escaped project')
    seed, split = job['simulator_seed'], job['split']
    episode = episode_id(baseline['episode_group_namespace_sha256'], seed, 0)
    if job['episode_id'] != episode or split not in ('development', 'train', 'validation', 'calibration'):
        raise ValueError('Invalid episode identity or locked test split')
    if collection['episode_index'] != 0 or collection['backend'] != ReplayBrancher.backend:
        raise ValueError('Only the audited first-episode replay backend is supported')
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / 'started.json', {'job': job, 'binding': binding, 'started_at': now()})
    progress = []
    started = time.perf_counter()
    geometry = MergeGeometry(repo / network)
    try:
        with upstream_session(repo / 'RL-MPC-LaneMerging-master', repo / baseline['upstream_config'], seed) as settings:
            write_once(output / 'settings.json', settings.export_settings())
            low, high = settings.MINIMUM_NEGATIVE_JERK, settings.MAXIMUM_POSITIVE_JERK
            probes = [low + f*(high-low) for f in (0, .25, .5, .75, 1)]
            if (probes != collection['probe_jerks'] or settings.TICK_LENGTH != collection['tick_s']
                    or not low <= collection['continuation_jerk'] <= high or collection['branch_repeats'] != 2):
                raise ValueError('Requested controls differ from accepted upstream semantics')
            policy = AuthorDDPGReference(repo / baseline['reference_checkpoint'], baseline['reference_sha256'],
                                         'cuda' if settings.CUDA else 'cpu')
            manager = ReplayBrancher()
            try:
                def discover():
                    return manager.discover_lane_roots(collection['root_targets'], collection['discovery_max_steps'], None, policy=policy)
                discovery = discover()
                if discovery != discover():
                    raise AssertionError('Reference discovery changed on repeat')
                write_once(output / 'discovery.json', discovery)
                progress = root_accounting(discovery['targets'])
                if [r['target'] for r in progress] != [r['id'] for r in collection['root_targets']]:
                    raise AssertionError('Discovery lost or reordered a predeclared target')
                if audit is not None:
                    audit('discovery', discovery)
                for number, planned in enumerate(discovery['targets']):
                    directory = output / f'r{number}'
                    if planned['prefix_steps'] is None:
                        write_once(directory / 'accounting.json', progress[number])
                        continue
                    root, history = manager.capture_with_history(planned['prefix_actions'], collection['history_steps'])
                    geometry.verify_runtime()
                    if root.traffic != planned['selection']['traffic']:
                        raise AssertionError('Discovered root changed after replay')
                    repeat, repeated_history = manager.capture_with_history(root.prefix, collection['history_steps'])
                    if (root.signature != repeat.signature or root.prefix_trace_hash != repeat.prefix_trace_hash
                            or history != repeated_history):
                        raise AssertionError('Repeated history/root differs')
                    root_id = digest([episode, root.signature])
                    inputs = build_actor_inputs(history, geometry, collection['neighbor_capacity'],
                                                collection['history_steps'], collection['tick_s'])
                    branches, futures, candidates = {}, {}, []
                    for index in [*range(5), *reversed(range(5))]:
                        key = str(index)
                        actions = [probes[index]] + [collection['continuation_jerk']] * (collection['horizon_steps']-1)
                        trace, future = branch_with_geometry(manager, root, actions, geometry)
                        repeated = key in branches
                        if repeated:
                            if trace != branches[key] or future != futures[key]:
                                raise AssertionError('Reverse-order repeat changed branch or geometry')
                        else:
                            branches[key], futures[key] = trace, future
                            candidates.append({'candidate_id': index, 'labels': build_branch_labels(
                                root.traffic, trace, actions, collection['tick_s'])})
                        suffix = '_repeat' if repeated else ''
                        write_once(directory / f'c{index}{suffix}.json', {'trace': trace, 'future_metadata': future})
                    pack = {'schema_version': 1, 'episode_id': episode, 'root_id': root_id, 'simulator_seed': seed,
                            'episode_index': 0, 'split': split, 'root_traffic': root.traffic,
                            'prefix_actions': list(root.prefix), 'history_available': False, 'history': None,
                            'training_ready': False, 'candidates': candidates}
                    payload = {'schema_version': 1, 'episode_id': episode, 'root_id': root_id, 'split': split,
                               'history': history, 'inputs': inputs, 'input_hash': fingerprint(inputs),
                               'shared_by_candidates': list(range(5)), 'formal_training_ready': False}
                    if audit is not None:
                        audit('root', {'number': number, 'root': root, 'history': payload, 'pack': pack,
                                       'branches': branches, 'futures': futures})
                    write_once(directory / 'labels.json', pack)
                    payload.update(label_pack=str((directory / 'labels.json').relative_to(repo)),
                                   label_pack_sha256=file_hash(directory / 'labels.json'))
                    write_once(directory / 'history.json', payload)
                    values = [r['labels'] for r in candidates]
                    selected_ids = {a for a in inputs['actor_ids'] if a is not None}
                    valid_counts = [sum(sum(mask) for actor, mask in zip(v['actor_ids'], v['future_mask'])
                                        if actor in selected_ids) for v in values]
                    counts = {reason: sum(v['termination_reason'] == reason for v in values) for reason in
                              (None, 'upstream_collision', 'upstream_arrival', 'upstream_time_limit', 'registered_time_limit')}
                    progress[number].update(status='complete', root_id=root_id, prefix_steps=len(root.prefix),
                        candidate_count=5, branch_executions=10, future_repeat_exact=True,
                        terminal_branches=sum(trace[-1]['done'] for trace in branches.values()),
                        future_observed_frames=sum(len(f['frames']) for f in futures.values()),
                        termination_counts={'horizon_nonterminal' if k is None else k: v for k, v in counts.items()},
                        full_time_horizon_branches=sum(v['full_time_horizon_observed'] for v in values),
                        root_neighbors=len(root.traffic['vehicles'])-1, selected_neighbors=sum(inputs['actor_mask']),
                        omitted_neighbors=len(inputs['root_omitted_ids']), active_summary_groups=sum(inputs['summary_mask']),
                        history_frames=len(history), response=response_magnitude(branches, root.traffic['vehicles']))
                    progress[number].update(valid_selected_actor_time_cells=sum(valid_counts),
                                            empty_trajectory_candidates=sum(n == 0 for n in valid_counts))
                    write_once(directory / 'accounting.json', progress[number])
                    print(f'[collect] split={split} seed={seed} root={number} repeat_exact=true', flush=True)
            finally:
                manager.close()
        if before_seal is not None:
            before_seal()
        return seal_episode(output, binding, {'seed': seed, 'split': split, 'episode_id': episode,
                    'roots': progress, 'status': 'complete', 'elapsed_s': time.perf_counter()-started,
                    'finished_at': now(), 'formal_training_ready': False})
    except BaseException as error:
        write_once(output / 'failure.json', {'job': job, 'binding': binding, 'roots': progress, 'status': 'failed',
                   'failure_reason': f'{type(error).__name__}: {error}', 'finished_at': now()})
        raise
