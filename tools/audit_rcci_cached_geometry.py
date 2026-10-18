"""Zero-inference RCCI04a exact cached-output verification and bounded timing."""
import argparse
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import numpy as np

import audit_rcci_feasibility as shared
import smoke_rcci_native as native
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from prediction_rl.features.road_consequence_geometry_cached import CachedRoadConsequenceMap
from prediction_rl.features.road_consequences_v2 import summarize_road_consequences_v2 as summary

SETTINGS = {'version': 'rcci04a_exact_static_geometry_cache_v1',
    'design': 'docs/design/rcci04a_cached_geometry_and_raw_reuse_v1.md',
    'native_request': 'artifacts/rcci03/rcci03_native_v1/request.json',
    'native_request_hash': '17e149d31001ab4ba6aed4123af1bb209b114ae55a6b54b5190f3230b5c7acf6',
    'native_report_sha256': 'f8ac303062a2ccd2a5e9e500dea80529c28e3ce5d8958d8e91daa2af074543f6',
    'native_index_sha256': '19aa5a965569e7870b3658250a9fcf28b00abc5611f7f453e3647503c7382290',
    'full_cached_summaries': 1780, 'native_cached_summaries': 18,
    'timing_warmup_rounds': 1, 'timing_rounds': 3, 'zone_capacity': 128,
    'predictor_forward_calls': 0, 'policy_forward_calls': 0, 'simulation_episodes': 0}


def require(ok, message):
    if not ok: raise ValueError(message)


def verify_inputs(baseline):
    parent, _, _, bound = native.parent_inputs(native.SETTINGS, baseline)
    path = inside(ROOT, SETTINGS['native_request']); q = read(path); directory = path.parent
    report, receipt, index = (read(directory/n) for n in ('report.json', 'receipt.json', 'index.json'))
    current = shared.sources()
    require(digest(q) == SETTINGS['native_request_hash'] and report['request_hash'] == digest(q)
        and report['status'] == 'complete' and report['engineering_gate']
        and sha(directory/'report.json') == SETTINGS['native_report_sha256']
        and sha(directory/'index.json') == SETTINGS['native_index_sha256']
        and receipt == {'request_hash': digest(q), 'report_sha256': SETTINGS['native_report_sha256'],
                       'index_sha256': SETTINGS['native_index_sha256']}
        and q['config'] == native.SETTINGS and q['environment'] == shared.environment()
        and all(current.get(k) == h for k, h in q['source_hashes'].items())
        and Path(q['baseline_root']).resolve() == Path(baseline).resolve(), 'Completed native parent changed')
    require(not (directory/'writer.lock').exists(), 'Native parent is active')
    for name, h in {**q['parent_artifact_hashes'], **bound}.items():
        require(sha(inside(ROOT, name)) == h, 'Historical parent changed: '+name)
    for name, h in index['files'].items():
        p = inside(directory, name)
        require(sha(p) == h, 'Native cached output changed: '+name)
        bound[p.relative_to(ROOT).as_posix()] = h
    for name in ('request.json', 'report.json', 'receipt.json', 'index.json'):
        p = directory/name; bound[p.relative_to(ROOT).as_posix()] = sha(p)
    return parent, bound


def equal_results(actual, expected):
    np.testing.assert_array_equal(actual['features'], np.asarray(expected['features'], dtype=np.float32))
    require({k: v for k, v in actual.items() if k != 'features'} ==
            {k: v for k, v in expected.items() if k != 'features'}, 'Complete summary diagnostics differ')


def cached_native():
    directory = inside(ROOT, SETTINGS['native_request']).parent
    rows = []
    for replay in ('stream', 'mixed'):
        for mode in ('ordinary', 'conditional'):
            for row in read(directory/(replay+'_rcci_'+mode)/'cached_examples.json'):
                rows.append((replay, mode, row))
    require(len(rows) == SETTINGS['native_cached_summaries'], 'Wrong native cache roster')
    return rows


def timing(rows, network):
    roads = {'original': RoadConsequenceMap(network), 'cached': CachedRoadConsequenceMap(network)}
    cases = [(r['history'], np.asarray(r['mean'], dtype=np.float32),
              np.asarray(r['ego'], dtype=np.float64)) for _, _, r in rows]
    for _ in range(SETTINGS['timing_warmup_rounds']):
        for case in cases:
            for road in roads.values(): summary(*case, road)
    samples = {name: [] for name in roads}
    for iteration in range(SETTINGS['timing_rounds']):
        for i, case in enumerate(cases):
            order = ('original', 'cached') if (iteration+i) % 2 == 0 else ('cached', 'original')
            results = {}
            for name in order:
                start = time.perf_counter(); results[name] = summary(*case, roads[name])
                samples[name].append(time.perf_counter()-start)
            equal_results(results['cached'], results['original'])
    metrics = {}
    for name, values in samples.items():
        metrics[name] = {'count': len(values), 'mean': float(np.mean(values)),
            **{'p'+str(p): float(np.percentile(values, p)) for p in (50, 95, 99)}, 'max': max(values)}
    return {'scope': '18_existing_native_caches_postprocessing_only_not_deployment',
        'samples_s': samples, 'metrics_s': metrics,
        'ratio_original_mean_over_cached_mean': metrics['original']['mean']/metrics['cached']['mean'],
        'cached_zone_cache': roads['cached'].cache_info()}


def audit(baseline, run_id):
    shared.clean()
    require(bool(re.fullmatch('[A-Za-z0-9_-]{1,24}', run_id)), 'Use a fresh short run ID')
    baseline = Path(baseline).resolve()
    parent, bound = verify_inputs(baseline)
    q = {**shared.metadata(), 'config': SETTINGS, 'environment': shared.environment(),
         'source_hashes': shared.sources(), 'baseline_root': str(baseline),
         'input_hashes': bound, 'design_sha256': sha(ROOT/SETTINGS['design'])}
    directory = ROOT/'artifacts/rcci04a'/run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_once(directory/'request.json', q)
    with run_lock(directory):
        report = {**shared.metadata(), 'request_hash': digest(q), 'status': 'running',
            'simulation_episodes': 0, 'predictor_forward_calls': 0, 'policy_forward_calls': 0,
            'training_performed': False, 'method_effect_established': False, 'training_released': False}
        write_once(directory/'started.json', report)
        try:
            network = ROOT/'RL-MPC-LaneMerging-master/merge.net.xml'
            road = CachedRoadConsequenceMap(network, zone_capacity=SETTINGS['zone_capacity'])
            cohort = inside(ROOT, native.SETTINGS['parent_request']).parent
            index = read(cohort/'index.json'); verified = []
            require(len(parent['roots']) == 890 and len(index['root_files']) == 890, 'Full roster changed')
            for i, (name, source) in enumerate(zip(index['root_files'], parent['roots'])):
                row = read(cohort/name); history = read(inside(baseline, source['history']))['history']
                require(row['diagnostics']['root_id'] == source['root_id'], 'Root/cache ordering differs')
                ego = np.asarray(row['forecasts']['ego_rollout'], dtype=np.float64)
                for mode in ('ordinary', 'conditional'):
                    neighbors = np.asarray(row['forecasts']['mean_predictions'][mode], dtype=np.float32)
                    actual = summary(history, neighbors, ego, road)
                    old = row['diagnostics']['coverage'][mode]
                    np.testing.assert_array_equal(actual['features'], np.asarray(old['v2_features'], dtype=np.float32))
                    require(actual['candidates'] == old['v2_candidates'], 'Full cached candidate diagnostics differ')
                    verified.append({'root_id': source['root_id'], 'mode': mode,
                        'features_sha256': digest(actual['features'].tolist()),
                        'candidates_sha256': digest(actual['candidates'])})
                if (i+1) % 100 == 0: print(f'[rcci04a] roots={i+1}/890', flush=True)
            rows = cached_native(); original = RoadConsequenceMap(network)
            for replay, mode, r in rows:
                args = (r['history'], np.asarray(r['mean'], dtype=np.float32), np.asarray(r['ego'], dtype=np.float64))
                actual = summary(*args, road); expected = summary(*args, original)
                equal_results(actual, expected)
                np.testing.assert_array_equal(actual['features'], np.asarray(r['features'], dtype=np.float32))
            require(len(verified) == SETTINGS['full_cached_summaries'], 'Wrong full summary count')
            write_once(directory/'equivalence.json', {'full': verified, 'native_exact_complete_summaries': len(rows)})
            print('[rcci04a] exact comparisons complete; bounded timing', flush=True)
            measured = timing(rows, network); write_once(directory/'timing.json', measured)
            # Source and all bound evidence must remain unchanged during recomputation.
            _, after = verify_inputs(baseline)
            require(after == bound and shared.sources() == q['source_hashes'] and
                    shared.git(ROOT, 'rev-parse', 'HEAD') == q['git_commit'], 'Inputs/source changed during audit')
            shared.clean()
            report.update(status='complete', engineering_gate=True, exact_full_summaries=len(verified),
                exact_native_summaries=len(rows), features_exact=True, candidates_exact=True,
                full_zone_cache=road.cache_info(), timing=measured['metrics_s'],
                ratio_original_mean_over_cached_mean=measured['ratio_original_mean_over_cached_mean'],
                files={n: sha(directory/n) for n in ('equivalence.json', 'timing.json')})
        except BaseException:
            report.update(status='failed', engineering_gate=False, failure_reason=traceback.format_exc()); raise
        finally:
            report['finished_at'] = shared.now(); write_once(directory/'report.json', report)
    write_once(directory/'receipt.json', {'request_hash': digest(q), 'report_sha256': sha(directory/'report.json')})
    print('rcci04a_report='+str(directory/'report.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', required=True); p.add_argument('--run-id', required=True)
    args = p.parse_args(); audit(args.baseline_root, args.run_id)
