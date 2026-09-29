"""Validate immutable completed collection without executing its historical code."""
import re
import subprocess

from .branching import fingerprint
from .collection_store import file_hash, read_json, verify_episode
from .collection_protocol import collection_plan, summarize
from .dataset_contract import digest
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.supervision import align_supervision


def inside(repo, name):
    path = (repo / str(name).replace('\\', '/')).resolve()
    if not path.is_relative_to(repo.resolve()):
        raise ValueError('Dataset path escaped project')
    return path


def verify_historical_inputs(repo, hashes, commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Need the actual clean collection commit')
    import json
    for name, expected in hashes.items():
        if name.startswith(('python_lf:', 'json_canonical:')):
            kind, relative = name.split(':', 1)
            path = inside(repo, relative).relative_to(repo).as_posix()
            # Read Git blobs only. Never import/execute historical scripts.
            raw = subprocess.check_output(['git', '-C', str(repo), 'show', commit+':'+path])
            text = raw.decode('utf-8').replace('\r\n', '\n')
            actual = fingerprint(text) if kind == 'python_lf' else digest(json.loads(text))
        else:
            actual = file_hash(inside(repo, name))
        if actual != expected:
            raise ValueError('Historical collection input mismatch: '+name)


def review_dataset(repo, request_path, report_path):
    repo = repo.resolve(); request_path = inside(repo, request_path); report_path = inside(repo, report_path)
    request, report = read_json(request_path), read_json(report_path)
    plan = collection_plan(request['protocol'])
    if (request['plan'] != plan or report['request_hash'] != digest(request)
            or report['status'] != 'complete' or report['working_tree_dirty'] is not False
            or report['test_locked'] is not True or set(report['splits']) != {'train','validation','calibration'}):
        raise ValueError('Require complete clean three-split collection, with locked test')
    output = request_path.parent
    if not report_path.is_relative_to(output/'invocations'):
        raise ValueError('Collection report belongs to another run')
    if read_json(output/'preparation.json')['request_hash'] != digest(request):
        raise ValueError('Prepared request changed')
    jobs = [j for j in plan['jobs'] if j['split'] != 'test']
    if report['completed_jobs'] != [j['job_id'] for j in jobs]:
        raise ValueError('Missing/duplicate/reordered completed episode jobs')
    if any((output/j['job_id']).exists() for j in plan['jobs'] if j['split']=='test'):
        raise ValueError('Test trajectories must remain unopened')
    verify_historical_inputs(repo, request['input_hashes'], report['git_commit'])
    targets = [t['id'] for t in request['protocol']['collection']['root_targets']]
    cfg = PredictorConfig.from_dict(request['protocol']['model'])
    records = {s:[] for s in ('train','validation','calibration')}
    hashes = {str(p.relative_to(repo)):file_hash(p) for p in (request_path, report_path, output/'preparation.json')}
    summaries = {}; metadata = []
    for number, job in enumerate(jobs):
        directory = output/job['job_id']
        receipt = verify_episode(directory, {'request_hash':digest(request),'job':job})
        hashes[str((directory/'complete.json').relative_to(repo))] = file_hash(directory/'complete.json')
        metadata.append((job,receipt['metadata']))
        for i, root in enumerate(receipt['metadata']['roots']):
            if root['status'] != 'complete': continue
            hp, lp = directory/f'r{i}/history.json', directory/f'r{i}/labels.json'
            h, p = read_json(hp), read_json(lp)
            if h['label_pack_sha256'] != file_hash(lp) or inside(repo,h['label_pack']) != lp:
                raise ValueError('History/label join changed')
            batch = align_supervision([h],[p],plan['split_manifest'],cfg,split=job['split'])
            cells = int(batch.target_mask.sum())
            if cells != root['valid_selected_actor_time_cells']:
                raise ValueError('Root supervision accounting changed')
            records[job['split']].append({'episode_id':job['episode_id'],'root_id':root['root_id'],
                'history':str(hp.relative_to(repo)),'labels':str(lp.relative_to(repo)),
                'history_sha256':file_hash(hp),'labels_sha256':file_hash(lp),'valid_cells':cells,
                'eligible_trajectory_root':cells>0})
        if (number+1)%64==0: print(f'[dataset_review] verified={number+1}/{len(jobs)}',flush=True)
    for split in records:
        pairs = [(j,m) for j,m in metadata if j['split']==split]
        summary = summarize(pairs,targets)
        if summary != report['summaries'][split]: raise ValueError('Collection summary differs from receipts')
        reached = [r for _,m in pairs for r in m['roots'] if r['status']=='complete']
        summary.update(eligible_roots=sum(r['eligible_trajectory_root'] for r in records[split]),
            excluded_empty_roots=[r['root_id'] for r in records[split] if not r['eligible_trajectory_root']],
            response_roots_at_development_threshold=sum((r['response']['max_neighbor_position_delta_m'] or 0)>=.1
                or (r['response']['max_neighbor_speed_delta_mps'] or 0)>=.05 for r in reached))
        summaries[split] = summary
    if not all(summaries[s]['eligible_roots'] for s in ('train','validation')):
        raise ValueError('No supervised train/validation roots; keep failures for design review')
    return {'schema_version':1,'status':'complete','collection_commit':report['git_commit'],
            'collection_request_hash':digest(request),'protocol':request['protocol'],
            'manifest':plan['split_manifest'],'input_hashes':hashes,'roots':records,'summaries':summaries,
            'eligible_for_initial_training_trial':True,'prediction_effect_established':False,'test_locked':True,
            'response_thresholds_descriptive_only':{'position_m':.1,'speed_mps':.05}}


def load_split(repo, review, split):
    if split not in ('train','validation'):
        raise ValueError('Training loader cannot read calibration or test targets')
    histories, packs = [], []
    for r in review['roots'][split]:
        hp, lp = inside(repo,r['history']), inside(repo,r['labels'])
        if file_hash(hp)!=r['history_sha256'] or file_hash(lp)!=r['labels_sha256']:
            raise ValueError('Frozen training data changed')
        histories.append(read_json(hp)); packs.append(read_json(lp))
    return align_supervision(histories,packs,review['manifest'],
                             PredictorConfig.from_dict(review['protocol']['model']),split=split)
