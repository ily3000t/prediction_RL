"""Generate a review-only protocol/manifest. There is no run/approve command."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from audit_environment import sha,write
from diagnose_branches import environment_metadata
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.protocol_plan import build_plan,resolve_upstream_settings


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def project_path(value):
    path=ROOT/Path(value.replace('\\','/'))
    if not path.resolve().is_relative_to(ROOT): raise ValueError('Evidence path escaped project')
    return path


def prepare(args):
    protocol=read(Path(args.config)); plan=build_plan(protocol)
    if not re.fullmatch('[A-Za-z0-9_-]{1,40}',args.run_id): raise ValueError('Use a short unique run ID')
    git=lambda *a:subprocess.check_output(['git','-C',str(ROOT),*a],text=True,encoding='utf-8').strip()
    if git('status','--porcelain'): raise ValueError('Commit the draft and planner before generating review evidence')
    sources=protocol['sources']; verified={}
    for kind in ('mechanism','training_interface'):
        path=project_path(sources[kind+'_report'])
        if sha(path)!=sources[kind+'_sha256']: raise ValueError('Prerequisite evidence hash changed')
        verified[str(path.relative_to(ROOT))]=sha(path)
    mechanism=read(project_path(sources['mechanism_report']))
    training=read(project_path(sources['training_interface_report']))
    if mechanism['status']!='complete' or not mechanism['engineering_gate']:
        raise ValueError('P3 mechanism prerequisite not accepted')
    if training['status']!='complete' or not training['training_interface_gate'] or training['formal_training_ready']:
        raise ValueError('P4d prerequisite not accepted as development-only')
    upstream_hashes={k:v for k,v in mechanism['source_sha256'].items() if k.startswith('RL-MPC-LaneMerging-master')}
    if digest(upstream_hashes)!=protocol['baseline']['episode_group_namespace_sha256']:
        raise ValueError('Do not rename the episode namespace to evade inspected seeds')
    for name,expected in upstream_hashes.items():
        if sha(project_path(name))!=expected: raise ValueError('Frozen upstream code/config/network changed: '+name)
        verified[name]=expected
    baseline=protocol['baseline']
    if (baseline['upstream_config']!=mechanism['plan']['upstream_config'] or
            baseline['reference_checkpoint']!=mechanism['plan']['reference_policy']['checkpoint'] or
            baseline['reference_sha256']!=mechanism['policy_checkpoint_hash'] or
            protocol['collection']['root_targets']!=mechanism['plan']['root_targets']):
        raise ValueError('Draft differs from accepted reference environment/root definitions')
    policy=project_path(baseline['reference_checkpoint'])
    if sha(policy)!=baseline['reference_sha256']: raise ValueError('Author reference checkpoint changed')
    verified[str(policy.relative_to(ROOT))]=sha(policy)  # Read bytes only; never load/execute the policy.
    label_dir=project_path(training['label_inputs'][0]['path']).parent
    previous_manifest=label_dir/'split_manifest.json'
    if sha(previous_manifest)!=training['split_manifest_sha256']:
        raise ValueError('Inspected development manifest changed')
    if read(previous_manifest)['development']!=plan['split_manifest']['development']:
        raise ValueError('Existing episode identities drifted')
    verified[str(previous_manifest.relative_to(ROOT))]=sha(previous_manifest)
    settings_path=ROOT/'RL-MPC-LaneMerging-master/config.py'
    resolved=resolve_upstream_settings(settings_path.read_text(encoding='utf-8'),
        read(project_path(baseline['upstream_config'])),cuda_available=bool(torch.cuda.is_available()),
        system='Windows' if os.name=='nt' else 'Linux')
    output=ROOT/'artifacts/p4'/args.run_id;output.mkdir(parents=True,exist_ok=False)
    report={'status':'preparing','git_commit':git('rev-parse','HEAD'),'working_tree_dirty':False,
            'started_at':datetime.now(timezone.utc).isoformat(),'command':subprocess.list2cmdline([sys.executable,*sys.argv]),
            'config_sha256':sha(Path(args.config)),'protocol_hash':plan['protocol_hash'],
            'plan_validation_pass':False,'execution_authorized':False,'formal_training_ready':False,
            'input_hashes':verified,'output_hashes':{},'blockers':plan['blockers']}
    try:
        report['environment_versions']=environment_metadata()
        products={'resolved_protocol.json':protocol,'plan.json':plan,'split_manifest.json':plan['split_manifest'],
                  'resolved_upstream_base_settings.json':resolved}
        for name,payload in products.items():
            write(output/name,payload);report['output_hashes'][name]=sha(output/name)
        if any(sha(project_path(name))!=value for name,value in verified.items()):
            raise AssertionError('Input evidence changed during planning')
        report.update(status='review_only',plan_validation_pass=True,collection_jobs=len(plan['collection_jobs']),
                      training_jobs=len(plan['training_jobs']),budget=plan['budget'],source_evidence_unchanged=True)
    except Exception as error:
        report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['finished_at']=datetime.now(timezone.utc).isoformat();write(output/'planning_report.json',report)
    return output/'planning_report.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--run-id',required=True)
    print(prepare(parser.parse_args()))
