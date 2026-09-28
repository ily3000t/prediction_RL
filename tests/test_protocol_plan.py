from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prediction_rl.data.protocol_plan import build_plan,resolve_upstream_settings
from prediction_rl.data.dataset_contract import validate_split_manifest


def protocol():
    return json.loads((ROOT/'configs/formal/p04_response_dataset_v1_draft.json').read_text(encoding='utf-8'))


def test_draft_budget_and_no_execution():
    p=build_plan(protocol())
    assert len(p['collection_jobs'])==512 and len(p['training_jobs'])==6
    assert sum(x['max_roots'] for x in p['budget'].values())==1536
    assert sum(x['max_unique_branches'] for x in p['budget'].values())==7680
    assert sum(x['max_branch_executions_including_repeats'] for x in p['budget'].values())==15360
    assert all(j['max_optimizer_steps']==4800 for j in p['training_jobs'])
    assert not p['execution_authorized'] and not p['formal_training_ready']
    assert all(not j['execution_authorized'] for j in p['collection_jobs']+p['training_jobs'])
    assert all(j['release_stage']=='after_frozen_predictors_and_calibration' for j in p['collection_jobs'] if j['split']=='test')
    assert len(validate_split_manifest(p['split_manifest']))==515


def test_deterministic_nonmutating_and_episode_identity_stable():
    config=protocol(); saved=deepcopy(config)
    a=build_plan(config); b=build_plan(config)
    assert a==b and config==saved
    config['protocol_id']='renamed_draft';config['training']['max_epochs']=80
    c=build_plan(config)
    assert a['split_manifest']==c['split_manifest'] and a['protocol_hash']!=c['protocol_hash']
    assert len({j['job_id'] for j in a['collection_jobs']})==512
    assert max(len(j['job_id']) for j in a['collection_jobs'])==5


@pytest.mark.parametrize('case',['approved','extra','nested_extra','overlap','development','bool_count','negative',
                                'absolute','windows','parent','namespace','test_unlock','reward','architecture','warm_start',
                                'drop_candidate','repeats','auto_replace','patience','nan_lr','bool_weight','unknown_calibration','hide_review'])
def test_unsafe_or_ambiguous_drafts_rejected(case):
    p=protocol()
    if case=='approved': p['review_status']='approved'
    elif case=='extra': p['extra']=1
    elif case=='nested_extra': p['training']['extra']=1
    elif case=='overlap': p['splits']['test']['start']=11010
    elif case=='development': p['splits']['train']['start']=0
    elif case=='bool_count': p['splits']['train']['count']=True
    elif case=='negative': p['splits']['validation']['count']=-1
    elif case=='absolute': p['baseline']['upstream_config']='/tmp/config.json'
    elif case=='windows': p['baseline']['upstream_config']='E:/absolute.json'
    elif case=='parent': p['baseline']['upstream_config']='../config.json'
    elif case=='namespace': p['baseline']['episode_group_namespace_sha256']='bad'
    elif case=='test_unlock': p['test_policy']['collection_release']='now'
    elif case=='reward': p['baseline']['reward_override']=0
    elif case=='architecture': p['model']['hidden_dim']=128
    elif case=='warm_start': p['training']['warm_start']=True
    elif case=='drop_candidate': p['collection']['probe_jerks'].pop()
    elif case=='repeats': p['collection']['branch_repeats']=1
    elif case=='auto_replace': p['collection']['unreachable_root']='replace_seed'
    elif case=='patience': p['training']['early_stop_patience']=101
    elif case=='nan_lr': p['training']['learning_rate']=float('nan')
    elif case=='bool_weight': p['training']['channel_weights'][0]=True
    elif case=='unknown_calibration': p['calibration']['method']='ensemble_std_is_probability'
    elif case=='hide_review': p['review_items']=[]
    with pytest.raises(ValueError): build_plan(p)


def test_parse_upstream_settings_without_executing_module():
    text="raise RuntimeError('must not execute')\nclass Settings:\n    X = 3\n    LOG_LEVEL = logging.INFO\n    CUDA = torch.cuda.is_available()\n    SYSTEM = 'Linux'\n"
    assert resolve_upstream_settings(text,{'X':4},cuda_available=False,system='Windows')=={
        'X':4,'LOG_LEVEL':20,'CUDA':False,'SYSTEM':'Windows'}
    with pytest.raises(ValueError): resolve_upstream_settings(text,{'other':4},cuda_available=False,system='Windows')
    malicious='class Settings:\n    X=__import__("os").system("echo do-not-run")'
    with pytest.raises(ValueError): resolve_upstream_settings(malicious,{},cuda_available=False,system='Windows')


def test_actual_upstream_defaults_preserve_reward_action():
    p=protocol()
    defaults=(ROOT/'RL-MPC-LaneMerging-master/config.py').read_text(encoding='utf-8')
    overrides=json.loads((ROOT/p['baseline']['upstream_config']).read_text(encoding='utf-8'))
    settings=resolve_upstream_settings(defaults,overrides,cuda_available=False,system='Windows')
    assert settings['REWARD_FUNCTION']=='Slotted Jerk'
    assert [settings[k] for k in ('CRASH_REWARD','SUCCESS_REWARD','TIME_REWARD','ALT_J_WEIGHT')]==[-10,10,-.1,.1]
    assert settings['MINIMUM_NEGATIVE_JERK']==-5 and settings['MAXIMUM_POSITIVE_JERK']==5
    assert settings['TICK_LENGTH']==.2 and settings['GYM_ENVIRONMENT']=='sumo-jerk-continuous-v0'
    assert settings['SEED']==0  # Future worker replaces ONLY with its assigned seed and records it.
