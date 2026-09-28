"""Review-only plans: no SUMO, model fitting, approval or execution functions."""
import ast
from copy import deepcopy
import json
from pathlib import PurePosixPath, PureWindowsPath
import re

from .dataset_contract import digest, episode_id, validate_split_manifest
from .geometric_roots import validate_targets
from prediction_rl.prediction.interface import PredictorConfig


def keys(value, names):
    if not isinstance(value,dict) or set(value)!=set(names.split()):
        raise ValueError('Missing/unknown protocol fields: '+names)


def relative_path(value):
    if (not isinstance(value,str) or not value or '\\' in value or
            PureWindowsPath(value).drive or PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts):
        raise ValueError('Use portable project-relative forward-slash paths')


def positive_integer(value, maximum):
    if type(value)is not int or not 1<=value<=maximum: raise ValueError('Invalid bounded positive integer')


def validate_protocol(p):
    keys(p,'schema_version protocol_id review_status baseline sources splits collection model training calibration test_policy review_items')
    if type(p['schema_version'])is not int or p['schema_version']!=1 or p['review_status']!='draft_not_approved':
        raise ValueError('This planner supports draft review only, never self-approval')
    if not re.fullmatch('[a-z][a-z0-9_]{1,63}',p['protocol_id']): raise ValueError('Invalid short protocol ID')
    b=p['baseline']; keys(b,'upstream_config reference_checkpoint reference_sha256 episode_group_namespace_sha256 execution_contract')
    for k in ('upstream_config','reference_checkpoint'): relative_path(b[k])
    for k in ('reference_sha256','episode_group_namespace_sha256'):
        if not isinstance(b[k],str) or not re.fullmatch('[a-f0-9]{64}',b[k]): raise ValueError('Pin baseline hashes')
    if b['execution_contract']!='simulation_blocking_exact_v1': raise ValueError('Method-effect must block simulation')
    s=p['sources']; keys(s,'mechanism_report mechanism_sha256 training_interface_report training_interface_sha256')
    for k in ('mechanism_report','training_interface_report'): relative_path(s[k])
    for k in ('mechanism_sha256','training_interface_sha256'):
        if not isinstance(s[k],str) or not re.fullmatch('[a-f0-9]{64}',s[k]): raise ValueError('Pin prerequisite reports')
    sp=p['splits']; keys(sp,'development_seeds train validation calibration test')
    if sp['development_seeds']!=[0,1,100] or any(type(i)is not int for i in sp['development_seeds']):
        raise ValueError('Never promote inspected development seeds')
    seen=set(sp['development_seeds'])
    for split in ('train','validation','calibration','test'):
        r=sp[split]; keys(r,'start count'); positive_integer(r['count'],2048)
        if type(r['start'])is not int or not 0<=r['start']<2**32 or r['start']+r['count']>2**32:
            raise ValueError('Invalid simulator seed range')
        values=set(range(r['start'],r['start']+r['count']))
        if seen & values: raise ValueError('Simulator seeds overlap across splits/development')
        seen |= values
    c=p['collection']; keys(c,'episode_index backend reference_policy root_targets discovery_max_steps branch_repeats worker_processes unreachable_root engineering_error history_steps neighbor_capacity horizon_steps tick_s probe_jerks continuation_jerk future_metadata')
    required={'episode_index':0,'backend':'fresh_sumo_prefix_replay_v1','reference_policy':'author_ddpg_greedy_timefeature_v1',
              'branch_repeats':2,'worker_processes':1,'unreachable_root':'record_without_replacement',
              'engineering_error':'stop_preserve_failure','history_steps':11,'neighbor_capacity':12,'horizon_steps':25,
              'tick_s':.2,'probe_jerks':[-5,-2.5,0,2.5,5],'continuation_jerk':0.,'future_metadata':'lane_position_length_width_v1'}
    if any(c[k]!=v for k,v in required.items()): raise ValueError('Unsupported collection contract')
    for k in ('episode_index','branch_repeats','worker_processes','history_steps','neighbor_capacity','horizon_steps'):
        if type(c[k])is not int: raise ValueError('Integer collection fields cannot be Boolean')
    validate_targets(c['root_targets']); positive_integer(c['discovery_max_steps'],250)
    if PredictorConfig.from_dict(p['model'])!=PredictorConfig(): raise ValueError('Keep the accepted P4c architecture')
    if any(type(x)not in (int,float) for x in [c['tick_s'],c['continuation_jerk'],*c['probe_jerks']]):
        raise ValueError('Control numbers cannot be Boolean')
    t=p['training']; keys(t,'methods member_seeds device torch_threads max_epochs batch_roots shuffle optimizer learning_rate betas epsilon weight_decay clip_grad_norm loss_contract channel_scales channel_weights selection_metric selection_tie early_stop_patience early_stop_min_delta validation_every_epochs warm_start bootstrap')
    required={'methods':['ordinary','conditional'],'device':'cpu','shuffle':'paired_seed_epoch_root_permutation',
              'optimizer':'Adam','betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,'loss_contract':'root_candidate_masked_mse_v1',
              'channel_scales':[10.,3.,5.,2.],'channel_weights':[1.,1.,1.,1.],
              'selection_metric':'validation_root_equal_scaled_mse','selection_tie':'earliest_epoch','validation_every_epochs':1}
    if any(t[k]!=v for k,v in required.items()) or t['warm_start']is not False or t['bootstrap']is not False:
        raise ValueError('Unsupported shared B2/B3 training protocol')
    if type(t['validation_every_epochs'])is not int or any(type(x)not in (int,float) for x in [t['weight_decay'],*t['channel_weights'],*t['channel_scales']]):
        raise ValueError('Training numbers cannot be Boolean')
    seeds=t['member_seeds']
    if (not isinstance(seeds,list) or len(seeds)!=3 or any(type(x)is not int or not 0<=x<2**32 for x in seeds)
            or len(set(seeds))!=3): raise ValueError('Require three distinct initialization seeds')
    for k,maxval in [('torch_threads',32),('max_epochs',1000),('batch_roots',256),('early_stop_patience',1000)]:
        positive_integer(t[k],maxval)
    if t['early_stop_patience']>t['max_epochs']: raise ValueError('Patience exceeds epoch budget')
    for k,upper in [('learning_rate',.1),('clip_grad_norm',100.),('early_stop_min_delta',1.)]:
        if type(t[k])not in (int,float) or not 0<t[k]<=upper: raise ValueError('Invalid finite training number')
    cal=p['calibration']; keys(cal,'method nominal_coverage disagreement_floor_scale scope')
    if cal!={'method':'episode_max_standardized_residual_v1','nominal_coverage':.9,'disagreement_floor_scale':.1,'scope':'observed_nonterminal_cells_only'}:
        raise ValueError('Unsupported draft calibration definition')
    test=p['test_policy']; keys(test,'collection_release selection_use reuse_for_ddpg')
    if test!={'collection_release':'after_predictors_and_calibration_frozen','selection_use':'never','reuse_for_ddpg':'forbidden'}:
        raise ValueError('Do not unlock test data or use it for selection')
    expected=['episode_budget_and_seed_ranges','three_root_reference_policy_coverage','training_budget_and_checkpoint_selection',
              'calibration_scope_and_censoring','P5_shared_ego_geometry_and_ranking_definition']
    if p['review_items']!=expected: raise ValueError('Keep unresolved review decisions visible')


def build_plan(protocol):
    validate_protocol(protocol)
    p=deepcopy(protocol); group=p['baseline']['episode_group_namespace_sha256']
    manifest={'development':[episode_id(group,s,0) for s in p['splits']['development_seeds']]}
    jobs=[]; budget={}; roots=len(p['collection']['root_targets']); candidates=len(p['collection']['probe_jerks'])
    for split in ('train','validation','calibration','test'):
        spec=p['splits'][split]; manifest[split]=[]
        for seed in range(spec['start'],spec['start']+spec['count']):
            ep=episode_id(group,seed,0); manifest[split].append(ep)
            jobs.append({'job_id':f'e{len(jobs):04d}','split':split,'simulator_seed':seed,'episode_index':0,'episode_id':ep,
                         'root_target_ids':[r['id'] for r in p['collection']['root_targets']],
                         'release_stage':'after_frozen_predictors_and_calibration' if split=='test' else 'after_user_review_and_collector_acceptance',
                         'execution_authorized':False})
        n=spec['count']; families=n*roots*candidates
        budget[split]={'episodes':n,'max_roots':n*roots,'max_unique_branches':families,
                       'max_branch_executions_including_repeats':families*p['collection']['branch_repeats']}
    validate_split_manifest(manifest)
    # Production mini-batches must keep candidate families intact. This is an upper
    # bound (all roots reached, no early stop), not a promise of equal attained epochs.
    batches=(budget['train']['max_roots']+p['training']['batch_roots']-1)//p['training']['batch_roots']
    training_jobs=[{'method':m,'member_seed':s,'execution_authorized':False,
                    'max_optimizer_steps':batches*p['training']['max_epochs']}
                   for m in p['training']['methods'] for s in p['training']['member_seeds']]
    return {'schema_version':1,'protocol_id':p['protocol_id'],'protocol_hash':digest(p),'status':'review_only',
            'execution_authorized':False,'formal_training_ready':False,'split_manifest':manifest,
            'manifest_hash':digest(manifest),'collection_jobs':jobs,'training_jobs':training_jobs,'budget':budget,
            'blockers':['human_protocol_review_pending','production_collector_not_accepted','production_trainer_not_implemented',
                        'calibration_not_implemented','P5_ego_geometry_ranking_definition_pending'],
            'resolved_protocol':p}


def resolve_upstream_settings(text, overrides, *, cuda_available, system):
    """Read literal Settings defaults without executing/importing upstream code."""
    if type(cuda_available)is not bool or system not in ('Windows','Linux'): raise ValueError('Explicit host context required')
    tree=ast.parse(text); classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Settings']
    if len(classes)!=1: raise ValueError('Expected one Settings class')
    defaults={}
    for node in classes[0].body:
        if not isinstance(node,ast.Assign): continue
        if len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name): raise ValueError('Unsupported default assignment')
        name=node.targets[0].id
        try: value=ast.literal_eval(node.value)
        except (ValueError,TypeError):
            expr=ast.unparse(node.value)
            if name=='LOG_LEVEL' and expr=='logging.INFO': value=20
            elif name=='CUDA' and expr=='torch.cuda.is_available()': value=cuda_available
            else: raise ValueError('Unsupported nonliteral upstream default: '+name)
        defaults[name]=value
    if not isinstance(overrides,dict) or set(overrides)-set(defaults): raise ValueError('Unknown upstream overrides')
    defaults.update(overrides); defaults['SYSTEM']=system
    return json.loads(json.dumps(defaults,allow_nan=False))
