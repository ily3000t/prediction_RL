"""Bounded P4d label/loss/resume audit, NOT a predictor training CLI."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from audit_environment import sha, write
from audit_predictor_interfaces import load_roots
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.prediction.supervision import align_supervision
from prediction_rl.prediction.losses import LOSS_CONTRACT
from prediction_rl.prediction.development_training import DevelopmentTrainer, TRAINING_CONTRACT


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def load_config(path):
    c=read(path)
    keys={'schema_version','source_interface_report','source_interface_sha256','source_history_report',
          'source_history_sha256','source_label_report','source_label_sha256','member_initialization_seeds',
          'model','torch_threads','optimizer_steps_per_path','training_split'}
    if not isinstance(c,dict) or set(c)!=keys: raise ValueError('Unknown/missing P4d audit fields')
    if (any(type(c[k]) is not int for k in ('schema_version','torch_threads','optimizer_steps_per_path')) or
            c['schema_version']!=1 or c['torch_threads']!=1 or c['optimizer_steps_per_path']!=2 or
            c['training_split']!='development' or c['member_initialization_seeds']!=[3101,3102,3103] or
            any(type(s)is not int for s in c['member_initialization_seeds']) or
            PredictorConfig.from_dict(c['model'])!=PredictorConfig()):
        raise ValueError('Development smoke bounds/configuration are frozen')
    hashes={'interface':'2a668c9ee0a49a114e2a840ebc7bbcecfae982f34491ab26778bf1e27623fda4',
            'history':'89d4f73676a1fa6816421731cc30ee4d6f379d3e6d971879ccfdda20cfbb6c8c',
            'label':'7f4e239d40872262189bf527b366764bd818f69c680e8debe252b595bcbd27c2'}
    for kind,expected in hashes.items():
        p=Path(c[f'source_{kind}_report'])
        if p.is_absolute() or '..' in p.parts or not (ROOT/p).resolve().is_relative_to(ROOT):
            raise ValueError('Use project-relative evidence paths')
        if c[f'source_{kind}_sha256']!=expected or sha(ROOT/p)!=expected:
            raise ValueError('Use unchanged accepted P4a/P4b/P4c evidence')
    return c


def equal_tree(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(v,b[k]) for k,v in a.items())
    if isinstance(a,list): return isinstance(b,list) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def run(args):
    config=load_config(args.config)
    if not re.fullmatch('[A-Za-z0-9_-]{1,40}',args.run_id): raise ValueError('Use short unique run ID')
    git=lambda *a: subprocess.check_output(['git','-C',str(ROOT),*a],text=True,encoding='utf-8').strip()
    if git('status','--porcelain'): raise ValueError('Commit source before acceptance')
    prior=read(ROOT/config['source_interface_report'])
    if prior['status']!='complete' or not prior['interface_gate'] or prior['source_history_sha256']!=config['source_history_sha256']:
        raise ValueError('P4c prerequisite failed')
    _,records=load_roots(config)
    label_report_path=ROOT/config['source_label_report']; label_report=read(label_report_path)
    if label_report['status']!='complete' or not label_report['label_contract_gate']:
        raise ValueError('P4a prerequisite failed')
    manifest_path=label_report_path.parent/'split_manifest.json'
    if sha(manifest_path)!=label_report['output_sha256']['split_manifest.json']:
        raise ValueError('Split manifest changed')
    histories,packs,label_records=[],[],[]
    for r in records:
        history=read(ROOT/r['path']); path=Path(history['label_pack'])
        if path.is_absolute() or '..' in path.parts or (ROOT/path).resolve().parent!=label_report_path.parent.resolve():
            raise ValueError('Label pack escaped pinned dataset')
        path=ROOT/path
        if sha(path)!=history['label_pack_sha256'] or sha(path)!=label_report['output_sha256'][path.name]:
            raise ValueError('Label pack hash mismatch')
        histories.append(history); packs.append(read(path))
        label_records.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    output=ROOT/'artifacts/p4'/args.run_id; output.mkdir(parents=True,exist_ok=False)
    resolved={**config,'loss_contract':LOSS_CONTRACT,'training_contract':TRAINING_CONTRACT}
    write(output/'resolved_config.json',resolved)
    report={'status':'running','training_interface_gate':False,'git_commit':git('rev-parse','HEAD'),
            'working_tree_dirty':False,'command':subprocess.list2cmdline([sys.executable,*sys.argv]),
            'started_at':datetime.now(timezone.utc).isoformat(),'config_sha256':sha(Path(args.config)),
            'resolved_config_sha256':sha(output/'resolved_config.json'),'history_inputs':records,'label_inputs':label_records,
            'source_hashes':{k:v for k,v in config.items() if k.endswith('_sha256')},
            'split_manifest_sha256':sha(manifest_path),'training_status':'development_smoke_only',
            'formal_training_ready':False,'execution_contract':'offline_fixed_batch_cpu_no_simulation',
            'optimizer_steps_per_member_path':2,'replayed_resume_steps_per_member':1,
            'member_initialization_seeds':config['member_initialization_seeds'],
            'environment_versions':{'python':sys.version,'executable':sys.executable,'torch':importlib.metadata.version('torch'),
                                    'numpy':importlib.metadata.version('numpy'),'os':platform.platform(),'device':'cpu'},'models':{}}
    started,previous=time.perf_counter(),torch.get_num_threads()
    try:
        torch.set_num_threads(config['torch_threads'])
        cfg=PredictorConfig.from_dict(config['model'])
        batch=align_supervision(histories,packs,read(manifest_path),cfg,split='development')
        report.update(root_count=len(batch.root_ids),episode_count=len(set(batch.episode_ids)),candidate_count=45,
                      data_hash=batch.fingerprint(),target_shape=list(batch.targets.shape),
                      valid_actor_time_cells=int(batch.target_mask.sum()),
                      mask_capacity_cells=batch.target_mask.numel())
        for mode in ('ordinary','conditional'):
            model=PredictorEnsemble(cfg,mode,config['member_initialization_seeds'])
            initial={k:v.detach().clone() for k,v in model.state_dict().items()}
            training=DevelopmentTrainer(model,batch)
            step1=training.step(batch)
            state1=output/(mode+'_s1.pt')
            provenance={'git_commit':report['git_commit'],'data_hash':report['data_hash'],
                        'resolved_config_sha256':report['resolved_config_sha256']}
            training.save(state1,provenance)
            resumed=DevelopmentTrainer.load(state1,batch,expected_mode=mode,expected_config=cfg)
            step2=training.step(batch); repeated=resumed.step(batch)
            if step2!=repeated or not equal_tree(training.model.state_dict(),resumed.model.state_dict()):
                raise AssertionError('Resumed update differs from uninterrupted update')
            if not all(equal_tree(a.state_dict(),b.state_dict()) for a,b in zip(training.optimizers,resumed.optimizers)):
                raise AssertionError('Adam moments differ after resume')
            changed=[any(not torch.equal(initial[k],v) for k,v in model.state_dict().items() if k.startswith(f'members.{i}.')) for i in range(3)]
            if not all(changed): raise AssertionError('A member did not update')
            model.eval(); resumed.model.eval()
            with torch.inference_mode():
                prediction=model(batch.observations,batch.plans)['member_trajectories']
                replay=resumed.model(batch.observations,batch.plans)['member_trajectories']
                if not torch.equal(prediction,replay): raise AssertionError('Post-update inference differs')
                if mode=='ordinary' and not torch.equal(prediction[:,:,:1].expand_as(prediction),prediction):
                    raise AssertionError('Ordinary output depends on plan after training')
            state2=output/(mode+'_s2.pt'); training.save(state2,provenance)
            report['models'][mode]={'step1':step1,'step2':step2,'resume_weights_exact':True,
                'resume_optimizer_exact':True,'resume_inference_exact':True,'all_members_updated':all(changed),
                'checkpoint_sha256':{p.name:sha(p) for p in (state1,state2)}}
            print(f'[p4d] {mode}: labels/loss and two-step Adam resume passed',flush=True)
        if (any(sha(ROOT/r['path'])!=r['sha256'] for r in records+label_records) or
                sha(manifest_path)!=report['split_manifest_sha256'] or
                any(sha(ROOT/config[f'source_{k}_report'])!=config[f'source_{k}_sha256'] for k in ('label','history','interface'))):
            raise AssertionError('Source evidence changed during audit')
        report.update(status='complete',training_interface_gate=True,source_evidence_unchanged=True,
                      next_stage='P4e_formal_collection_design_and_training_protocol_review')
    except Exception as error:
        report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        torch.set_num_threads(previous)
        report.update(finished_at=datetime.now(timezone.utc).isoformat(),elapsed_s=time.perf_counter()-started)
        write(output/'report.json',report)
    return output/'report.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--run-id',required=True)
    print(run(parser.parse_args()))
