"""Two epochs on six development roots; validation uses the third development episode."""
from copy import deepcopy
import argparse
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import train_response_predictors as cli
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once
from prediction_rl.prediction.epoch_training import MemberTrainer, subset
from prediction_rl.prediction.supervision import align_supervision
from prediction_rl.prediction.interface import PredictorConfig
from audit_masked_training import equal_tree


def run(args):
    cli.clean(); out=cli.output_dir('p4',args.run_id)
    source=ROOT/'artifacts/p4/p4_training_v1_01/report.json'
    if sha(source)!='174146a8d78962929e18bcd426f9c136db26f0f94deda5bfc449c488459dc3b5':
        raise ValueError('Development source changed')
    evidence=read(source); histories=[]; packs=[]
    for record in evidence['history_inputs']:
        hp=ROOT/record['path']
        if sha(hp)!=record['sha256']: raise ValueError('Development history changed')
        h=read(hp); lp=ROOT/h['label_pack']
        if sha(lp)!=h['label_pack_sha256']: raise ValueError('Development labels changed')
        histories.append(h); packs.append(read(lp))
    manifest=read((ROOT/evidence['label_inputs'][0]['path']).parent/'split_manifest.json')
    cfg=PredictorConfig(); batch=align_supervision(histories,packs,manifest,cfg,split='development')
    if batch.fingerprint()!=evidence['data_hash']: raise ValueError('Development batch drift')
    train,val=subset(batch,list(range(6))),subset(batch,[6,7,8])
    training=deepcopy(read(ROOT/'configs/formal/p04_response_dataset_v1.json')['training'])
    training.update(max_epochs=2,early_stop_patience=2,batch_roots=2)
    report={**cli.metadata(),'status':'running','epoch_training_gate':False,'training_purpose':'development_smoke',
            'source_hashes':cli.source_hashes(),'environment':cli.env(),'members':[],'formal_training_started':False,
            'training_episodes':sorted(set(train.episode_ids)),'validation_episodes':sorted(set(val.episode_ids))}
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for mode in training['methods']:
            for seed in training['member_seeds']:
                member=MemberTrainer(cfg,mode,seed,training,train,val,purpose='development_smoke')
                first=member.run_epoch(); path=out/f'{mode}_{seed}.pt'; binding={'audit':'p4h','mode':mode,'seed':seed}
                member.save(path,binding)
                resumed=MemberTrainer.load(path,cfg,mode,seed,training,train,val,binding,purpose='development_smoke')
                second=member.run_epoch(); repeated=resumed.run_epoch()
                if (second!=repeated or not equal_tree(member.model.state_dict(),resumed.model.state_dict())
                        or not equal_tree(member.optimizer.state_dict(),resumed.optimizer.state_dict())
                        or not equal_tree(member.best_state,resumed.best_state)):
                    raise AssertionError('Stochastic epoch resume is not exact')
                if mode=='ordinary':
                    with torch.inference_mode():
                        y=member.model(val.observations,val.plans)
                        if not torch.equal(y[:,:1].expand_as(y),y): raise AssertionError('Ordinary predictor leaks plan')
                report['members'].append({'mode':mode,'seed':seed,'epoch1':first,'epoch2':second,'resume_exact':True,
                                          'checkpoint_sha256':sha(path)})
                print(f'[epoch_audit] {mode} {seed} two epochs/resume exact',flush=True)
        for seed in training['member_seeds']:
            rows=[r for r in report['members'] if r['seed']==seed]
            if any(rows[0][k]['root_order']!=rows[1][k]['root_order'] for k in ('epoch1','epoch2')):
                raise AssertionError('B2/B3 paired sampling differs')
        report.update(status='complete',epoch_training_gate=True)
    except BaseException as error:
        report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}'); raise
    finally:
        torch.set_num_threads(previous); report['finished_at']=cli.now(); write_once(out/'report.json',report)
    print('epoch_training_audit='+str(out/'report.json')); return out/'report.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run-id',required=True); run(p.parse_args())
