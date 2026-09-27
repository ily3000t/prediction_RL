"""Two-step CPU engineering smoke. Deliberately NOT a formal training command."""
import copy
import json
from pathlib import Path
import re
import uuid

import torch

from .checkpoint import CONTRACT
from .interface import PredictorConfig
from .losses import LOSS_CONTRACT, trajectory_loss
from .model import PredictorEnsemble

TRAINING_CONTRACT = {'version': 'p4d_two_step_cpu_smoke_v1', 'optimizer': 'Adam', 'lr': .001,
                     'betas': [.9,.999], 'eps': 1e-8, 'weight_decay': 0., 'clip_grad_norm': 1.,
                     'max_steps': 2, 'split': 'development', 'stochastic_sampling': False}


class DevelopmentTrainer:
    def __init__(self, model, batch):
        if not isinstance(model,PredictorEnsemble) or batch.split != 'development':
            raise ValueError('This bounded trainer only accepts development data')
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('CPU float32 smoke only')
        self.model, self.data_hash, self.step_count = model, batch.fingerprint(), 0
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=TRAINING_CONTRACT['lr'],
            betas=tuple(TRAINING_CONTRACT['betas']),eps=TRAINING_CONTRACT['eps'],weight_decay=0., foreach=False)
            for m in model.members]

    def step(self, batch):
        if self.step_count >= TRAINING_CONTRACT['max_steps']:
            raise ValueError('Two-step engineering budget exhausted; formal training is not authorized')
        if batch.split != 'development' or batch.fingerprint() != self.data_hash:
            raise ValueError('Data/split changed since initialization or resume')
        self.model.train()
        losses, counts = [], None
        # Prevalidate all members before any update; do not train the ensemble mean.
        for optimizer in self.optimizers: optimizer.zero_grad(set_to_none=True)
        for member in self.model.members:
            loss, counts = trajectory_loss(member(batch.observations,batch.plans),batch.targets,batch.target_mask)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in member.parameters()):
                raise ValueError('Missing/nonfinite member gradient')
            torch.nn.utils.clip_grad_norm_(member.parameters(),TRAINING_CONTRACT['clip_grad_norm'],error_if_nonfinite=True)
            losses.append(float(loss.detach()))
        for optimizer in self.optimizers: optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError('Nonfinite model after update; run is invalid')
        self.step_count += 1
        return {'step':self.step_count,'member_losses':losses,'supervision':counts}

    def save(self,path,provenance):
        path=Path(path)
        if path.exists(): raise FileExistsError('Never overwrite development training evidence')
        if not isinstance(provenance,dict) or not re.fullmatch('[0-9a-f]{40}',str(provenance.get('git_commit',''))):
            raise ValueError('Record the source commit')
        if any(not torch.isfinite(v).all() for v in self.model.state_dict().values()):
            raise ValueError('Cannot save nonfinite model state')
        metadata=json.loads(json.dumps(provenance,allow_nan=False))
        payload={'version':'p4d_training_state_v1','training_status':'development_smoke_only',
                 'model_contract':CONTRACT,'loss_contract':LOSS_CONTRACT,'training_contract':TRAINING_CONTRACT,
                 'model_config':self.model.config.to_dict(),'mode':self.model.mode,
                 'initialization_seeds':list(self.model.initialization_seeds),'data_hash':self.data_hash,
                 'step_count':self.step_count,'model_state':copy.deepcopy(self.model.state_dict()),
                 'optimizer_states':[copy.deepcopy(o.state_dict()) for o in self.optimizers],
                 'provenance':metadata}
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.parent/(uuid.uuid4().hex[:12]+'.tmp')
        try:
            torch.save(payload,temporary)
            path.hardlink_to(temporary)
        finally: temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls,path,batch,*,expected_mode,expected_config):
        p=torch.load(Path(path),map_location='cpu',weights_only=True)
        keys={'version','training_status','model_contract','loss_contract','training_contract','model_config',
              'mode','initialization_seeds','data_hash','step_count','model_state','optimizer_states','provenance'}
        if (not isinstance(p,dict) or set(p)!=keys or p['version']!='p4d_training_state_v1' or
                p['training_status']!='development_smoke_only' or p['model_contract']!=CONTRACT or
                p['loss_contract']!=LOSS_CONTRACT or p['training_contract']!=TRAINING_CONTRACT or
                p['mode']!=expected_mode or PredictorConfig.from_dict(p['model_config'])!=expected_config):
            raise ValueError('Training checkpoint contract mismatch; no partial migration')
        if type(p['step_count']) is not int or not 0 <= p['step_count'] <= 2 or p['data_hash']!=batch.fingerprint():
            raise ValueError('Changed data or invalid optimizer step count')
        if not isinstance(p['provenance'],dict) or not re.fullmatch('[0-9a-f]{40}',str(p['provenance'].get('git_commit',''))):
            raise ValueError('Missing source provenance')
        model=PredictorEnsemble(expected_config,expected_mode,p['initialization_seeds'])
        expected=model.state_dict(); state=p['model_state']
        if (not isinstance(state,dict) or set(state)!=set(expected) or any(
                not isinstance(state[k],torch.Tensor) or state[k].dtype!=v.dtype or state[k].shape!=v.shape or
                not torch.isfinite(state[k]).all() for k,v in expected.items())):
            raise ValueError('Incomplete/nonfinite model state')
        if any(not torch.equal(state[k],v) for k,v in model.named_buffers()):
            raise ValueError('Changed fixed model buffers')
        model.load_state_dict(state,strict=True)
        trainer=cls(model,batch)
        if not isinstance(p['optimizer_states'],list) or len(p['optimizer_states'])!=3:
            raise ValueError('Require all three independent optimizer states')
        for member,optimizer,loaded in zip(model.members,trainer.optimizers,p['optimizer_states']):
            template=optimizer.state_dict()
            if (not isinstance(loaded,dict) or set(loaded)!={'state','param_groups'} or
                    loaded['param_groups']!=template['param_groups'] or not isinstance(loaded['state'],dict)):
                raise ValueError('Optimizer groups/hyperparameters changed')
            params=list(member.parameters())
            ids=template['param_groups'][0]['params']
            if set(loaded['state']) != (set(ids) if p['step_count'] else set()):
                raise ValueError('Missing optimizer moments')
            for index,param in zip(ids,params):
                if not p['step_count']: continue
                moment=loaded['state'][index]
                if not isinstance(moment,dict) or set(moment)!={'step','exp_avg','exp_avg_sq'}:
                    raise ValueError('Invalid Adam moment keys')
                for key in ('exp_avg','exp_avg_sq'):
                    value=moment[key]
                    if (not isinstance(value,torch.Tensor) or value.dtype!=param.dtype or value.shape!=param.shape or
                            not torch.isfinite(value).all() or (key=='exp_avg_sq' and (value<0).any())):
                        raise ValueError('Invalid Adam moments')
                step=moment['step']
                if not isinstance(step,torch.Tensor) or step.shape!=() or step.dtype!=torch.float32 or step.item()!=p['step_count']:
                    raise ValueError('Optimizer/model step mismatch')
            optimizer.load_state_dict(loaded)
        trainer.step_count=p['step_count']
        return trainer
