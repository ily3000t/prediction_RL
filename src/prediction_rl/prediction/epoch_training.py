"""CPU member training with root-family sampling and immutable epoch checkpoints."""
from copy import deepcopy
import math
from pathlib import Path
import uuid

import torch

from .checkpoint import CONTRACT
from .losses import LOSS_CONTRACT, trajectory_loss
from .model import ResponsePredictor
from .supervision import SupervisedBatch

VERSION = 'root_family_epoch_cpu_v1'


def subset(batch, indices):
    i = torch.as_tensor(indices,dtype=torch.long)
    return SupervisedBatch({k:v[i] for k,v in batch.observations.items()},batch.plans[i],batch.targets[i],batch.target_mask[i],
        tuple(batch.root_ids[n] for n in indices),tuple(batch.episode_ids[n] for n in indices),batch.split,batch.manifest_hash)


def eligible(batch):
    return batch.target_mask.flatten(1).any(1).nonzero().flatten().tolist()


def initial_member(config, mode, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ResponsePredictor(config,mode)


def selection_update(state, value, epoch, delta):
    if not math.isfinite(value): raise ValueError('Nonfinite validation cannot select a checkpoint')
    improved = state['best_loss'] is None or value < state['best_loss']
    if improved: state.update(best_loss=value,best_epoch=epoch)
    significant = state['patience_anchor'] is None or value < state['patience_anchor']-delta
    if significant: state.update(patience_anchor=value,bad_epochs=0)
    else: state['bad_epochs'] += 1
    return improved


def evaluate(model, batch, batch_size):
    model.eval(); total = 0.; count = 0
    ids = eligible(batch)
    if not ids: raise ValueError('No validation supervision')
    with torch.inference_mode():
        for offset in range(0,len(ids),batch_size):
            b = subset(batch,ids[offset:offset+batch_size])
            loss, stats = trajectory_loss(model(b.observations,b.plans),b.targets,b.target_mask)
            total += float(loss)*stats['supervised_roots']; count += stats['supervised_roots']
    return total/count


def atomic_torch(path, payload):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.parent/(uuid.uuid4().hex[:12]+'.tmp')
    try:
        torch.save(payload,temporary)
        path.hardlink_to(temporary)
    finally: temporary.unlink(missing_ok=True)


class MemberTrainer:
    def __init__(self, config, mode, seed, training, train, validation, *, purpose='formal'):
        if purpose not in ('formal','development_smoke'):
            raise ValueError('Unknown training purpose')
        splits = ('train','validation') if purpose=='formal' else ('development','development')
        if (train.split,validation.split)!=splits or set(train.episode_ids)&set(validation.episode_ids):
            raise ValueError('Disjoint episode-group train/validation required')
        if train.manifest_hash != validation.manifest_hash:
            raise ValueError('Different dataset manifests')
        if type(seed)is not int or not 0<=seed<2**32 or mode not in ('ordinary','conditional'):
            raise ValueError('Invalid training member')
        if (training['device']!='cpu' or training['torch_threads']!=1 or training['optimizer']!='Adam'
                or training['warm_start'] or training['bootstrap'] or training['shuffle']!='paired_seed_epoch_root_permutation'
                or training['loss_contract']!=LOSS_CONTRACT['version']
                or training['channel_scales']!=LOSS_CONTRACT['channel_scales']
                or training['channel_weights']!=LOSS_CONTRACT['channel_weights']):
            raise ValueError('Unsupported CPU training contract')
        if torch.get_num_threads()!=1: raise ValueError('Set exactly one CPU Torch thread')
        self.config,self.mode,self.seed,self.training,self.purpose = config,mode,seed,deepcopy(training),purpose
        self.train,self.validation = train,validation
        self.ids = eligible(train)
        if not self.ids or not eligible(validation): raise ValueError('Empty training/validation supervision')
        self.data_hashes = [train.fingerprint(),validation.fingerprint()]
        self.model = initial_member(config,mode,seed)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=training['learning_rate'],
            betas=tuple(training['betas']),eps=training['epsilon'],weight_decay=training['weight_decay'],foreach=False)
        self.generator = torch.Generator().manual_seed(seed)
        self.cpu_rng = torch.Generator().manual_seed(seed).get_state()
        self.epoch = self.steps = 0; self.history = []
        self.selection = {'best_loss':None,'best_epoch':None,'patience_anchor':None,'bad_epochs':0}
        self.best_state = None

    @property
    def finished(self):
        return self.epoch>=self.training['max_epochs'] or self.selection['bad_epochs']>=self.training['early_stop_patience']

    def run_epoch(self):
        if self.finished: raise ValueError('Frozen epoch/patience budget exhausted')
        order = torch.randperm(len(self.ids),generator=self.generator).tolist()
        ids = [self.ids[i] for i in order]
        total = 0.; count = 0
        self.model.train()
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.cpu_rng)
            for offset in range(0,len(ids),self.training['batch_roots']):
                b = subset(self.train,ids[offset:offset+self.training['batch_roots']])
                self.optimizer.zero_grad(set_to_none=True)
                loss,stats = trajectory_loss(self.model(b.observations,b.plans),b.targets,b.target_mask)
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                    raise ValueError('Missing/nonfinite model gradient')
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.training['clip_grad_norm'],error_if_nonfinite=True)
                self.optimizer.step(); self.steps += 1
                if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                    raise ValueError('Nonfinite model update')
                total += float(loss.detach())*stats['supervised_roots']; count += stats['supervised_roots']
            self.cpu_rng = torch.get_rng_state().clone()
        validation_loss = evaluate(self.model,self.validation,self.training['batch_roots'])
        self.epoch += 1
        improved = selection_update(self.selection,validation_loss,self.epoch,self.training['early_stop_min_delta'])
        if improved: self.best_state = deepcopy(self.model.state_dict())
        row = {'epoch':self.epoch,'optimizer_steps':self.steps,'train_pre_update_loss':total/count,
               'validation_loss':validation_loss,'selected_best':improved,'root_order':ids,
               'excluded_empty_train_roots':len(self.train.root_ids)-len(ids),
               'excluded_empty_validation_roots':len(self.validation.root_ids)-len(eligible(self.validation))}
        self.history.append(row)
        return row

    def save(self,path,binding):
        atomic_torch(path,{'version':VERSION,'purpose':self.purpose,'binding':binding,'config':self.config.to_dict(),
            'mode':self.mode,'seed':self.seed,'training':self.training,'contract':CONTRACT,'loss_contract':LOSS_CONTRACT,
            'data_hashes':self.data_hashes,'epoch':self.epoch,'steps':self.steps,'history':self.history,
            'selection':self.selection,'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
            'best_model':self.best_state,'sampler_rng':self.generator.get_state(),'cpu_rng':self.cpu_rng})

    @classmethod
    def load(cls,path,config,mode,seed,training,train,validation,binding,*,purpose='formal'):
        p = torch.load(path,map_location='cpu',weights_only=True)
        trainer = cls(config,mode,seed,training,train,validation,purpose=purpose)
        expected = {'version':VERSION,'purpose':purpose,'binding':binding,'config':config.to_dict(),
                    'mode':mode,'seed':seed,'training':training,'contract':CONTRACT,'loss_contract':LOSS_CONTRACT,
                    'data_hashes':trainer.data_hashes}
        if set(p)!=set(expected)|{'epoch','steps','history','selection','model','optimizer','best_model','sampler_rng','cpu_rng'}:
            raise ValueError('Unsupported checkpoint envelope')
        if any(p[k]!=v for k,v in expected.items()): raise ValueError('Training checkpoint identity/contract mismatch')
        if type(p['epoch'])is not int or not 1<=p['epoch']<=training['max_epochs'] or len(p['history'])!=p['epoch']:
            raise ValueError('Invalid epoch history')
        batches=math.ceil(len(trainer.ids)/training['batch_roots'])
        if p['steps']!=p['epoch']*batches: raise ValueError('Invalid optimizer step count')
        selection = {'best_loss':None,'best_epoch':None,'patience_anchor':None,'bad_epochs':0}
        gen=torch.Generator().manual_seed(seed)
        for i,row in enumerate(p['history'],1):
            if selection['bad_epochs']>=training['early_stop_patience']:
                raise ValueError('Checkpoint continued after frozen early stop')
            order=[trainer.ids[n] for n in torch.randperm(len(trainer.ids),generator=gen).tolist()]
            if (row['epoch']!=i or row['optimizer_steps']!=i*batches or row['root_order']!=order
                    or not math.isfinite(row['train_pre_update_loss'])): raise ValueError('Corrupt sampler/history')
            best=selection_update(selection,row['validation_loss'],i,training['early_stop_min_delta'])
            if best!=row['selected_best']: raise ValueError('Corrupt checkpoint selection')
        if selection!=p['selection'] or not torch.equal(gen.get_state(),p['sampler_rng']):
            raise ValueError('Sampler/selection state changed')
        template=trainer.model.state_dict()
        for state in (p['model'],p['best_model']):
            if not isinstance(state,dict) or set(state)!=set(template) or any(
                    not isinstance(state[k],torch.Tensor) or state[k].shape!=v.shape or state[k].dtype!=v.dtype
                    or not torch.isfinite(state[k]).all() for k,v in template.items()):
                raise ValueError('Incomplete/nonfinite member weights')
            if any(not torch.equal(state[k],v) for k,v in trainer.model.named_buffers()):
                raise ValueError('Fixed model buffers changed')
        opt=p['optimizer']; expected_opt=trainer.optimizer.state_dict()
        if set(opt)!={'state','param_groups'} or opt['param_groups']!=expected_opt['param_groups']:
            raise ValueError('Optimizer hyperparameters changed')
        ids=expected_opt['param_groups'][0]['params']
        if set(opt['state'])!=set(ids): raise ValueError('Missing optimizer moments')
        for ident,param in zip(ids,trainer.model.parameters()):
            moment=opt['state'][ident]
            if set(moment)!={'step','exp_avg','exp_avg_sq'}: raise ValueError('Wrong Adam moments')
            for k in ('exp_avg','exp_avg_sq'):
                v=moment[k]
                if not isinstance(v,torch.Tensor) or v.shape!=param.shape or v.dtype!=param.dtype or not torch.isfinite(v).all():
                    raise ValueError('Invalid Adam moment tensor')
            if (not isinstance(moment['step'],torch.Tensor) or (moment['exp_avg_sq']<0).any()
                    or moment['step'].shape!=() or moment['step'].item()!=p['steps']):
                raise ValueError('Invalid Adam step/moment')
        trainer.model.load_state_dict(p['model'],strict=True); trainer.optimizer.load_state_dict(opt)
        trainer.generator.set_state(p['sampler_rng']); trainer.cpu_rng=p['cpu_rng']
        check=torch.Generator(); check.set_state(trainer.cpu_rng)
        trainer.epoch,trainer.steps=p['epoch'],p['steps']; trainer.history=p['history']
        trainer.selection,trainer.best_state=p['selection'],p['best_model']
        return trainer
