"""Same-root ID alignment; targets stay outside the observation dictionary."""
from dataclasses import dataclass
import hashlib
import json

import torch

from prediction_rl.data.dataset_contract import digest, validate_rows, TERMINATIONS
from .interface import collate_inputs, probe_plans

LABEL_FEATURES = ['root_ego_relative_x_m', 'root_ego_relative_y_m', 'speed_mps', 'acceleration_mps2']


@dataclass(frozen=True)
class SupervisedBatch:
    observations: dict
    plans: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    root_ids: tuple
    episode_ids: tuple
    split: str
    manifest_hash: str

    def fingerprint(self):
        h = hashlib.sha256(json.dumps([self.root_ids, self.episode_ids, self.split,
                                      self.manifest_hash], sort_keys=True).encode())
        values = {**self.observations, 'plans': self.plans, 'targets': self.targets, 'target_mask': self.target_mask}
        for name, value in sorted(values.items()):
            x = value.detach().cpu().contiguous()
            h.update(json.dumps([name, list(x.shape), str(x.dtype)]).encode())
            h.update(x.numpy().tobytes())
        return h.hexdigest()


def align_supervision(histories, packs, manifest, config, *, split):
    if split not in ('development', 'train', 'validation', 'calibration', 'test'):
        raise ValueError('Explicit split required')
    if not histories or len(histories) != len(packs):
        raise ValueError('Complete matched root families required')
    pack_map = {(p['episode_id'], p['root_id']): p for p in packs}
    if len(pack_map) != len(packs):
        raise ValueError('Duplicate root label pack')
    rows, roots, episodes, observations, targets, masks = [], [], [], [], [], []
    expected_plans = probe_plans(1, config)[0]
    for history in histories:
        key = (history['episode_id'], history['root_id'])
        if key not in pack_map or key[1] in roots:
            raise ValueError('Missing/duplicate root history')
        pack = pack_map[key]
        if (history['split'] != split or pack['split'] != split or
                history['schema_version'] != 1 or pack['schema_version'] != 1 or
                history['shared_by_candidates'] != list(range(5))):
            raise ValueError('Root/split/candidate contract mismatch')
        source = pack['root_traffic']
        actual = history['history'][-1]
        if actual['simulation_time_s'] != source['simulation_time_s'] or set(actual['vehicles']) != set(source['vehicles']):
            raise ValueError('Input and label root traffic differ')
        for actor, state in source['vehicles'].items():
            if any(actual['vehicles'][actor].get(k) != v for k, v in state.items()):
                raise ValueError('Input and label root actor differ')
        inputs = history['inputs']
        # Collation validates the fixed observation contract separately from labels.
        collate_inputs([inputs], config)
        selected = inputs['actor_ids']
        present = [a for a in selected if a is not None]
        if (len(selected) != config.neighbor_capacity or len(present) != len(set(present)) or
                'ego' in present or any(a not in source['vehicles'] for a in present) or
                [a is not None for a in selected] != inputs['actor_mask']):
            raise ValueError('Invalid stable physical actor IDs/mask')
        origin = source['vehicles']['ego']['position']
        for slot, actor in enumerate(selected):
            if actor is None: continue
            state = source['vehicles'][actor]
            expected = torch.tensor([state['position'][0]-origin[0], state['position'][1]-origin[1], state['speed'], state['acceleration']])
            if not torch.equal(torch.tensor(inputs['actor_features'][slot][-1][:4]), expected):
                raise ValueError('Selected actor root frame mismatch')
        candidates = pack['candidates']
        ids = [c['candidate_id'] for c in candidates]
        if any(type(i) is not int for i in ids) or sorted(ids) != list(range(5)):
            raise ValueError('Require all five unique candidate IDs')
        root_targets, root_masks = [], []
        for candidate in sorted(candidates, key=lambda c: c['candidate_id']):
            cid, labels = candidate['candidate_id'], candidate['labels']
            rows.append({'episode_id': key[0], 'root_id': key[1], 'candidate_id': cid, 'split': split})
            actor_ids = labels['actor_ids']
            if (len(actor_ids) != len(set(actor_ids)) or set(actor_ids) != set(source['vehicles']) or
                    labels['feature_names'] != LABEL_FEATURES):
                raise ValueError('Incomplete/duplicate actor labels or wrong feature order')
            values = torch.as_tensor(labels['future'], dtype=torch.float32)
            valid = torch.as_tensor(labels['future_mask'])
            transition = torch.as_tensor(labels['transition_mask'])
            n, h = len(actor_ids), config.horizon_steps
            if values.shape != (n,h,4) or not torch.isfinite(values).all():
                raise ValueError('Invalid/nonfinite future labels')
            if valid.shape != (n,h) or valid.dtype != torch.bool or transition.shape != (h,) or transition.dtype != torch.bool:
                raise ValueError('Future/transition masks must be Boolean on the frozen grid')
            steps, reason = labels['observed_steps'], labels['termination_reason']
            if type(steps) is not int or not 1 <= steps <= h or reason not in TERMINATIONS | {None}:
                raise ValueError('Invalid terminal metadata')
            if not torch.equal(transition, torch.arange(h) < steps) or (valid & ~transition).any():
                raise ValueError('Trajectory extends beyond observed transitions')
            if reason is None and steps != h:
                raise ValueError('Incomplete nonterminal branch')
            if reason is not None and valid[:, steps-1:].any():
                raise ValueError('Terminal and post-terminal trajectories must remain masked')
            plan = torch.as_tensor(labels['requested_plan'], dtype=torch.float32)
            if not torch.equal(plan, expected_plans[cid]):
                raise ValueError('Candidate ID/declared intervention mismatch')
            target = torch.zeros(config.neighbor_capacity,h,4)
            mask = torch.zeros(config.neighbor_capacity,h,dtype=torch.bool)
            for slot, actor in enumerate(selected):
                if actor is None: continue
                index = actor_ids.index(actor)
                mask[slot] = valid[index]
                target[slot] = torch.where(valid[index, :, None], values[index], 0.)
            root_targets.append(target); root_masks.append(mask)
        roots.append(key[1]); episodes.append(key[0]); observations.append(inputs)
        targets.append(torch.stack(root_targets)); masks.append(torch.stack(root_masks))
    validate_rows(rows, manifest)
    return SupervisedBatch(collate_inputs(observations,config), probe_plans(len(roots),config),
                           torch.stack(targets), torch.stack(masks), tuple(roots), tuple(episodes), split, digest(manifest))
