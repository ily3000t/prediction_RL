"""Read-only visit-distribution diagnostics, not policy inputs or new labels."""
from collections import Counter
import numpy as np


class InterfaceCoverage:
    def __init__(self):
        self.observations = 0; self.probes = 0; self.following = Counter(); self.cross = Counter()
        self.failures = Counter(); self.latencies = []; self.omitted_observations = 0

    def add(self, obs, info):
        meta = info['prediction_features']
        if meta['terminal_zero_block']: return
        self.observations += 1; self.latencies.append(float(meta['latency_s']))
        self.omitted_observations += int(obs[-1] > 0)
        for c in meta.get('rcci_candidates', []):
            self.probes += 1
            self.following[str(c['front']['shared_prefix_points'])] += 1
            self.cross[str(c['conflict'].get('shared_prefix_points','not_applicable'))] += 1
            for p in (c['ego_prefix'], *c['actor_prefixes'].values()):
                if p['first_unavailable_reason'] is not None: self.failures[p['first_unavailable_reason']] += 1

    def report(self):
        a = self.latencies
        return {'decision_observations':self.observations,'probes':self.probes,
            'following_prefix_points':dict(self.following),'cross_prefix_points':dict(self.cross),
            'repeated_track_first_failure_reasons':dict(self.failures),
            'omitted_actor_observations':self.omitted_observations,
            'latency_s':None if not a else {'count':len(a),'mean':float(np.mean(a)),
                **{'p'+str(p):float(np.percentile(a,p)) for p in (50,95,99)},'max':max(a)},
            'scope':'visited_decision_states_descriptive_not_independent_scenes_or_deployment'}
