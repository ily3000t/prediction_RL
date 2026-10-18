"""Separate RCCI wrapper: delegates original stepping, refuses Raw identity."""
from prediction_rl.envs.prediction_features import PredictionFeatureEnv
from prediction_rl.features.rcci_provider import FrozenRCCIV2Features
from prediction_rl.features.road_consequences_v2 import VERSION, contract


class RCCIV2FeatureEnv(PredictionFeatureEnv):
    def __init__(self, base, mode, provider, read_frame):
        if mode not in ('ordinary','conditional') or not isinstance(provider,FrozenRCCIV2Features):
            raise ValueError('RCCI requires its dedicated frozen provider')
        super().__init__(base,mode,provider,read_frame)
        # The old Raw preset/checkpoint contract explicitly rejects these arms.
        # No training registration is added in this engineering-only stage.
        self.mode = 'rcci_'+mode
        self.feature_contract = contract()

    def _augment(self, obs, info, terminal=False):
        result,info = super()._augment(obs,info,terminal)
        meta = info['prediction_features']
        meta.update(version=VERSION,release=self.provider.release,old_raw_weights_compatible=False,
                    ensemble_forward_calls=self.provider.calls)
        if not terminal:
            meta['rcci_candidates'] = self.provider.last['result']['candidates']
            meta['observation_time_s'] = self.provider.last['observation_time_s']
        return result,info
