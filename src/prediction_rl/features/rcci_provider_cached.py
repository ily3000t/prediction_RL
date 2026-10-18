"""Opt-in exact cached geometry; no change to old providers or registration."""
from prediction_rl.features.rcci_provider import FrozenRCCIV2Features
from prediction_rl.features.road_consequence_geometry_cached import CachedRoadConsequenceMap


class FrozenCachedRCCIV2Features(FrozenRCCIV2Features):
    def __init__(self, network, config, model):
        super().__init__(network, config, model)
        self.road = CachedRoadConsequenceMap(network)
