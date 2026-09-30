# P6b original DDPG integration

Reuse installed ALL 0.5.3 `ddpg`, not a rewritten DDPG. Reuse its 400/300 MLP,
Adam, Gaussian exploration and clipping, uniform replay, discount 0.98, Polyak
0.005, update frequency 1, replay capacity 1e6 and cosine schedule horizon 2e6.
Use original upstream `Settings.LEARNING_RATE` (2e-4 in the selected configuration).
The factory checks the installed preset signature/defaults and network dimensions.
TimeFeature(scale0.001) is retained: B0 actor21/critic22 inputs; expanded arms
actor170/critic171. No new action discretization, reward, ST or commitment.

`LegacyFeatureView` discards only reset info when converting to ALL's legacy API.
Step info preserves original upstream keys while keeping bulky audit/feature
metadata in a separate recorder. Learning does not use those metadata fields;
full traffic dictionaries should not be retained per replay transition.
Checked replay delegates original storage/sampling once, preserving terminal-mask
behavior. NaN, incorrect dimensions/masks and illegal continuous actions raise
errors; no repair, truncation or fallback. ALL's original episode-boundary budget
check remains in force; requested frames are NOT an exact mid-episode cutoff.

B1/B2/B3 have identical layouts/network sizes. Shared run seed produces identical
initial actor/critic weights, verified in separate processes; subsequent weights
need not match when prediction inputs affect control. Frozen CPU predictor weights
are never part of policy optimizers. Loading them occurs before the owned seeded
session so predictor construction cannot shift policy initialization.

## Weights contract

New immutable state-dict checkpoints bind arm, observation/time layout, feature
name hash, predictor hash, upstream configuration hash, action limits, ALL version
and resolved preset. Both policy/critic are validated fully BEFORE either is
loaded. No partial loading, silent B0-to-expanded transplant or B2/B3 swap.
Loading uses `weights_only=True`; legacy pickled upstream modules are not imported
through this interface. These checkpoints are **evaluation only**, not optimizer/
replay/RNG resumes; their loaded agent refuses `act()` training calls. Strict
resume requires a separate complete training-state design, not fresh Adam state
silently substituted for a continued run.

## Development optimizer smoke only

Four arms, run seed0 (same upstream/optimizer seed semantics), requested64 frames,
at most4 complete episodes, each original500-step cap. Total per-arm upper bound
563 steps due episode-boundary checking; four-arm upper bound2,252 steps.
Use CPU/one Torch thread for bounded diagnostics. Two explicitly logged smoke
overrides only: warmup8/minibatch8, as in the earlier P1 optimizer diagnostic.
The ordinary non-smoke factory retains warmup5000/minibatch100; it is NOT released
as an approved formal CLI or an outcome-selected training protocol.

Positive optimizer changes, finite parameters, predictor freezing, replay shape,
exact weight roundtrip and matched initialization are engineering gates. Episode
return, collision or success cannot determine this smoke gate. Negative method
outcomes require future paired evaluation, not seed deletion.

Next P7 work: freeze small exploration seeds, training budget, selection rule and
development evaluation set, then provide user-run training/evaluation commands.
Do not treat smoke weights as converged policies or use them as formal warm starts.
