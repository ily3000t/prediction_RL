# P6a frozen engineering interface

This implements the earlier draft after user authorization. It does not release
DDPG training, change predictors or establish prediction effectiveness.

Original float32 observation is an unchanged prefix. Append **149** channels:

- 120: probe-major, then target front/rear, times 0.2/1/3/5 s,
  then world-axis dx/100 m, dy/100 m, scalar relative speed/30 m/s.
- 20: probe-major, same times, minimum front-bumper Euclidean distance/100 m
  across root-selected physical neighbors, computed AFTER ensemble trajectory mean.
- 9 root channels: front present/history fraction, rear present/history fraction,
  pool present, ego history fraction, selected count/12,
  omitted count/(12+omitted count), omitted-present flag.

Exact names/order are `probe_features.NAMES`. Five probes retain first-step
jerks [-5,-2.5,0,2.5,5], then zero jerk for the remaining 24 original ticks.
Twelve slots exclude ego. Root mandatory identities stay fixed during each
hypothetical prediction. History fraction counts observed cells out of 11.
Missing roles/pool have zero descriptors AND explicit root masks. These are not
future availability masks. No future label fields enter the provider.

Ego geometry uses the already audited root-only constrained kinematic recurrence
and pinned lane chain. Beyond arrival, its lane-end extrapolation is a hypothetical
reference, not a forecast of continued simulation or a safety guarantee. The
feature block has no oracle terminal mask; all methods use the same convention.
No TTC, body clearance, calibrated safety probability, jerk cost, progress reward,
location-dependent CV switching or ensemble-disagreement channel is added.
Normalization scales are fixed SI constants, not learned on validation/test.
Finite implausible predictions are retained, not clipped into safe states.
NaN/shape errors invalidate the wrapper; there is no stale/deadline substitution.

B0 (`baseline`) has original shape. B1 (`zero`) has the same expanded shape as
B2 (`ordinary`) and B3 (`conditional`), with ALL 149 new channels zero. B1 does
not observe traffic or invoke a predictor. B2/B3 reuse strict frozen CPU ensembles.
Only observations change; requested continuous actions, reward and termination
delegate unchanged to the audited upstream environment. Terminal appended blocks
are zero, including time limits; no prediction is attempted after ego removal.
Reset clears predictor history while preserving upstream continuing traffic.

The interface is blocking synchronous, not asynchronous. Timing is diagnostic only.
CV remains an offline reference; no extra closed-loop arm is silently authorized.
The wrapper is not yet connected to original DDPG replay buffers/network/training.
That P6b integration requires explicit network-shape/checkpoint guards and a short
four-arm smoke before user-run exploration, with budget/seeds separately frozen.
