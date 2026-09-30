# P5b: shared-ego proximity ranking diagnostic

## Goal and prerequisites

Accept the completed P5a point-error/attribution evidence, then test whether better
neighbor predictions improve **same-root candidate proximity ordering**. This is
an exploratory validation diagnostic, not independent test or DDPG release.
P5a must be complete, its source revision recoverable, root summaries reproducible
from saved rows, and original training/calibration/data hashes unchanged.

## Allowed and prohibited changes

Allowed: new root-only analytic ego rollout, retrospective metric implementation,
independent configuration, bounded development audit, tests and runbook.
Do not change upstream execution/reward, trained weights, data, seeds, selection,
calibration or old reports. Do not run formal validation or training on behalf of
the user. No test collection, DDPG, ST, Risk or new reward is authorized here.

## Implementation and frozen estimand

- Five probes: first-tick jerk −5, −2.5, 0, 2.5, 5; 24 zero-jerk continuation ticks.
  Zero jerk retains acceleration; it is not constant-speed continuation.
- Shared ego motion uses root speed/acceleration/lane position, original clipping
  bounds, 0.2 s control interval and the pinned lane-chain map. Integrate distance
  with new speed and carry `(new_speed-old_speed)/dt` acceleration. This is a
  tested approximation, not a replacement SUMO simulator or safety certificate.
- Reconstruct future ego only from observed root + plan. Compare against observed
  nonterminal ego labels separately. Fixed numerical tolerances: position 0.01 m,
  speed 0.001 m/s, acceleration 0.001 m/s². Empty ego support is unverified, not a
  pass. Inspect parity failures before using geometric conclusions.
- Ordinary/CV/conditional predictions share exactly this ego trajectory. Distance
  means Euclidean distance between front-bumper positions, **not body clearance,
  collision probability, TTC, reward or task value**. Larger minimum distance is
  the proxy preference; it may favor stopping and cannot define a driving policy.
- Candidate scores use the actor-time intersection observed in all five branches
  (including ego). This label-derived censoring is evaluation-only. It must NEVER
  become an online feature/mask. No imputation of terminal or absent trajectories.
- Report two views of the same roots: common observed support, and complete 5 s
  for every root-selected neighbor across all five branches. These are support
  sensitivity views, not new independent cohorts. Early termination can be
  action-dependent; neither view estimates unconditional collision safety.
- Only selected physical neighbors have trajectory outputs. Record omitted root
  actors and newly entering actors; summary tokens do not supply their trajectories.
  Do not interpret selected-actor proximity as all-vehicle safety.
- Report score MAE, ten-pair score-difference MAE, informative-pair ordering
  accuracy, proxy regret and best-proxy numeric-tie hit rate. Numeric equality
  tolerance is 1e-6 m. Predicted ties choose lowest candidate ID. Truth ties are
  excluded from ordering accuracy, counted in its denominator, not called wins.
- Root-equal and episode-equal summaries and paired right-minus-left differences
  are both saved. Adjacent roots/candidates/time samples are not independent.
  No p-values or significance claims. Show score spreads to expose near ties.
- Do not use the P5a very wide calibrated bands to produce safety labels or change
  their calibration. This diagnostic uses frozen ensemble-mean trajectories only.

## Tests and acceptance

Test the isolated original speed helper against the new scalar update, future
isolation, continuation/saturation, lane transitions, perfect/wrong/tied ordering,
shared censoring, empty cases, identity/NaN failures, root/episode weighting,
confirmation/no-overwrite protections, preparation without inference, and normal
completion of negative results. Run full regression suite.

After implementation commit on a clean branch, run only the existing 9-development-
root audit. Require its interface/parity gate before preparing a user-run request.
Bind source, config parameters, environment, prior report, model and data hashes.
Tracked JSON is canonical-hashed to survive LF/CRLF checkout changes; immutable
artifacts retain byte hashes. Record actual tests and audit, then merge locally.

## Pause / handoff

No automatic P6 release. If conditional prediction does not improve ranking,
inspect score spread, censoring, selected actor scope and prediction errors before
any expensive RL work. If ego parity fails, diagnose execution/geometry first.
A positive proxy result remains narrower than task/safety viability. Agree a
P6 feature/utility contract before using any diagnostic as a policy observation.
Commands and output locations: `docs/runbooks/p05_candidate_ranking.md`.
