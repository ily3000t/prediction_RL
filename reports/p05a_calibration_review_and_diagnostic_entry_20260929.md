# P5a calibration review and offline diagnostic entry — 2026-09-29

Implementation commit: `4a300d37e27c6e8dabc1ecae160876e74a56791e`.
Branch: `codex/p05-offline-diagnostics`. Existing data, models, training/calibration
configurations and reports remain unchanged. No formal diagnostic, test, SUMO or
DDPG execution. Old project/upstream source unchanged; no push/tag.

## Completed user calibration: engineering success, wide intervals

Run commit `eb981ce6f07f01b3befdb82c8d735deeab5164cd`, clean tree.
UTC 2026-09-29 06:53:37.092751–06:53:41.530372. Both methods use 64 eligible
calibration episodes / 169 roots / 179400 selected actor-time cells, no empty
episodes. Rank 59/64, preserved frozen nominal 90% rule.

| Method | q | Episode fit coverage | Mean full x width (m) | y width (m) | speed width (m/s) | acceleration width (m/s²) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 46.175522 | 92.1875% | 92.4204 | 27.7055 | 46.2012 | 19.0048 |
| Conditional | 48.051269 | 92.1875% | 96.3079 | 28.8308 | 48.1865 | 20.5208 |

Coordinate fit coverage is 99.9990% / 99.9993%. This is not contradictory: a single
extreme coordinate can cause an episode miss while almost all coordinates lie in
very broad bands. The fit is descriptive in-sample coverage, NOT held-out test
evidence. Neither q alone nor calibration success proves useful uncertainty.
Conditional mean-prediction loss improvement from training does not imply better
calibrated bands; widths here are slightly larger, not smaller.

Given these widths, do not claim deployable safety bounds or feed them into a
new safety gate without separate design. Episode-wide maxima, channel scaling,
small ensemble disagreement and tail prediction errors are plausible contributors;
their relative contribution has NOT yet been measured. No q/coverage/floor changes
were made based on these observed results.

## Implemented P5a diagnostic contract

The independent development config fixes validation point-error comparison among
root-anchored map/lane-chain constant speed, ordinary ensemble and conditional
ensemble. Reference has no access to label packs, no candidate conditioning and
no upstream controller mutation. Route geometry is pinned; roundoff between XML
length and polyline arc is handled explicitly and root XY is anchored.

Common masks/candidate/actor layouts; cell→candidate→root and separate episode-equal
aggregation. Last-observed displacement is NOT mislabeled full-horizon FDE; the
latter only uses valid step-25 branches and reports denominators. Empty/censored
roots remain visible. Paired differences use shared root/episode identities, not
pseudo-independent candidate samples. No significance test or performance gate.

All ten candidate pairs provide common-mask response-delta scaled MSE, testing
whether conditional information captures neighbor-response differences. This is
NOT action safety/viability ranking. P5b shared-ego rollout and geometric/terminal
definitions remain pending; no oracle future ego is introduced as a model input.

Frozen-band attribution on validation and calibration identifies the worst
root/actor/candidate/channel/time with truth, mean, std and scale, plus per-channel
floor fractions, coverage and widths. No recalibration occurs. Validation has
already selected checkpoints and calibration fitted q; both explicitly remain
non-independent-test diagnostics. Test episodes remain unopened.

## Tests and bounded checks actually run

Full suite: **355 passed in 9.98 s** (19 new tests). Reference speed/first tick,
map transitions, zero speed/missing actors, no label access, exact/empty predictions,
censoring, nonfinite rejection, candidate-pair mask intersections, root vs episode
weighting, paired identity, argmax on valid cells, calibration attribution,
train/test loader rejection, confirmation and no-evaluation prepare are covered.
Staged whitespace/file/sensitive-string checks passed; no raw outputs staged.

Actual `p5a_interface_v1_01`: 9 roots from the same three development episodes.
Pinned real trained ensembles and CV reference load and produce finite metrics;
all 9 roots per model produce attribution records. Ordinary and CV zero-response
contrast errors are exactly equal, as required by plan invariance. Interface gate
passed. No formal validation/calibration predictions or refits were performed.
Recorded audit loop 0.633 s excludes prerequisite/data loading and is not a
formal-evaluation or deployment runtime claim.

Calibration source review verified request/actual historical code/clean run,
training and model hashes, locked splits and recomputed the frozen order statistic
from already saved episode scores. This is read-only validation of saved numbers,
not a new prediction or fit on calibration trajectories.

## Evidence and handoff

| Relative artifact | SHA256 |
| --- | --- |
| `artifacts/calibration/predictor_v1/report.json` | `c294783af31c2f2c4dac5c22423ce7e42ebd2b65da08fb95afee5388ba47a8ec` |
| `artifacts/calibration/predictor_v1/calibration.json` | `3c015220d3c269f81331224a2eb7ae597824bc78788e6f461b0bd42dadd4d35a` |
| `artifacts/p5/p5a_interface_v1_01/report.json` | `c643cfaf3fe1bdae7289b4b234550cefa58c4cc997be229f5916552e315d2a7a` |
| `artifacts/offline/predictor_v1_diag/request.json` | `bf95d0bcd6f9fd671ddb76e20f51ee239ed1ab0a1802129bfb2e90e9f6691da6` |

Prepared request confirmation hash:
`2a651b355533f1499776f6b060ee7dc87fa2c2604d04dfb446b2e7da1f1377d4`.
Current source/environment/input verification passed; formal started.json absent;
predictor test episode directories=0. Audit/prepare ran from clean implementation.

User command and metric definitions: `docs/runbooks/p05_offline_diagnostics.md`.
P5a results are currently unknown. Do not infer CV superiority, the worst-error
channel, candidate ranking improvement or DDPG readiness before running/reviewing.
