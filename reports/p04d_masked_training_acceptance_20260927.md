# P4d masked supervision and resume acceptance — 2026-09-27

Result: **training_interface_gate=true**. This is engineering acceptance only.
formal_training_ready=false. No new SUMO episode, formal predictor training,
calibration, checkpoint selection or DDPG experiment ran.

## Provenance

- Code commit: ab5adb44c7ad46c1356b05ebbeb03f9778b5ffea; clean worktree at execution.
- Config: configs/development/p04_masked_training_v1.json.
- Config SHA256: 8b892ab14408a95f53bb5d8efa4d286efaad2597d065c8e79e5cd5e141b8e4b5.
- Resolved config SHA256: 7f51e48e106dcfb6c9047e0d800b72dce3240ac367383189c419477701cf9053.
- Report: artifacts/p4/p4_training_v1_01/report.json.
- Report SHA256: 174146a8d78962929e18bcd426f9c136db26f0f94deda5bfc449c488459dc3b5.
- Aligned batch fingerprint: 904c3ab7a40a2e3a22674ef373577ac8fda7fc3fdd62817a9ffe2cf942eacae2.
- Input reports: previously accepted P4a/P4b/P4c, all pinned by SHA256 in config.
- All history files, label packs, source reports and split manifest remained unchanged.
- Existing development simulator seeds: 0, 1, 100; member initialization seeds: 3101, 3102, 3103.
- CPU float32, one Torch thread; Python 3.10.16 / Torch 2.5.1 / NumPy 2.2.6.

## Supervision coverage

| Item | Result |
| --- | ---: |
| Original development episodes | 3 |
| Root families | 9 |
| Complete candidate branches | 45 |
| Selected physical neighbor capacity per root | 12 |
| Future time steps | 25 |
| Valid actor/time cells | 8,988 |
| Total actor/time mask cells | 13,500 |
| Fully unsupervised candidate branches | 0 |
| Fully unsupervised roots | 0 |

The target tensor is [9,5,12,25,4]. Masks are [9,5,12,25]. Terminal/missing cells
are not zero-trajectory supervision. Ego and summary tokens are excluded from
physical-neighbor targets. Cell count is NOT an independent statistical sample
count: the evidence still comes from only three development episodes.

## Bounded update and resume checks

| Check | Ordinary B2 | Conditional B3 |
| --- | --- | --- |
| Independent members updated | 3/3 | 3/3 |
| Updates per member trajectory | 2 | 2 |
| Extra second-step replay on restored copy | 1 | 1 |
| Resumed vs uninterrupted model tensors | exact | exact |
| Resumed vs uninterrupted Adam moments/steps | exact | exact |
| Resumed vs uninterrupted inference | exact | exact |
| Finite losses and gradients | pass | pass |
| Same complete supervision batch | yes | yes |

B2 remains invariant to candidate jerk after these updates. Each member optimizes
its own loss, not the ensemble mean. The fixed batch is deterministic with no
dropout, shuffle or scheduler. This acceptance does not establish exact resume
for a future stochastic/GPU trainer.

Per-member step losses and checkpoint hashes are retained in the raw report.
Loss values are measured before the corresponding update; they are not validation
scores or epoch curves. Loss reduction was not an acceptance criterion. Do not
use this smoke to infer convergence, B3 superiority or deployability.

## Regression and implementation boundaries

- Added P4d tests: 27 passed.
- Full regression: 183 passed (pytest reported 6.43 seconds).
- git diff --cached --check passed.
- Tests cover ID/candidate permutation, incomplete families, root/feature/plan
  mismatch, episode split conflicts, terminal censoring, nonfinite data, root/
  candidate loss weighting, masked gradients, changed data, two-step budget,
  exact optimizer resume and invalid checkpoint/moment rejection.
- Old project, upstream source/reward/action/termination and DDPG unchanged.
- DevelopmentTrainer cannot become a formal trainer by increasing a CLI flag;
  its two-step bound and development-only split are explicit contracts.

## Next work

P4e: design/review the formal collection and training protocol, root sampling,
episode coverage, held-out split rules, budgets, selection and calibration plans;
then implement their controlled entry points. No formal dataset or full training
is approved by this report. Shared ego-plan/geometric features and P5 prediction/
candidate-ranking checks remain outstanding before expensive DDPG comparison.

No remote push or production milestone tag. Upstream publication permission
remains unresolved. Checkpoints/raw logs remain under ignored artifacts/.
