# P7 exploration workflow acceptance — 2026-09-30

## Scope

P7 orchestration/measurement is implemented and bounded-engineering accepted.
The actual 12-policy / 240-episode development experiment is NOT executed here.
P8 formal freeze remains human-approved. Old E:/WCDT_ACCVP files were untouched;
no predictors, reward, continuous action, upstream code or training algorithm
were changed, and no Git push was attempted.

Implementation commit: 800f2fb39760de6af937030ab450926d151f1aba
(feature branch codex/p07-exploration-workflow, branched from main 1adbd9e).
The acceptance documentation is a separate atomic commit.

## Changes and frozen exploratory request

- Standalone configs: configs/development/p07_exploration_v1.json and
  configs/development/p07_workflow_smoke_v1.json, no inheritance/unknown keys.
- tools/explore_feature_ddpg.py: immutable prepare / train / evaluate / aggregate,
  exact request confirmation, clean-tree source/environment locking, process
  isolation, strict checkpoint provenance and verified-complete-only resume.
- src/prediction_rl/evaluation/exploration.py: upstream outcome measurement and
  paired simulator-scene deltas within each training seed, followed by equal
  seed-level descriptive means/SD. No effect-success gate.
- tests/test_exploration.py, task document and user runbook.

B0/B1/B2/B3 are original observation, matched zero channels, ordinary prediction,
and conditional prediction. Reuse the accepted three-member predictor ensembles
without warm-starting policies. Exploration fixes coupled run seeds 0,1,2,
20,000 requested frames per policy, original 5,000 warmup / 100 minibatch,
final checkpoint selection, CPU/one Torch thread and serial training.
Evaluation fixes development scenes 200–219, two fresh-process workers and
blocking-exact observations. Twenty scenes are shared across training seeds;
240 cells are not 240 independent traffic scenes. No sealed test data are opened.
This small budget cannot certify baseline convergence or formal superiority.

## Tests actually run

466 passed in 12.18 seconds (439 pre-existing + 27 new).
Command: pytorch Python -m pytest tests -q -p no:cacheprovider
with a new ignored artifacts/pytest_p7_<uuid> basetemp.
CLI --help and staged git diff --check pass.
Coverage includes exact configs/job rosters, hierarchical pairing, invalid/
missing/duplicate outcomes, same-scene hashes, read-only zero-channel query,
failed-job preservation, no retry after failure, complete-only reuse, strict
policy lineage and immutable aggregation.

## Real SUMO engineering audit

Command:
python tools/explore_feature_ddpg.py audit --run-id p7_workflow_01

Clean source commit: 800f2fb39760de6af937030ab450926d151f1aba.
Request: artifacts/p7/p7_workflow_01/request.json
Canonical request hash:
cdea40902b33e4dd60946e133e6e882f89985eb581ee675896e97c228c89616b
Aggregate: artifacts/p7/p7_workflow_01/aggregate.json
Aggregate byte SHA256:
1b5613a612aa266ca7cfcb6e7d5e7396a879d851e577ac1798943c2d9f75c781

This is the distinct smoke protocol: one training seed 0, 64 requested frames,
8/8 warmup/minibatch, at most four training episodes and one development
evaluation scene 200 per arm. Actual totals: 430 training + 250 evaluation =
680 transitions, below the predefined 4,252 maximum.

| Arm | Actual training steps | Episodes | Actor/critic Adam updates | Evaluation steps |
| --- | ---: | ---: | ---: | ---: |
| B0 original | 69 | 1 | 62/62 | 74 |
| B1 zero | 158 | 2 | 152/152 | 65 |
| B2 ordinary | 102 | 1 | 95/95 | 55 |
| B3 conditional | 101 | 1 | 94/94 | 56 |

All job inventories verify; final weights load strictly with matching feature,
predictor, upstream config and DDPG contracts. Expanded policies share initial
actor and critic hashes. Frozen predictors remain unchanged. Evaluation policy/
critic hashes remain unchanged, optimizer states stay empty and replay stays
empty. The extra zero-channel query does not execute or advance TimeFeature.

Initial scene hash, identical across the four policies:
99668bc8ec5b89b09e2263bcb1b76c5c3a737db0084e4e9098868038e08e0612

The tiny evaluation includes collisions for B0/B1 and arrivals for B2/B3;
all are valid outcomes and aggregate normally. Do NOT interpret this as a
method ranking, convergence result or significance evidence. Aggregate status
is complete, engineering_complete=true, method_effect_gate=null,
formal_claim=false and test_opened=false.

Real --resume checks then reuse all four train and four evaluation jobs, without
launching simulation/training. Their invocation reports are:
artifacts/p7/p7_workflow_01/invocations/ae9a54db9dcf/report.json
artifacts/p7/p7_workflow_01/invocations/8c1d6bbed16d/report.json

## Environment and unchanged limitations

Torch 2.5.1; NumPy 2.2.6; Gym 0.26.1; ALL 0.5.3; TraCI 1.25.0;
tensorboardX 2.6.2.2; SUMO 1.22.0, CPU float32, one Torch thread.
The request locks SUMO executable byte hash, upstream compiled module hash,
package versions, upstream normalized sources and installed audited ALL sources.
Gym warnings remain informational in this pinned compatibility setup.

Point-distance is not body clearance. Jerk may be unavailable on terminal ego
removal; retain its validity counts. Early termination can reduce duration.
Zero-channel sensitivity may be out of distribution and does not establish
causal benefit. Intermediate upstream logs/checkpoints are not used for selection;
only final weights are used. Weights-only files cannot resume interrupted training.
Failed/incomplete jobs are preserved and block reuse; no automatic seed change,
retry, artifact deletion or failure-to-pass tuning is allowed.

## User handoff

See docs/runbooks/p07_exploration.md. Prepare-only creates the exploration request;
review it, then manually train, evaluate and aggregate in that order with its
printed confirmation hash. Formal experiments, larger budgets and architecture
changes remain separately approved; assess all three seeds including negative
results before P8.
