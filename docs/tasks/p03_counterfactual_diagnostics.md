# P3 task record

## Objective

Verify repeatable same-root interventions without branch contamination;
separate ego kinematics from action-dependent neighbor response.

## Preconditions

P2 environment parity passed at main 1df4595. Existing pytorch/SUMO environment.
Fixed original reward, DDPG interface, SUMO scenario and execution constraints.

## Allowed/prohibited scope

Allowed: independent branching module, development plan, bounded diagnostics,
tests and audit documentation on codex/p03-counterfactual-diagnostics.
Prohibited: old-project changes, upstream control/reward/config edits, seed
replacement, formal collection/training, or unvalidated native snapshot use.

## Implementation

Exact first-episode prefix replay with joint Python/RNG/traffic signatures;
native RNG/high-precision save/load probe; forward/reverse probe repeats;
same-ID/time absolute neighbor-response metrics with censoring.

## Acceptance and actual result

57 unit tests passed. 6 roots, 60 branch rollouts: exact repeatability passed.
Native continuation failed despite visible root equality.
Neighbor response was exactly zero in all 6 roots; mechanism gate failed.
Pause here instead of implementing/training P4. Negative results remain valid.

## Artifacts and Git

Implementation checkpoint: 3ec6e8f. Evidence: artifacts/p3/p3_dev_01/.
Summary: reports/p03_mechanism_diagnosis_20260926.md.
Source, configuration, tests and docs are local commits; raw trajectories and
SUMO snapshots are ignored. Keep the P3 branch unmerged pending the stage
decision; no push or runnable-success tag.

## Pause condition and handoff

The planned stop condition (no meaningful neighbor response) was reached.
Review neutral merge-region root coverage and unresolved native control-state
restoration. No alternative seeds, horizons or environment parameters have
been applied.

Reproduction command:
python tools/diagnose_branches.py --config configs/development/p03_mechanism_v1.json --run-id local_p3_01

Use a new ID. See runbook; a rerun alone does not resolve the mechanism gap.

## V2 continuation — 2026-09-27

Scope: result-independent geometric root coverage, not altered scenarios or
replacement seeds. Implemented lane-position discovery with strict schema,
bounded budget, unavailable-root reporting and exact discovery/replay equality.
Original v1 shared experimental parameters remain identical (regression tested).

Implementation commit: 78d4bb1. Tests: 77 passed. Clean bounded SUMO run:
artifacts/p3/p3_merge_v2_01/report.json, 249.32 seconds, 9/9 roots and 90 rollouts
including repeats. Engineering PASS; response at 3/9 roots, all in seed 100;
unchanged two-seed mechanism gate FAIL. No native snapshot, training or push.

Acceptance record: reports/p03_merge_region_diagnosis_20260927.md.
P4 remains paused. P3 branch remains unmerged. Next review may authorize a new
author-policy reference-trajectory diagnostic; it is not implemented here.
Do not change the response gate or reuse the current run ID.
