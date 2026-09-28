# P4e planning-interface acceptance — 2026-09-28

Result: plan_validation_pass=true; status=review_only.
execution_authorized=false; formal_training_ready=false.
This accepts the engineering planner, NOT the proposed experiment or budgets.
No scene collection, predictor training or DDPG evaluation was run.

## Provenance

- Code commit: e26efe1c0a94582c77452a0dc558e1a09b021ae0; clean tree at preparation.
- Config: configs/formal/p04_response_dataset_v1_draft.json.
- Run report: artifacts/p4/p4_plan_v1_01/planning_report.json.
- Report SHA256: 325f44f54cb4f092bd2c5887788ec98e134ff641a93f2635f4de25848368cdd5.
- Plan SHA256: 809705aaf9269340297cee2e6c5962f30e51a133997e750f6bc37dffb0fc834e.
- Split manifest SHA256: d985999bcc38461360f1dec21e8f31113250925c911133cfb5047bf6a64b5d59.
- Resolved protocol SHA256: 63406a1bb3f6f935ca0fb0df685980ad7d1edc16900aef83b37dde5e54ebb6ea.
- Original P3/P4d prerequisite reports, upstream files, reference checkpoint and
  existing development manifest verified by hash and unchanged after planning.
- Installed versions recorded; SUMO --version only, no simulation process/session.

The interrupted previous turn left four source/config/test files but no design
documents or full-suite run. Work resumed without recreating existing artifacts.

## Plan counts (proposed, not executed)

| Split | Episodes | Maximum roots | Unique branches | Branch executions with repeats |
| --- | ---: | ---: | ---: | ---: |
| Train | 256 | 768 | 3840 | 7680 |
| Validation | 64 | 192 | 960 | 1920 |
| Calibration | 64 | 192 | 960 | 1920 |
| Sealed test | 128 | 384 | 1920 | 3840 |
| Total new | 512 | 1536 | 7680 | 15360 |

The existing three inspected development episodes stay separate. All 512
collection jobs and six proposed B2/B3 member-training jobs are disabled. Each
training job has an upper bound of 4800 optimizer steps under the draft, assuming
all train roots exist and no early stopping. No budget adequacy claim is made.

## Validation

- Planner tests: 27 passed.
- Full regression: 210 passed (pytest reported 5.70 seconds).
- git diff --cached --check passed.
- Tests cover split overlap, inspected seed reuse, invalid/unknown configuration,
  test unlocking, self-approval, architecture/reward overrides, deterministic
  manifests, stable episode identities and no-execution Settings parsing.
- All 112 resolved upstream settings semantically match the accepted P4b seed-0
  settings. File hashes differ only due to serialization ordering of stringified
  ACCELERATION_VALUES_DQN dictionary keys; its individual values also match.
  No reward/action/config parameter change is hidden by this ordering difference.
- Model weights were hashed as source artifacts, never loaded for planning.

## Review and remaining work

User review is pending for the 256/64/64/128 episode budgets/ranges, the three-root
author-policy coverage restriction, 100-epoch / 15-patience checkpoint rule and
the proposed observed-coordinate calibration scope. A 90% calibration target is
not an established coverage or safety guarantee.

Production collection, shuffled training/resume, calibration, shared ego-plan
geometry and P5 candidate-ranking definitions are still pending. Do not run the
base upstream settings through main.py or treat an engineering plan-valid flag
as experiment authorization. Formal runs remain user-executed after review and
implementation acceptance.

No old-project/source-result mutation, remote push or runnable milestone tag.
Upstream publication permission remains unresolved.
