# P4i — Training acceptance and frozen calibration entry

## Goal / prerequisites

The user completed six formal CPU members under the P4h request. Verify all
member receipts, selected-epoch identity, ensemble state and historical code.
Implement the already declared episode-max calibration protocol, not a new
model-selection rule or an effectiveness gate.

## Allowed / forbidden

Allowed: read-only training review; strict ensemble consumer; separate calibration
loader, episode grouping/quantile algorithm; tests; nine-development-root inference
audit; immutable manual-start calibration request; documentation and atomic commits.
Forbidden: changing existing data/model/training configuration, more epochs,
formal calibration or evaluation without user launch, test collection, DDPG,
old-project edits, upstream publication or threshold/seed replacement.

## Acceptance

- Complete selected ensembles exactly match their best member epochs.
- Only calibration split can fit formal q; training loader remains unchanged.
- All roots/candidates/cells in an episode yield one maximum score.
- Population member disagreement, frozen SI floor, one-based finite-sample order.
- If rank > eligible episode count, preserve unbounded intervals, never clamp.
- Explicit empty episodes, terminal censoring, in-fit rather than test coverage.
- Prepare cannot predict/fit; run needs exact request confirmation and clean code.
- Negative or wide-band outcomes remain valid results, not engineering failures.

## Outputs / Git / handoff

Ignored `artifacts/training_reviews`, `artifacts/p4`, `artifacts/calibration` hold
evidence. Commit source/tests before audit, documentation after. Merge locally
without squash on acceptance; no push/tag. Stop on provenance/load/mask/quantile
errors. User then runs the exact prepared calibration request in the runbook.
P5 reference prediction and same-root candidate-ranking protocol are subsequent
work; calibration alone does not release tests or approve DDPG integration.
