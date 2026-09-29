# P5a — Reference errors, action-response contrast and frozen-band attribution

## Goal / prerequisite

User completed immutable predictor calibration. Its episode-max bands are wide;
diagnose before claiming useful uncertainty or proceeding to DDPG. No new model,
calibration choice or favorable threshold is selected here.

## Allowed / forbidden

Allowed: independent map-aware constant-speed reference, validation metrics,
candidate response contrast error, channel/time/root attribution of frozen bands,
unit tests, nine-development-root smoke, manual-start diagnostic request, docs.
Forbidden: formal diagnostic execution without user launch, test collection,
DDPG training, label-driven actor selection, reward/action changes, old-project
edits, retraining, new q, replacing seeds or overwriting original reports.

## Acceptance

- Constant-speed reference uses only root traffic and pinned map, no future data.
- All methods share physical actors, candidate layouts, labels and observed masks.
- Separate observed-last error from fixed 5-second FDE; count censoring/empty cases.
- Report root-equal and episode-equal values, paired episode/root differences.
- Response contrast uses all ten candidate pairs on common observed masks.
- Residual attribution identifies worst channel/time/actor/candidate without
  fitting new intervals. Calibration-fit diagnostics are not test evidence.
- Clean source, immutable outputs, exact request confirmation; test stays sealed.

## Outputs / Git / pause conditions

Commit source/tests before real development audit, record evidence separately,
merge locally without squash, no push/tag. Outputs in ignored artifacts/p5 and
artifacts/offline. Stop for provenance or metric-contract errors; negative effects
complete normally. User launches the prepared request after reading its scope.

P5a is NOT candidate safety/viability ranking. P5b still needs shared ego rollout
and geometric outcome definitions, including execution constraints and terminal
handling, before ranking can be evaluated. Do not substitute response contrast
or a metric using oracle future ego for an actionable ranking claim.
