# P4c task: ordinary/conditional response interfaces

## Goal and prerequisites

Accepted P4b histories/actor adapter and P4a labels. Implement independently of
the old project. Establish masks, conditioning isolation and complete checkpoint
semantics before any training.

## Allowed scope / forbidden scope

Allowed: new prediction modules, unit tests, bounded offline audit, documentation
and development configuration. Forbidden: upstream environment/reward changes,
old-project writes, SUMO collection, training, evaluation tuning or publication.

## Implementation / acceptance

Shared GRU and one actor attention layer, dedicated summary encoder, three
independent single-mode response members, matched B2/B3 plan information ablation.
Require empty/history/summary mask checks, slot equivariance, B2 independence,
B3 gradient connectivity, root/candidate batching equality, exact strict
checkpoint roundtrip and explicit invalid-state rejection. Reuse all nine pinned
development roots; no source/result overwrites or favorable subset selection.

## Artifacts / commits

Code in src/prediction_rl/prediction; reports/ contains small acceptance record;
raw reports and untrained checkpoints in ignored artifacts/p4/<short-run-id>.
Feature branch codex/p04-predictor-interfaces; code/test/docs atomic commit before
clean-tree audit, evidence documentation commit afterward, non-squash local merge
only after acceptance. No remote push while license permission is unresolved.

## Pause and handoff

Any invariance/provenance failure pauses acceptance; keep failed artifacts and
fix the implementation, not tolerance/seeds. Commands are in the P4c runbook.
Next is P4d target/loss/training interfaces, not formal training or P5 claims.
