# P2 task record

## Stage objective

Expose upstream reset/step and audit continuous jerk execution, reward,
termination, explicit run seed and traffic continuity without modifying them.

## Preconditions

P1 bounded compatibility passed; local source baseline and hashes exist.
User requested continued development after uploading Git independently.
Upstream publication permission remains unresolved in project records;
this stage does not push code or claim that upload resolved the license.

## Allowed modifications

New src/prediction_rl/envs modules, bounded audit tools, tests and documentation.
Local codex/p02-environment-contract branch and atomic commits.

## Prohibited modifications

No old-project edits. No upstream environment/reward/action/config changes.
No formal data collection, predictor/DDPG training or large policy evaluation.
No new safety intervention, action space or wall-clock observation fallback.

## Implementation

Strict run-session configuration, single-process ownership and cleanup;
continuing-traffic reset contract; original four-value transition adapter;
separate request/reconstruction/measured execution fields; no fabricated values
after ego removal. Process-isolated raw-versus-adapted trace auditor.

## Tests and acceptance

All unit regressions must pass. Raw/adapted full traces must be identical for
streaming, mixed-jerk and time-limit cases. Repeated seed must repeat; changed
seed must change the scene. The internal extra terminal step must remain.

## Outputs

Source: src/prediction_rl/envs/upstream.py.
Audit tool: tools/audit_environment.py.
Evidence: artifacts/p2/p2_contract_01 and artifacts/p2/p2_clean_01.
Summary: reports/p02_environment_acceptance_20260926.md.

## Git requirements

Separate implementation/tests and integration-audit commits, reviewed staged
diffs, no large artifacts or weights. Merge with history retained after passing.
No push this stage.

## Pause conditions

Any unexpected reward/action/traffic difference, nonfinite data, reset mismatch,
seed nonrepeatability, or changed terminal behavior. Investigate instead of
relaxing equality or replacing seeds. P3 must pass before model data collection.

## Handoff command

python tools/audit_environment.py --run-id local_p2_01

Use a fresh ID; see docs/runbooks/p02_environment_audit.md. P3 is not yet
implemented and no invented P3 execution command is provided.
