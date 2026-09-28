# P4g user-run collection entry

Goal: release an independent manual-start configuration and a source-bound
collector for train/validation/calibration. Existing draft and evidence stay intact.

Prerequisites: P4f collector acceptance and reviewed P4e dataset design. The user
must confirm the prepared request hash when starting their long collection.

Allowed: shared collector core, user-run prepare/run CLI, frozen independent
configuration, tests, documentation and bounded replay on existing development
seeds. Forbidden: executing any new formal seed, test release, training, changes
to original reward/control/network or old-project files, automatic seed replacement.

Acceptance: fresh data construction without old labels; shared-core development
parity; unchanged experimental parameters; strict request/source/environment
binding; explicit manual start; split/test guards; unavailable-root accounting;
failed/partial preservation and verified resume; complete outcome aggregation
without requiring favorable collision/arrival results.

Artifacts: ignored artifacts/p4/<audit> and artifacts/data/<run>/request.json,
preparation.json; NO formal episode folders created by this development task.

Git: codex/p04-formal-collection; code/tests commit before clean-tree bounded
audit and plan generation; record actual acceptance then non-squash local merge.
No push/tag. Stop on parity errors, unresolved provenance or any need to modify
the formally frozen seed budget/thresholds. User runs the commands in the runbook.
