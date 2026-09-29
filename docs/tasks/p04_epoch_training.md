# P4h — Completed-data review and epoch training entry

## Goal and prerequisites

Review the user's completed immutable 384-episode collection. Add a CPU trainer
for the already frozen ordinary/conditional predictor protocol. P4g collection,
P4d masked objective and the clean collection revision must be verifiable.

## Scope

Allowed: independent dataset review, predictor epoch driver, strict checkpoints,
tests, bounded two-epoch development audit, manual-start request and documentation.
Forbidden: upstream reward/action changes, old-project changes, formal training,
test collection, calibration fitting, DDPG training, replacing failed seeds.

## Implementation and acceptance

- Verify historical collection code via Git blobs, receipts and artifact hashes;
  do not loosen the collector's current-source resume protection.
- Keep all roots in the review; explicitly exclude all-empty roots from losses.
- Preserve episode splits, full candidate families, paired initialization/order.
- Train members separately, select strictly best validation loss, and use the
  frozen min-delta only for patience; preserve earliest ties.
- Verify exact CPU epoch-boundary resume including Adam, sampler and best state.
- Require an exact request hash and clean code for user-started formal training.
- Save immutable member checkpoints, selection history and uncalibrated ensembles.

## Artifacts, Git and pause conditions

Large artifacts stay under ignored `artifacts/`. Commit reviewed source/tests
before the development audit; record evidence separately. Stop on provenance,
split, nonfinite or deterministic-resume failures. Do not change the data or
training protocol to pass. No push while upstream licensing is unresolved.

## Handoff

The runbook will provide the exact prepared request hash only after tests and
the development audit pass. The user starts the six full training runs.
