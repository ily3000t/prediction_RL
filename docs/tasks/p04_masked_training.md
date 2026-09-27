# P4d task: supervision alignment and bounded training interfaces

## Goal / prerequisites

P4a label contract, P4b history adapter and P4c predictor interfaces accepted.
Align full candidate families and verify masked learning/resume mechanics.

## Allowed / prohibited scope

Add independent supervision/loss/development trainer, tests, offline audit,
development config and documentation. Do not alter upstream reward, environment,
continuous actions, DDPG, old project, existing evidence or formal seed lists.
No formal training, new simulation collection, test-set tuning or remote push.

## Tasks / acceptance

Validate root, selected actor ID and plan joins; preserve terminal/missing masks;
enforce episode split manifest; keep targets out of model inputs; explicitly
weight trajectory losses and count unsupervised branches. Three independent
member optimizers, strict checkpoint validation and exact two-step CPU resume.
Read the nine pinned development roots; no root/seed filtering by outcome.
Negative/flat loss change is not an engineering failure.

## Artifacts / Git

Code in src/prediction_rl/prediction; outputs ignored under artifacts/p4; small
acceptance record in reports. Work on codex/p04-masked-training. Commit audited
code before clean-tree smoke, commit the acceptance record separately, then
non-squash merge locally after passing checks. No production milestone tag.

## Pause / handoff

Pause if provenance, mask alignment, gradients or exact resume fail. Preserve
failure evidence; no seed/tolerance replacement. Runbook: p04_masked_training.md.
Next is P4e formal collection/training design and review; formal runs remain
user-executed only after review. Full P4/P5 are not complete.
