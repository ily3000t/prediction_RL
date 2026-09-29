# P4h dataset review and predictor training entry — 2026-09-29

Implementation commit: `6f2756065995d69792003ef7ac70e7db90c47e4b`.
Branch: `codex/p04-predictor-training`; user commit `2687335` preserved.
No old-project/upstream source edits, formal optimizer run, test collection or
remote push. Licensing remains unresolved. This is an engineering milestone,
not prediction/closed-loop effectiveness evidence or a runnable-method tag.

## Completed user collection

Collection commit: `2687335fcc5257571e5d7b1ef545c892e5558652`, clean tree.
UTC start/end: 2026-09-28 06:45:20 / 16:26:11 (about 9 h 41 min).
All 384 requested episodes completed; none reused; no test episode directories.
Historical source/config blobs, original artifact hashes, all episode inventories,
history/label joins, ID alignment, masks and aggregate counts passed verification.

| Split | Episodes | Reached roots | Eligible trajectory roots | Candidates | Valid selected actor-time cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 256 | 713 | 710 | 3,565 | 743,772 |
| Validation | 64 | 177 | 176 | 885 | 190,044 |
| Calibration | 64 | 169 | 169 | 845 | 179,400 |

All episodes reached at least one target. Three train roots and one validation
root have entirely empty trajectory supervision (15 and 5 empty candidates).
They remain in immutable data/review; loss sampling explicitly excludes them.
Missing internal/downstream targets are retained as termination-before-target,
not replaced: train 12/43, validation 6/9, calibration 9/14.

Neighbor-response roots at the existing descriptive development thresholds
(position >=0.1 m OR speed >=0.05 m/s) are 484/713, 129/177, 118/169
(67.9%, 72.9%, 69.8%). These are not a newly tuned success gate or independent
samples: multiple roots belong to the same episode, and all five candidates share
their root. Response coverage supports an initial training trial, not proof of
adequate sample size, accurate candidate ranking or closed-loop improvement.

Counterfactual candidate terminations:

| Split | Horizon nonterminal | Arrival | Collision | Timeout |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,406 | 834 | 1,325 | 0 |
| Validation | 375 | 191 | 319 | 0 |
| Calibration | 346 | 184 | 315 | 0 |

These are intervened candidate branches, NOT reference-policy evaluation rates.
Terminal/post-terminal trajectories stay masked; short-horizon censoring and the
fixed author-policy state distribution limit interpretation. Physical labels are
for the selected 12 neighbors; omitted summaries provide context, not individual
supervision for every background vehicle. P5 must explicitly assess coverage,
simple references and same-root ranking rather than only average training loss.

## Training implementation and tests actually run

- Historical provenance verification reads Git blobs without executing old code;
  the existing collector's strict current-source resume check is unchanged.
- Independent train/validation loader rejects calibration/test/development access.
- Root-family epoch sampling, paired B2/B3 initialization/order, independent members.
- CPU float32, one Torch thread; frozen 100-epoch/15-patience budget, batch 16.
- Strict best-loss checkpoint with earliest ties; min-delta only for patience.
- Immutable epoch checkpoints include model, Adam, sampling RNG, CPU RNG and best
  model/selection history. Epoch-boundary restoration rejects incompatible state.
- Selected members export as three-member ensembles, explicitly uncalibrated and
  test-unassessed. No DDPG integration or partial-weight loading.
- Manual exact request-hash confirmation, clean source, frozen input/receipt/
  environment hashes; negative training outcomes are not hidden or reseeded.

Full suite: **305 passed in 9.33 s**, including 32 new tests. New tests cover exact
resume/early-stop selection, empty supervision, leakage rejection, tampered model/
Adam/sampler state, historical-source reads, manual confirmation, no-training
prepare, interrupted epoch replay, member reuse and ensemble publication.
Staged whitespace and explicit file/sensitive-string review passed.

Actual bounded development audit `p4_epoch_v1_01`: six train roots from two
development episodes, three validation roots from the third; neither partition
was promoted to formal data. Both modes × seeds 4101/4102/4103 completed two
epochs. Restoring after epoch one reproduced epoch two exactly: model, Adam,
best state and metrics. Ordinary outputs remained candidate-invariant; B2/B3
orders matched. Recorded audit loop elapsed ~7.36 s, excluding initial data load.
No formal training update occurred; this timing is not a full-training ETA.

## Immutable evidence

Paths relative to `E:/Prediction_RL`; large artifacts remain Git-ignored.

| Artifact | SHA256 |
| --- | --- |
| `artifacts/data/response_v1/invocations/701dcad2a0c5/report.json` | `89e85ccdc6181bda5b0d40b5e2c653e0cb068200cc845e8a53dcb7e2017b5232` |
| `artifacts/data_reviews/response_v1_review/review.json` | `0f59c02cb6daa075e65ab0c7461ca5ff5ca1bfb7f13af9d41ef85ae57632d312` |
| `artifacts/p4/p4_epoch_v1_01/report.json` | `deb5ed5dc1e6859d7ff88c69bac198266af5506eac52f8cf9c62943251a486f6` |
| `artifacts/training/predictor_v1/request.json` | `15ba46a2198d81bd53cabab16b83e3e4cf95a34d497eba655a2c59f903f66e24` |

Training request canonical confirmation hash:
`5c8039f45c798e7377f21858fc98da47b3aeffaaf9eee93ca9fc5846520fada7`.
Training/validation tensor fingerprints:
`3bf281dc198b9ed0f912b7f91d5fcb04c2f73ee849949091610bf707a3a19d4b` /
`4e509714cfdb12ff38b89b1a5dfbea146ada533f05afe6e3fbcb9bfc1cc08427`.

Review, development audit and prepare began from clean implementation commit.
Read-only prepared-request verification passed; member directories started=0;
test episode directories=0. The old collection reports are unchanged, including
their historical `formal_training_ready=false`. The new review grants eligibility
for the first training trial separately, not retroactively.

## Remaining limitations and next step

Formal six-member training is user-run. No formal accuracy/calibration or policy
claim yet. Calibration fitting, P5 shared-ego candidate geometry/ranking metrics,
test release, strict inference consumer and DDPG feature integration remain later
stages. Interruption during multi-file publication and stale lock diagnosis are
manual, not auto-repaired. The runbook documents the supported epoch-boundary
resume and immutable-output behavior.

Next command and expected outputs: `docs/runbooks/p04_predictor_training.md`.
