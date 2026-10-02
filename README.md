# Prediction RL

Local reproduction and action-conditioned prediction research based on
`jlubars/RL-MPC-LaneMerging`.

P7d's user-run 140-episode external comparison is now complete and verified.
Author DDPG, ST and both original RL+MPC variants finish all 20 shared scenes;
conditional-feature DDPG finishes 41/60 across three training seeds. Two seeds
show comfort/reward advantages over RL+MPC, but the third parks on the ramp in
19/20 scenes. No model was retrained; P8 is not approved. The proposed frozen
noise diagnostic is still DRAFT, not an implemented CLI. See the
[full P7d result review](reports/p07d_results_review_20261001.md),
[P7d acceptance](reports/p07d_external_baselines_acceptance_20261001.md) and
[exact manual command/contract](docs/runbooks/p07d_external_baselines.md).

P7e's 80 new episodes are now complete and verified with 200 historical references.
Author DDPG reaches 16/20 scenes in P7 versus 20/20 in P7d; project B0 reaches
40/60 versus 60/60 with identical frozen weights. Protocol sensitivity is therefore
material, while B3's third seed still stalls in 19/20 scenes under both protocols.
Native scoring/termination contracts remain separate; no cross-protocol reward
delta, retraining or old-result overwrite occurred. See
[completed P7e review](reports/p07e_results_review_20261002.md),
[P7e acceptance](reports/p07e_crosscheck_acceptance_20261002.md) and
[manual crosscheck runbook](docs/runbooks/p07e_model_protocol_crosscheck.md).

P7f now implements the user-approved driver x warmup diagnosis: Gym/author loop
at 20/50 seconds, with the same frozen models and three preselected scenes.
602 tests and six new bounded engineering episodes pass; all same-warmup initial
traffic pairs match. Full evaluation contains 42 new plus 42 verified historical
cells and is USER-run, not executed here. Native rewards/collision definitions
remain separate and missing historical observations remain explicitly unavailable.
No training, formal P8 approval or protocol selection occurred. See
[P7f acceptance](reports/p07f_driver_warmup_acceptance_20261002.md) and
[manual diagnostic runbook](docs/runbooks/p07f_driver_warmup.md).

The earlier P7 four-arm exploratory DDPG runs and paired evaluation are complete.
The conditional predictor improves on ordinary prediction but does not outperform
the matched zero-channel control. Formal P8 freeze is not recommended yet; see
the [complete result review](reports/p07_results_review_20261001.md).
P7b's user-run 36-episode diagnosis is now complete: all 9,948 transitions exactly
reproduce P7. It confirms actual ramp stagnation, identifies predictor-input
coverage gaps, and finds no evidence of nearest-actor omission or numerical input
explosion in these recordings. These are diagnostic findings, not causal proof
or P8 approval. See the [P7b result review](reports/p07b_results_review_20261001.md),
[engineering acceptance](reports/p07b_recording_acceptance_20261001.md), and
[diagnostic commands](docs/runbooks/p07b_closed_loop_diagnostics.md).
Predictor training/calibration and P5 diagnostics are complete; P6 preserves
the original continuous-control/reward semantics. Long experiments remain user-run;
P8 formal training and confirmation still require explicit approval. See the
[P7 acceptance](reports/p07_exploration_workflow_acceptance_20260930.md) and
[training/evaluation commands](docs/runbooks/p07_exploration.md).
Source supplied by the user
remains in `RL-MPC-LaneMerging-master/`; no directory migration is implied.
Original DDPG, reward and continuous jerk control are retained.

Upstream publication permission is unresolved. Do not publish vendored code or
weights. `provenance/` records source identity; `reports/` records verification.
Large/generated outputs belong under ignored `artifacts/`.

The shared pytorch environment is used at the user's request. Do not silently
downgrade its core packages. No formal training is authorized in this stage;
the released P7 request is a small development exploration, not a paper test.

Bounded RL, ST, RL+ST, optimizer and original training-entry checks now pass.
This is not full paper reproduction. See
[acceptance report](reports/p01_compatibility_acceptance_20260926.md) and
[local runbook](docs/runbooks/p01_baseline_smoke.md) for commands and limitations.

P2 adds an independent continuous-jerk execution audit without changing upstream
training. See [P2 acceptance](reports/p02_environment_acceptance_20260926.md),
[contract](docs/design/p02_environment_contract.md), and
[audit commands](docs/runbooks/p02_environment_audit.md).
P3 exact replay passed. V1 found no neighbor response; v2 neutral merge-region
coverage found responses in 3/9 roots in one seed. V3 uses the pinned author DDPG
reference and finds responses in 5/9 roots across all three seeds, passing the
unchanged mechanism gate. Native snapshot continuation failed parity and remains
disabled. Read the
[P3 findings](reports/p03_mechanism_diagnosis_20260926.md) and
[v2 merge-region findings](reports/p03_merge_region_diagnosis_20260927.md), plus the
[v3 acceptance](reports/p03_author_policy_acceptance_20260927.md) and
[diagnostic runbook](docs/runbooks/p03_mechanism_diagnostics.md).
This verifies simulator mechanism coverage, not prediction or policy improvement.
Formal datasets and training still require the subsequent design/acceptance steps.

P4a validates all 45 existing development branches with episode-group splits and
explicit future/event masks. It does not invent missing history or train models.
See [P4 data design](docs/design/p04_data_contract.md),
[label acceptance](reports/p04a_label_contract_acceptance_20260927.md) and
[offline audit runbook](docs/runbooks/p04_label_audit.md).

P4b adds exact observed history, 12 neighbor slots (ego separate), mandatory
roles and masked omitted-actor summaries without changing original replay.
See [input contract](docs/design/p04_history_actor_contract.md),
[P4b acceptance](reports/p04b_history_actor_acceptance_20260927.md) and
[bounded pilot command](docs/runbooks/p04_history_actors.md).

P4c adds independent 64-wide masked GRU/interaction encoders and ordinary versus
action-conditioned single-trajectory heads, each with three members. The nine
existing development roots pass batching and strict checkpoint tests. These
checkpoints are explicitly UNTRAINED, not prediction-effect evidence or policies.
See [interface design](docs/design/p04_predictor_interface.md),
[P4c acceptance](reports/p04c_predictor_interface_acceptance_20260927.md) and
[offline audit command](docs/runbooks/p04_predictor_interfaces.md).

P4d aligns physical targets by root actor IDs, enforces episode splits and trains
each member separately with an explicit masked trajectory objective. A two-step
CPU smoke on the same nine development roots passes exact model/Adam/inference
resume. These smoke weights are not formally trained or approved for control.
See [supervision design](docs/design/p04_masked_training.md),
[P4d acceptance](reports/p04d_masked_training_acceptance_20260927.md) and
[bounded audit command](docs/runbooks/p04_masked_training.md).

P4e provides a review-only draft for 256 train / 64 validation / 64 calibration /
128 sealed test episodes. Planning validates provenance and split isolation but
does not approve or execute any job. See the
[proposal](docs/design/p04_collection_training_proposal.md),
[planning acceptance](reports/p04e_protocol_planning_acceptance_20260928.md) and
[review runbook](docs/runbooks/p04_protocol_review.md).

P4f verifies a bounded resumable response collector on the same three development
episodes. All 90 branch executions preserve original traces, history and labels;
new future geometry stays in a separate audit sidecar. Checksum-verified resume
reuses all episodes without simulation. This is NOT the formal dataset executor.
See [P4f acceptance](reports/p04f_response_collection_acceptance_20260928.md) and
[collection audit runbook](docs/runbooks/p04_response_collector.md).

P4g adds the separate manual-start train/validation/calibration collector. The
shared core again passes all nine development roots and exact supervised-tensor
parity. A 384-episode request is prepared but NOT executed; 128 test episodes stay
locked. See [P4g acceptance](reports/p04g_formal_collection_entry_acceptance_20260928.md)
and the [user-run collection instructions](docs/runbooks/p04_formal_collection.md).
Full predictor/DDPG training has not started.

P4h now verifies the user's completed 384-episode collection: 710/176/169 usable
trajectory roots in train/validation/calibration, with empty roots preserved and
test still locked. The CPU epoch trainer passes 305 tests and six-member bounded
development resume checks. A full ordinary/conditional training request is
prepared, **not executed**. See [P4h acceptance](reports/p04h_dataset_and_training_entry_acceptance_20260929.md)
and [manual predictor training](docs/runbooks/p04_predictor_training.md).

P4i verifies the completed six-member training and adds a separate episode-level
calibration entry. All selected ensembles match their best epochs; 336 tests and
bounded development inference checks pass. Formal calibration is prepared, **not
fitted**; test and DDPG remain locked. See [training/calibration acceptance](reports/p04i_training_and_calibration_entry_acceptance_20260929.md)
and [manual calibration command](docs/runbooks/p04_predictor_calibration.md).

P5a reviews completed calibration: episode coverage meets the nominal fit target,
but bands are very wide. A separate manual diagnostic now compares validation
errors with a root-only lane-chain CV reference, checks candidate-response deltas,
and attributes frozen-band extremes. 356 tests and a nine-root smoke pass; full
diagnosis is prepared, **not run**. This is not action ranking or test evidence.
See [P5a acceptance](reports/p05a_calibration_review_and_diagnostic_entry_20260929.md)
and [diagnostic command/definitions](docs/runbooks/p05_offline_diagnostics.md).
