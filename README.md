# Prediction RL

Local reproduction and action-conditioned prediction research based on
`jlubars/RL-MPC-LaneMerging`.

Current stage: P4e draft protocol/planning interface validated; awaiting user
review before production collector/trainer implementation and formal execution.
Full P4 and formal training readiness are
not yet complete.
Source supplied by the user
remains in `RL-MPC-LaneMerging-master/`; no directory migration is implied.
Original DDPG, reward and continuous jerk control are retained.

Upstream publication permission is unresolved. Do not publish vendored code or
weights. `provenance/` records source identity; `reports/` records verification.
Large/generated outputs belong under ignored `artifacts/`.

The shared pytorch environment is used at the user's request. Do not silently
downgrade its core packages. No formal training is authorized in this stage.

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
