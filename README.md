# Prediction RL

Local reproduction and action-conditioned prediction research based on
`jlubars/RL-MPC-LaneMerging`.

Current stage: P4a label/split contract passed; next is P4b history and actor
adapter implementation. Full P4 and training readiness are not yet complete.
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
