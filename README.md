# Prediction RL

Local reproduction and action-conditioned prediction research based on
`jlubars/RL-MPC-LaneMerging`.

Current stage: P3 diagnostics implemented; mechanism gate failed and P4 is paused. Source supplied by the user
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
coverage found responses in 3/9 roots, all in one seed, so the unchanged
two-seed mechanism gate still fails. Native snapshot continuation also failed
parity and remains disabled. Read the
[P3 findings](reports/p03_mechanism_diagnosis_20260926.md) and
[v2 merge-region findings](reports/p03_merge_region_diagnosis_20260927.md), plus the
[diagnostic runbook](docs/runbooks/p03_mechanism_diagnostics.md).
Do not start predictor training until the interaction-coverage decision is resolved.
