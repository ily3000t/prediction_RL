# P7b observed-input and execution diagnosis

No retraining is needed. Reuse the 12 P7 final policies and frozen predictors.
Original P7 outputs stay immutable. New results are separate under artifacts/p7b.

## Prepare (offline only)

```powershell
cd E:\Prediction_RL
python tools/diagnose_feature_ddpg.py prepare --config configs/development/p07b_closed_loop_diagnostics_v1.json --run-id p7b_diag_v1
```

This verifies the historical P7 source, runtime and complete results and builds
input references from 710 already eligible training histories. It does not start
SUMO or training, parse future labels or open the sealed test. The three previously
empty training roots remain outside the frozen eligible roster; no new filtering
uses method outcomes. Preparation may take time due to two frozen-ensemble input
passes and hashing. It prints request path and confirm_request_hash.

## User-run diagnosis

Replace HASH with the preparation hash:

```powershell
python tools/diagnose_feature_ddpg.py run --request artifacts/p7b/p7b_diag_v1/request.json --confirm-request-hash HASH
```

Run automatically aggregates when all recordings verify. It uses all 12 policies
and existing development scenes 200,210,219: 36 episodes, maximum 18,000 steps.
Scene selection is first/middle-index/last, not chosen for favorable results.
Two workers; CPU/one Torch thread; original deterministic actions and blocking
simulation. No reward/seed/control/noise change and no new checkpoint selection.
Each full replay must EXACTLY match its original P7 actions, rewards and outcomes.
Recording errors or replay differences preserve a failure and stop later batches.

Outputs:
- request.json / preparation.json / reference.json / reference_rows.json.
- recording/<arm>_s<seed>_v<scene>/observations.json: original observation, actual
  DDPG input including TimeFeature/mask, observed frame, physical input tensors/
  masks and actor identities/roles/omissions. Baseline/zero inputs are shadow-only.
- actions.json / episode.json: original execution, unchanged outcomes.
- diagnostics.json: low speed, lane/progress/speed, command-gap and masked ranges.
- report.json / complete.json: immutable inventory and exact parent parity.
- invocations/<id>: child logs and complete/failed invocation report.
- summary.json: all per-job diagnostics, all_replays_exact; no method-effect gate.

Use --resume only for checksum-verified complete recordings. Failed/incomplete
recordings are not automatically retried or overwritten. An existing summary is
not rewritten. No raw artifacts are committed to Git.

## Interpret carefully

Features/actual state are observed input/execution audits, not a future oracle.
Train min/max and p01/p99 outside fractions are descriptive; different lanes,
incomplete history and absent actors can explain differences. Missing reference
cohorts remain null. Do not clip/renormalize policy inputs based on this diagnosis.
Root/actor/history cells overlap and are not independent scene samples.
The command speed is reconstructed, not a wire-level command capture.
Episode time limit is simulation time/steps, never a CPU inference deadline.
Zero-channel action queries remain non-executed OOD sensitivity checks.
No exploratory-noise experiment is implemented in v1; its role cannot yet be
separated from train/development scenario differences.

Large detailed observation logs are expected; keep them in ignored artifacts.
Diagnostic inference/recording overhead is NOT a deployment-runtime benchmark.

## Engineering audit

```powershell
python tools/diagnose_feature_ddpg.py audit --run-id p7b_record_01
```

Four arms at training seed 2 / scene 200, maximum 2,000 transitions, deliberately
covering long/failed paths. This tests instrumentation/parity, not method effect.
Use a new short run ID if one exists; never delete the old failed result.

After the user-run diagnosis, review lane/low-speed/progress and feature coverage
for all seeds before proposing budget or model changes. P8 remains separately
approved. Current final.pt files are evaluation-only, not resumable training state.
