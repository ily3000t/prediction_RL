# P7k bounded 60k matched DDPG experiment

Run from `E:\Prediction_RL` in the `pytorch` environment. Do not run old P7/P7j commands
against this request. Config: `configs/development/p07k_matched_60k_v1.json` (fully resolved,
no nested extends). Original reward and environment config remains
`RL-MPC-LaneMerging-master/configs/train_default_1.json`; hashes and resolved settings are recorded.

## Prepare (no training)

```powershell
python -B tools/train_matched_ddpg.py prepare --run-id p7k_60k_v1
```

This requires clean Git, accepted immutable P7j prechecks and the pinned P7 predictors.
Save the printed `confirm_request_hash`. The source/runtime/config must stay frozen until
all stages finish. Copy that value into `HASH_FROM_PREPARE` below.

## User-run training

```powershell
python -B tools/train_matched_ddpg.py run --request artifacts/p7k/p7k_60k_v1/request.json --confirm-request-hash HASH_FROM_PREPARE --stage train --resume
```

12 from-scratch instances, serial training, requested total 720k control steps.
Each instance runs continuously to 60k; 20k/40k snapshots do not reset optimizer,
replay, SUMO or RNG. Original complete-episode overshoot is at most 499 steps per node.
The new 20k node is an internal reference, not a promise of bitwise equality to old P7.
The original 2M scheduler is NOT shortened. B1/B2/B3 initial policy/critic weights must match.

`--resume` ONLY reuses sealed COMPLETE whole jobs. It never loads a `.pt` file to resume
training. An interrupted/failed training directory is retained and rejected, even if
it contains 20k/40k weights. Do not delete it or silently retrain under the same request.
Review failure and create an explicitly documented recovery run/version if needed.
No automatic retries or deadline-based stale features.

## User-run frozen development evaluation

```powershell
python -B tools/train_matched_ddpg.py run --request artifacts/p7k/p7k_60k_v1/request.json --confirm-request-hash HASH_FROM_PREPARE --stage evaluate --resume
python -B tools/train_matched_ddpg.py aggregate --request artifacts/p7k/p7k_60k_v1/request.json --confirm-request-hash HASH_FROM_PREPARE
```

Evaluation requires all 12 verified training jobs. Each checkpoint/scene uses a new process,
Gym20 warmup, no exploration, no optimizer update, one episode, max 500 control steps.
Two independent evaluation workers; blocking exact prevents CPU latency changing observations.
36 snapshots × 20 scenes = 720 evaluations, but ONLY 20 independent traffic scenarios.
Scene pairing is within training seed, then equal means over the three training seeds.
No Author50 data or best-development checkpoint selection.

## Outputs and interpretation

Under `artifacts/p7k/p7k_60k_v1`:

- `request.json`, `preparation.json`: SHA, resolved protocol, commands, environment,
  upstream/predictor/precheck identities and roster.
- `train/<arm>_s<seed>/n20000.pt`, `n40000.pt`, `n60000.pt` plus `.json`: strictly bound
  evaluation-only weights and actual steps, optimizer counts and cumulative coverage.
- `replay.jsonl.gz`: actual current/action/reward/next/mask and physical state measurements;
  requested jerk is the action, measured jerk the environment response.
- `minibatches.jsonl.gz`: actual original sampled transition IDs, including repeats;
  reconstruct sampled coverage from replay IDs without drawing new samples.
- `losses.jsonl.gz`, `training_logs/`, `episodes.json`: original optimizer losses and noisy
  training returns. ALL's `evaluation/returns` TRAINING tag is not independent validation.
- `raw_training_events.jsonl.gz`: per-episode independent raw ego events and endpoint evidence;
  continuing training traffic is not reseeded between episodes.
- `evaluate/<arm>_s<seed>_n<node>_v<scene>/`: fresh frozen episode, actions, states,
  raw events, receipt and resolved settings.
- `training_curves.json`: raw training episode returns and actor/critic loss series.
- `development_curves.json`: genuine independent frozen-policy development node results.
- `aggregate.json`, `aggregate_receipt.json`: paired outcomes, action saturation, low speed,
  failures, actual training/replay and sampled coverage, hashes.

Availability/presence channels are interpreted for learned methods only: zero-filled B1
channels are not evidence of no actors; history coverage for B0/B1 is marked unavailable.
Coverage is cumulative per node. A sampled state can be counted repeatedly by design.
Requested action saturation means `abs(jerk)>=4.99`, not proof the pre-clipped policy was out
of bounds. Raw ego collision and natural arrival are verified separately from native labels;
disagreement invalidates engineering evidence, it does not change the original reward.

After 60k, inspect all three seeds, final paired outcomes and learning curves, then make a
bounded continue/stop decision. Loss trends alone do not prove convergence. No automatic
100k/200k, architecture/reward/Shield changes, seed selection, or formal test opening.

## Agent-only engineering check

```powershell
python -B -m pytest -q tests/test_matched_learning_curve.py
python -B tools/train_matched_ddpg.py audit --run-id p7k_smoke_v1
```

Smoke changes ONLY its separately versioned budget/seed roster/replay warmup/batch;
it is not a learning result. Several small requested nodes may share one episode-boundary
checkpoint; synthetic tests verify continuation across distinct boundaries.
