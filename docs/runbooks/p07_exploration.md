# P7 exploration: four arms, original continuous DDPG

Use the existing pytorch environment and a clean committed tree in
E:\Prediction_RL. No old safe_rl pipeline is involved. Existing frozen three-member
ordinary/conditional predictors are reused; policies are trained from scratch.

## Prepare only

```powershell
cd E:\Prediction_RL
python tools/explore_feature_ddpg.py prepare --config configs/development/p07_exploration_v1.json --run-id p7_explore_v1
```

Preparation does not train or evaluate. It prints request path and
confirm_request_hash. Review the request before running: 12 trainings (four arms
times seeds 0,1,2), 20,000 requested frames each; original 5,000 warmup and
100 minibatch, CPU/one Torch thread. This is an exploratory budget, not proof of
convergence. Select the final checkpoint only; no evaluation-based selection.
The request has complete resolved protocol, source/runtime and predictor hashes.
Use a new short run ID if one already exists.

## User-run training, then evaluation, then aggregate

Replace HASH with the printed hash. Do not run training and evaluation together.

```powershell
python tools/explore_feature_ddpg.py run --request artifacts/p7/p7_explore_v1/request.json --confirm-request-hash HASH --stage train
python tools/explore_feature_ddpg.py run --request artifacts/p7/p7_explore_v1/request.json --confirm-request-hash HASH --stage evaluate
python tools/explore_feature_ddpg.py aggregate --request artifacts/p7/p7_explore_v1/request.json --confirm-request-hash HASH
```

Training is serial. ALL completes the current episode, so actual frames may
exceed the request by up to 499. Evaluation starts only after all training
receipts validate. Each simulator seed 200–219 is initialized in its own fresh
worker; two evaluation workers run concurrently. One deterministic episode per
policy/scene yields 240 episodes. Prediction blocks simulation without timeout
substitution, ST takeover or action snapping. No sealed test data are opened.

Outputs under artifacts/p7/p7_explore_v1:

- request.json / preparation.json: immutable approval and source bindings.
- train/<arm>_s<seed>: final.pt (evaluation-only actor/critic weights), episodes.json,
  resolved_config.json, original TensorBoard logs, report.json / complete.json.
- evaluate/<arm>_s<seed>_v<scene>: episode.json, compact action/zero-channel-query
  trace, resolved_config.json, report.json / complete.json.
- invocations/<id>: child stdout logs and complete/failed stage report.
- aggregate.json: per-arm/per-training-seed summaries, paired scene deltas,
  equal-training-seed descriptive means/SD and input-sensitivity summaries.

Training episodes are not evaluation results. Missing measured jerk/distance
remains null; episode rows record coverage. A smaller duration can mean an early
collision. Front-bumper-point distance is not body clearance. The zero-channel
query is not executed and does not prove causal benefit or safety.

## Reuse and failures

Add --resume to run only to skip verified complete jobs. This does not resume
optimizer/replay state inside interrupted trainings. An incomplete/failed job
blocks reuse; preserve it, diagnose the cause and obtain an approved new request.
No automatic retry, removal of seeds, overwrite, threshold change or unverified
checkpoint reuse. Aggregate is immutable and should be called once. Logs do not
change episode receipts. No wall-clock cutoff changes the driving observation.

## Engineering audit (not the exploration)

```powershell
python tools/explore_feature_ddpg.py audit --run-id p7_workflow_01
```

This runs four bounded 64-frame requests (8/8 smoke warmup/minibatch), at most four
training episodes each, and one development evaluation scene 200 per arm.
Maximum 4,252 transitions. Its completion proves pipeline/checkpoint integrity,
not predictor utility, adequate DDPG learning or paper reproduction.

After exploration finishes, inspect all three training seeds and all outcomes.
If predictions change actions but hurt return/arrival/safety, record that result
and diagnose utilization/task relevance before new budget/model design.
P8 formal training/confirmation remains separately approved.
