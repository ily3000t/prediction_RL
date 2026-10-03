# P7j: training target and state-distribution prechecks

## Goal / prerequisites

Read verified original P7 training and Gym20 evaluation receipts. Independently
audit terminal rewards, replay tuple alignment, TimeFeature, terminal storage,
reset bridges and original ALL 0.5.3 masked TD targets before approving any 60k
matched learning-curve experiment. Old P7/P7b source and artifacts stay immutable.

## Allowed / prohibited scope

Add new diagnostics, tests and documents only. No changes to original reward,
DDPG, predictor, SUMO control or reset contract. No 60k training, formal test,
Shield, architecture changes, source publication or push. Old WCDT is read-only.

## Implementation and acceptance

- Synthetic transitions use the installed original ALL algorithm. Instrumented
  and uninstrumented actions, replay, optimizer/target/scheduler and RNG match.
- Actual upstream step methods cover collision, arrival, ordinary step and the
  internal deadline's extra cleanup step. Outer registered TimeLimit is separate.
- Four short optimizer jobs, seed2, request64, two-episode cap: at most563
  transitions/job. Replay-start/minibatch8 are ONLY explicit optimizer-smoke
  overrides, not old training or evidence that a policy learns in64 steps.
- Frozen B1/B3 seed2 checkpoints: three continuing Gym20 episodes each at scene200
  (at most3000 controls). Episode0 must exactly match old first-episode actions
  and rewards. No replay/optimizer update; later episodes never replace approved
  evaluation or justify switching to a more favorable protocol.
- Fresh initial-state census across200..219, no policy rollouts. All20 traffic
  hashes must match old P7; prediction history starts at one observed frame.
- Independently observe raw ego collisions/arrivals/removal during new training
  and frozen runs. Any reward/terminal/state error closes the audit, no relabeling
  of rewards and no automatic retry. Result-negative policies are valid.
- Report density, initial state, short history, masks/feature metadata, low-speed
  coverage and measured-vs-commanded jerk. Baseline/zero missing history is N/A.
- Old20k logs do not save full replay or minibatch state coverage: explicitly
  report unavailable, never replace it with new smoke or predictor data coverage.

## Artifacts / Git / pause

artifacts/p7j/<short-id>: request, immutable receipts, historical_limits,
transitions/raw events/states, initial census and aggregate. Raw output ignored.
Feature branch codex/p07j-training-contract-audit; atomic commits, no push.
Pause for any semantic mismatch or missing raw evidence. Passing bounded checks
does not establish historical distribution equality or approve training.

## Conditional next experiment (not implemented/approved here)

After review: B0..B3 x seeds0/1/2, from scratch, continuous60k each. Snapshots at
the first completed-episode boundary at/after20k/40k/60k; store actual steps and
full provenance. Deterministic independent-process Gym20 evaluation at all20
development scenes per snapshot (720 evaluation episodes,20 scenarios). Final60k
is the predeclared endpoint; earlier nodes are not checkpoint selection. Keep
last_frame scheduler settings, all seeds, reward and features frozen. Do not
load final.pt as a resume checkpoint or auto-extend to100k/200k.
