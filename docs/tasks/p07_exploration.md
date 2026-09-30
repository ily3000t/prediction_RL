# P7: bounded exploratory closed-loop comparison

## Goal and prerequisites

Inspect whether frozen ordinary/action-conditioned prediction information changes
continuous DDPG actions and original upstream outcomes. P6a/P6b engineering audits,
their historical source bindings and strict ensemble identities must verify.
This is development exploration, not formal baseline convergence or a paper result.

## Allowed and prohibited changes

Only the new orchestration, descriptive evaluation, standalone configs, tests and
documentation change. Do not change the upstream environment/reward/termination,
DDPG implementation, feature definitions, predictors or their selected weights.
No ST replacement, new Risk/commitment/reward, probe snapping, stale observations,
future-label access, checkpoint selection on evaluation, excluded seeds or warm-start.

## Frozen v1 plan

- B0 original 20-dimensional observation; B1 matched 149 zero channels; B2 ordinary
  ensemble; B3 conditional ensemble. ALL appends its unchanged time feature.
- Three original coupled run seeds 0,1,2; four arms, 12 independent fresh trainings.
- Request 20,000 frames each, CPU/one Torch thread, one training worker. Preserve
  warmup 5,000, minibatch 100, actor/critic 400/300, LR 0.0002 and all other audited
  ALL defaults, including its original 2,000,000-frame scheduler endpoint.
- Original training ends at episode boundaries. Actual frames can exceed the
  request by up to 499; report the actual budget instead of truncating episodes.
- Final checkpoint only, no selection based on evaluation. This short budget
  does not establish adequate learning or justify formal comparison.
- 20 independent development scene seeds 200–219, each in a fresh SUMO worker,
  with one deterministic evaluation episode per trained policy (240 episodes).
  These are not the sealed predictor test set or a future confirmation set.
- Evaluation uses two isolated processes, blocking-exact computation and unchanged
  continuous jerk execution. Concurrency affects elapsed time, not observations.

## Measurements and interpretation

Report original return, arrival/collision/time-limit rates, simulation duration,
invalid-action rate, mean absolute measured jerk with valid coverage, and minimum
observed 2D front-bumper-point distance. Distance is not polygon/body clearance;
duration alone can look favorable when collision terminates early. Terminal ego
removal can make jerk unavailable: retain null and report valid counts.

Pair simulator scenes within each run seed, then average the three run-seed
deltas equally. Retain all paired rows and descriptive run-seed SD, but no
significance claim, effect-pass gate or 60-independent-scene interpretation.

For B2/B3, also query the same frozen actor after zeroing only prediction channels,
using the exact same time feature as the executed action. The query is not
executed and cannot advance TimeFeature, update weights or sample exploration.
This is input sensitivity, not a causal performance estimate; zero inputs may be
out of distribution. Training returns are stored separately from evaluation.

## Audit, artifacts, Git and pause rules

Each job records actual commit, immutable request/config/environment hashes,
seeds, predictor/policy identities, execution contract, timestamps and inventory.
All runs require clean committed source; docs-only commits are permitted when
normalized source/config/environment fingerprints remain identical.
A complete negative episode is valid. NaN, incompatible model, inconsistent
scene hashes, failed child or incomplete artifact is an engineering failure,
not an episode to exclude. Stop subsequent batches; preserve failures.

Run receipts, weights and logs stay ignored in artifacts/p7/<short_id>.
Use atomic Conventional Commits on codex/p07-exploration-workflow, inspect staged
diff, merge only after bounded tests/audit; no push or upstream redistribution.

The engineering audit is distinct: one run seed, 64 requested frames, at most
four training episodes per arm, 8/8 smoke warmup/minibatch, one evaluation scene
200 per arm. Its maximum is 4,252 transitions, not a learning experiment.
P7 exploration is user-run after request approval; P8 formal freeze still
requires explicit human approval. Negative effects must not trigger seed changes.
