# P7i DDPG stability diagnosis

## Goal and prerequisites

After complete P7h raw-event validation, diagnose learning/execution sensitivity
before expanding the predictor or approving P8. Read all 12 original P7 training
receipts, actor/critic event logs and final weights. Reuse all B0/B1/B2/B3 final
checkpoints, predictor ensembles, original reward, Gym20 execution and seeds.
Complete pinned P7/P7b evidence is required; no permissive lineage repair.

## Allowed and prohibited changes

Only add diagnostic source/config/tests/docs. E:/WCDT_ACCVP is read-only. Do not
change old executable files, configs, checkpoints or artifacts. No training,
replay updates, new reward, ST takeover, shielding, commitment, feature clipping,
action snapping, checkpoint selection, seed deletion or sealed-test access.
Do not pool Gym20 diagnostic outcomes with author50 external-method results.

## Implementation

- Freeze installed ALL 0.5.3 noise/time/train/eval code identities and preset.
- Read q TD MSE, policy negative-Q loss and noisy training episode returns into
  fixed 0/5k/10k/15k/20.5k windows. Keep raw scalar series for future plots.
- Original ALL training tags include `evaluation/returns`; these recorded events
  are noisy training episodes, not independent deterministic validation.
- Audit loss counts against recorded optimizer updates and returns against
  episode receipts. final.pt remains evaluation-only, not a training-resume state.
- Each new rollout calls checked.eval once per decision plus original terminal
  handling, never checked.act. A separate CPU Torch generator adds the original
  zero-mean Gaussian (std 0.5 m/s^3) then original [-5,5] clipping order.
- Share a scene/repeat noise prefix across arms and training seeds. Do not claim
  to recover the unknown original training noise sequence.
- Record actor/noise/preclip/requested jerk, TimeFeature/mask, actual execution
  audit, speed/lane, frozen critic queries and independent raw SUMO events.
- Critic queries never select an action. Realized discounted suffix discrepancy
  is retrospective/descriptive, not Bellman TD error, calibration or causal proof.

## Tests and acceptance

Unit/regression: distribution, clipping, isolated RNG, repeatability, shared
prefix, null denominators, signed actor loss, full seed/scene roster, immutable
receipts and negative-effect aggregation. Bounded smoke: four arms, seed2,
scene200, deterministic and noise repeat0; eight episodes / at most 4,000 steps.
All deterministic traces must match original P7 actions/rewards/native outcomes
exactly. All networks/optimizers/replay remain frozen. Raw collisions override
arrived flags in sidecar analysis only; original rewards/termination stay intact.

## Full request and interpretation

36 deterministic fresh-process replays + 108 noise episodes = 144 episodes,
at most 72,000 decisions, two workers, one Torch thread each. Deterministic runs
are deliberately repeated instead of only reusing P7b because both conditions
need the same independent raw-event reporting, including B1/B2.
There are only three independent traffic scenarios, not 144 independent samples.
First average the three noise repeats within checkpoint/scene, then scenes within
training seed, then equal-weight the original three training seeds. Keep failures
and other terminal categories; no winner, significance or method-effect gate.

## Outputs, Git and pause rules

artifacts/p7i/<short-id>: immutable request, training_audit.json,
training_curves.json, episodes, raw events, aggregate and receipts. Raw outputs
stay ignored. Atomic Conventional Commits on codex/ feature branch, no push.
Stop on malformed input, nonfinite query, unexpected trajectory drift or changed
source; do not retry or overwrite failure. Complete-only resume is permitted.
Noise helping is diagnostic evidence, not approval to use noisy formal evaluation.
Neither P8 nor expanded training/data/model changes are authorized by this task.

Handoff commands: docs/runbooks/p07i_ddpg_stability.md.
