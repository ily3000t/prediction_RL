# P7k: one bounded matched learning-curve experiment

## Objective and prerequisites

P7j accepted the original reward, transition, bootstrap, action and initial Gym20 contracts.
Its short trajectories did NOT reconstruct historical 20k replay or establish complete
training/evaluation distribution equivalence. The user approved one from-scratch 60k
development experiment after these prechecks, not indefinite training or architecture changes.

## Scope

- B0 baseline, B1 zero channels, B2 ordinary prediction, B3 conditional prediction.
- Training seeds 0/1/2; same original single seed coupling and continuing traffic.
- Reuse the pinned three-member predictors; no predictor training or old-weight warm start.
- Original Slotted Jerk reward, continuous [-5,5] jerk, ALL 0.5.3 DDPG and TimeFeature.
- CPU / one torch thread; original 5000 replay warmup, batch 100, gamma .98,
  lr .0002, original 2,000,000-step scheduler, replay capacity 1,000,000.
- One continuous training instance per cell, requested 20k/40k/60k nodes saved at the
  first complete episode boundary reaching each node. Actual steps are explicit.
- Final 60k is primary; earlier nodes are diagnostics, never best-development selection.
- Independent fresh-process Gym20 evaluation, scenes 200–219 at each node: 720 episodes,
  only 20 unique traffic scenarios. Author50 is not combined with this curve.

## Implementation and acceptance

Only new project-local files may change. Preserve every historical executable file and artifact.
Verify segmentation, snapshot serialization and streaming recorder leave exact RNG,
actions, replay transitions, Adam state, targets and schedules unchanged in synthetic tests.
Record actual replay transitions and ORIGINAL sampled transition IDs, state/actor-history
coverage, original optimizer losses, requested action saturation, low speed and raw event outcomes.
Use scoped raw-event observers; do not change training reward/mask based on diagnostic labels.
Abort invalid/nonfinite runs. Negative method results still aggregate normally.

Agent may run the separate bounded workflow smoke (four <=563-step training instances;
twelve <=500-step fresh evaluation episodes), NOT the full experiment.
Full training/evaluation is user-run after configuration review. No auto-extension or
replacement of failed seeds. Formal claims and new test seeds remain closed.

## Git and handoff

Atomic Conventional Commits on codex/p07k-matched-learning-curve; verify staged files,
no artifacts or upstream publication, no push. Merge tested implementation without squash.
Deliver exact commands, frozen request hash, acceptance record and remaining limits.
