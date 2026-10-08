# P5 difference-objective trial

- Goal: add same-root response-difference supervision and establish legacy-path
  equivalence before spending policy-training budget.
- Base commit: `75051acc06f6ea2ab6dda9be678f191b63d04569`.
- Branch: `codex/p05-response-difference`.
- Worktree: `E:/Prediction_RL_worktrees/response-difference-v1`.
- Allowed: opt-in loss, minimal trainer/artifact hooks, standalone development
  configuration, manual predictor CLI, tests, documentation and local commits.
- Forbidden: edits to either original worktree/artifacts, upstream behavior,
  model architecture, online policy features, reward/actions, training seeds,
  new traffic cohorts, automatic DDPG training or pushing/merging.
- Acceptance: lambda-zero equivalence, aligned common masks, pair/root weighting,
  empty/nonfinite handling, independently trained members, exact resume, strict
  new artifact identity, old artifact regression, read-only reference reuse and
  actual preparation against frozen train/validation data.
- Products: source/tests/docs/config in this branch; ignored requests/checkpoints
  and diagnostics in this worktree's `artifacts/prediction_difference/`.
- Pause: review offline differences and absolute-error tradeoffs before approving
  any new policy run. Do not replace seeds, tune using test/control results or
  change evaluation protocol to produce a favorable outcome.
- Handoff: see `docs/runbooks/p05_candidate_difference_v1.md`.
