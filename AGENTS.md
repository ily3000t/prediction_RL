# Development boundaries

- This project studies upstream RL-MPC merging with original reward and continuous DDPG. No PPO, new reward, commitment or independent Risk network in P1.
- E:/WCDT_ACCVP is read-only reference. Never stage or alter its files.
- Work on codex/ feature branches. Use atomic Conventional Commits; inspect staged files, exclude weights, raw results, logs and secrets. No destructive resets or forced pushes.
- Keep an immutable local source/provenance baseline before compatibility changes. The upstream license is unresolved: do not publish upstream source or weights until permission is documented. A source tag is not a runnable milestone.
- Preserve upstream reward, action and termination semantics; isolate compatibility in project files, never globally monkey-patch installed packages.
- Follow P0-P9 sequentially. P1 permits bounded smoke only. Formal data collection, ensemble/DDPG training and evaluation are user-run after configuration approval.
- Record resolved configuration, seed, command, environment, commit, dirty diff and hashes for runs. Negative results are valid; never change seeds/thresholds to turn a failure into a pass.
- Do not edit artifacts of active experiments. No silent partial checkpoint loading or wall-clock-dependent observations.
- Document tests actually run, remaining blockers and exact next commands. Never claim a source snapshot or a single successful episode is a reproduced paper.

## Interaction curriculum worktree

- This checkout is the isolated `codex/interaction-curriculum-v1` research branch, based on `75051acc06f6ea2ab6dda9be678f191b63d04569`. E:/Prediction_RL is the validated baseline and is read-only here, including its ignored artifacts. Do not switch, edit, stage, or merge changes into that main checkout.
- Follow `docs/design/interaction_curriculum_v1.md` and the IC01-IC05 acceptance sequence. The design is not yet an executable training protocol. Do not launch training while task construction, reset semantics, thresholds, and the original target distribution remain unresolved.
- Reuse the frozen B2/B3 predictors by verified, explicit read-only asset references. Do not copy old policies into training initialization, silently relax lineage checks, or make writable links to baseline artifacts. Write all new outputs under this checkout's ignored artifacts/ directory with new short run IDs.
- The sole proposed experimental variable is task distribution/order. Preserve reward, terminal/bootstrap semantics, continuous jerk/probes, observation schema, predictor weights, policy architecture, and the original uniform replay/optimization path. Curriculum audit metadata must not enter network inputs.
- A new reset/task protocol, if required, must be named, audited, and explicitly approved before matched training. All four B2/B3 x Mix/Curriculum arms use the same approved task entry protocol. The old B3 is then historical context, not the matched Mix control.
- Permit read-only training-side analysis and explicitly bounded engineering smoke after its protocol is frozen. Full matched training and evaluation remain user-run. Do not open predictor test data, add generalization/author50 experiments, extend budgets, remove failed seeds, or tune strata using evaluation outcomes.
- Keep curriculum work on this branch. Do not merge to main, push, or create a runnable milestone tag without a separate request and completed acceptance evidence.
