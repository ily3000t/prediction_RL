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
