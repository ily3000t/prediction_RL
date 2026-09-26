# Dependency and smoke audit - 2026-09-26

Status: PARTIAL / BLOCKED; not a completed paper reproduction.

## Scope and versions

User-supplied source: RL-MPC-LaneMerging-master/. Interpreter: E:\Programs\EnvAnaconda3\envs\pytorch\python.exe.
P0 upstream reference: jlubars/RL-MPC-LaneMerging master d60087f511e7a551e27c4e9fdd87f87f8a84b406. Local ZIP has not been exhaustively verified against that commit. Upstream license remains unresolved; no publication performed.

Installed into the existing environment: Cython 3.0.12, cvxopt 1.3.2, autonomous-learning-library 0.5.3, tensorboardX 2.6.2.2. ALL was installed with --no-deps to avoid unrelated simulator packages. Retained Python 3.10.16, torch 2.5.1, NumPy 2.2.6, Gym 0.26.1 and SUMO 1.22.0. Runtime TraCI came from site-packages (distribution 1.25.0), not SUMO tools.

Before installation pip check passed. After installation it reports ALL requirements opencv-python and pybullet missing. Existing opencv-python-headless 4.12.0.88 provides cv2 and was not overwritten. Tested DDPG imports work, but this is NOT a fully resolved upstream environment.

## Actual tests

- Cython build: failed; Microsoft Visual C++ 14.0 or greater is required. Generated st_cy.c, no compiled extension.
- DDPG imports: passed with permission for import-time runs/ creation.
- Direct ContinuousJerkEnv reset plus 20 zero-jerk steps: passed, observation dimension 20, jerk range [-5,5], finite observations/rewards, final reward approximately -0.02. Loaded train_default_1.json, seed 0.
- Pretrained DDPG: one episode passed; merged=true, crashed=false, 143 control steps, 28.6 simulated seconds. Loaded combined_default_1.json, seed 100, but executed agent.do_control (NO ST takeover), original run_episode with 100-second simulation cap.
- Training environment: reset passed, first step failed because Gym references np.bool8 removed in NumPy 2. Gym 0.26 TimeLimit also expects five step outputs while upstream and ALL use four; fixing only the alias is insufficient.

Smoke processes temporarily set USE_CYTHON=False and SYSTEM=Windows, without modifying source or configuration files. Python ST was selected to avoid the unavailable extension import. No ST or RL+ST episode ran; Python/Cython equivalence is not verified. No optimizer updates or formal experiments occurred.

## Pretrained model SHA256

From pretrained_models/ddpg_default1_extended:

- policy.pt: 548663282e348356f658fe8204b44b7ad589d935b05a1f18fc3d8927b3dd7750
- q.pt: a17d7a891ca773b78f0096e1c1f176f15a2eee51adcb9f981e3f496f6efc5213

## Remaining work

Install Microsoft C++ Build Tools with desktop C++ workload/Windows SDK and rebuild. Save baseline provenance before source compatibility edits. Add project-local Gym compatibility preserving reward, jerk execution and time-limit semantics; do not globally patch site-packages or downgrade shared core packages without approval. Repeat bounded RL/ST/RL+ST smoke and a genuine optimizer-update test.

No source edits, Git commits/tags/pushes, or changes to WCDT_ACCVP were made. Generated artifacts include st_cy.c, runs/, Python caches and SUMO logs; these are not formal results.
