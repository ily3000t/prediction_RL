# P1 bounded compatibility acceptance — 2026-09-26

## Scope and outcome

Local Windows compatibility and bounded smoke passed. This is NOT paper-level
reproduction, convergence evidence, or a method comparison. No formal training,
data collection, P2 environment adapter or ACCVP implementation was started.
The old E:/WCDT_ACCVP project was not edited or staged.

The earlier p01_environment_smoke_20260926.md is a historical pre-fix record.
This report supersedes its current-status conclusions, without altering evidence.

## Provenance and permissions

126/126 supplied source/model blobs matched upstream commit
d60087f511e7a551e27c4e9fdd87f87f8a84b406 before compatibility edits.
See provenance/upstream_snapshot.json. Local source baseline: 08ea31c;
immutable source tag: upstream/rl-mpc-d60087f (NOT a reproduction milestone).
License/republication permission remains unresolved. No source push was made.
Weights/raw results are ignored, not committed. No reproduction tag was created.

## Changes

- 44b5c3b: project-local Gym four-result wrapper retains registered 500-step
  limit, reward, action conversion and original ALL observation dtype conversion.
  No global np.bool8 alias, package monkey patch or core-package downgrade.
  Time-limit behavior follows Gym 0.23.1:
  https://raw.githubusercontent.com/openai/gym/0.23.1/gym/wrappers/time_limit.py
- 7e3efa7: ALL 0.5.3 uses colons in log directory timestamps, illegal on Windows.
  Local writer subclass changes directory naming only; training loop and logging
  methods are inherited. ddpg.py selects this class.
- a067c9d: bounded audit runner with unique output IDs, resolved config/hash,
  source snapshots/hashes, commit, dirty diff, seeds, checkpoint hashes,
  environment versions and logs.
- 1332df7: diagnostic Tee tolerates logger shutdown after its log file closes.
  No simulator or model behavior changed.

Git comparison with the source tag confirms configs/, config.py, merge_gym.py,
control.py, prediction.py, st.py and st_cy.pyx were not changed.

## Environment

Existing pytorch environment retained: Python 3.10.16, torch 2.5.1 CUDA 12.4,
NumPy 2.2.6, Gym 0.26.1. Added Cython 3.0.12, cvxopt 1.3.2,
autonomous-learning-library 0.5.3 and tensorboardX 2.6.2.2.
ALL was installed without unrelated optional simulator dependency expansion.

Microsoft Build Tools 2022 17.14.41 was installed at
E:/Programs/MicrosoftBuildTools2022 with VCTools and recommended SDK components.
Installer signature was valid Microsoft Corporation; SHA256:
985969f472caad75d993a5cb4c35a6a4271460cc12b343e2433b994d173aa990.
Original setup.py build_ext --inplace succeeded. Extension SHA256:
498a57f2096e905bb1151eb37dfde41cfb1b76668c343c16e55ae63fcebc76b3.

SUMO executable: E:/Program Files/sumo-1.22.0/bin/sumo.exe, version 1.22.0.
TraCI package: 1.25.0 from site-packages. This version difference is recorded,
not asserted equivalent to the author's original SUMO environment.
pip check still reports ALL requires pybullet and opencv-python.
Existing opencv-python-headless supplies cv2; no overlapping cv2 distribution
was installed. DDPG/SUMO tests pass; the complete upstream dependency set is
NOT declared resolved. Shared core packages were not downgraded.

## Actual bounded results

Clean-source runs at a067c9d are in artifacts/smoke/p1_clean_<mode>/.
RL/ST/combined each use original combined_default_1.json, seed 100,
Cython enabled, 100 simulated second cap. Author checkpoints are used for RL
and combined. Training/semantics use train_default_1.json, seed 0.

| Diagnostic | Observed result |
| --- | --- |
| RL | merged, no collision, 143 steps, 28.6 simulated seconds |
| ST | merged, no collision, 58 steps, 11.6 simulated seconds |
| RL + ST | merged, no collision, 192 steps, 38.4 simulated seconds, 6 takeover steps |
| Raw vs adapted | 20 identical action steps; observation at declared dtype, reward and done exactly match |
| Optimizer | 64 steps; actor L1 parameter change 77.28557184, critic 127.61608799; finite parameters |
| Original DDPGAgent.train | one episode, 54 environment steps (ALL frame counter ends at 55), return 8.69956463; TensorBoard log preserved |
| Unit/regression tests | 13 passed |

Optimizer-only smoke explicitly uses warmup=8 and minibatch=8; it does NOT
change formal config. The separate original training-entry smoke leaves these
defaults unchanged. Requesting one frame runs one whole episode because ALL
checks budget between episodes; registered cap remains 500.

The clean suite at a067c9d completed with exit code 0 but emitted a diagnostic
Tee shutdown warning after saving results. Fixed in 1332df7; clean run
artifacts/smoke/p1_final_entry/report.json confirms normal exit and the same
training return. Its console.log retains the original training output.

"ST solver not happy with rollout state" occurred six times in combined mode.
This is the original dqn.py supervisor branch invoking ST, not an unhandled
exception. A single episode cannot establish takeover quality or safety.

## Preserved failures and diagnostic corrections

- p1_semantics_01: invalid initial test comparison retained as failed.
  Sequential SUMO starts inherited the upstream module-global control.delay;
  raw float64 observations were also compared against ALL's existing float32
  conversion. Corrected test resets delay to its fresh-process value for BOTH
  traces and compares at the declared dtype, without numerical tolerances.
  The production traffic logic was not changed. This global state warrants P2
  seed/reset auditing, not a silent semantic change in P1.
- p1_entry_01: Windows log-path error retained. p1_entry_02 passed after the
  local writer adaptation.
- p1_train_01 incorrectly recorded shutil.which('sumo') as ./sumo.PY.
  Actual subprocess version was SUMO 1.22.0. Later runs explicitly resolve
  sumo.exe; earlier report was not overwritten.
- One pytest invocation hit an inaccessible system temp directory. A fresh,
  unique artifacts/pytest_p1_<uuid> directory resolved test fixture setup.
- Old failed reports and successful reruns were not overwritten or deleted.

## Limits and next gate

No long-run convergence, multi-seed performance, checkpoint resume cycle,
Python/Cython equivalence, original main.py dispatch, or paper numerical match
has been established. Tested training entry is DDPGAgent.train plus a separate
real optimizer-update smoke. Source checkpoint loading still uses upstream
full-object torch.load; use only verified trusted author files.

Next scoped work is P2 environment/execution/seed auditing, then P3 snapshot
branching and neighbor-response diagnostics. Do not begin large predictor/DDPG
training based solely on this smoke. Publishing remains blocked by permission.
