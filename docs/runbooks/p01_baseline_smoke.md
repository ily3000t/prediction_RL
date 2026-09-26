# P1 local baseline runbook

## Goal and boundary

Verify upstream RL/ST/RL+ST and DDPG execution, preserving original reward and
continuous jerk. Run from E:/Prediction_RL in the existing pytorch environment.
These commands are bounded diagnostics, not full reproduction or training.
No P2-P9 CLI is implied.

## Prerequisites

The user-supplied RL-MPC-LaneMerging-master tree and its verified author
pretrained_models/ddpg_default1_extended checkpoint must be present.
Weights are not in Git. See provenance/upstream_snapshot.json for hashes.
SUMO 1.22.0 must be on PATH; SUMO_HOME points to its installation.
Keep the pinned local ALL/Gym compatibility modules; do not pip-upgrade blindly.

If the Cython extension is absent, build from source in the pytorch environment
with the installed Microsoft C++ compiler:

~~~powershell
conda activate pytorch
Set-Location E:/Prediction_RL/RL-MPC-LaneMerging-master
python setup.py build_ext --inplace
Set-Location E:/Prediction_RL
~~~

Installed additions: Cython==3.0.12, cvxopt==1.3.2,
autonomous-learning-library==0.5.3 (--no-deps), tensorboardX==2.6.2.2.
This is a tested local subset, not a complete portable environment lock.
Do not install opencv-python on top of the existing headless cv2 automatically.

## Tests and bounded runs

Use a NEW run ID each time; existing directories are rejected, never overwritten.

~~~powershell
conda activate pytorch
Set-Location E:/Prediction_RL
$taskTemp = 'E:/Prediction_RL/artifacts/pytest_' + [guid]::NewGuid().ToString('N')
python -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp

python tools/smoke_baseline.py --mode semantics --run-id local_semantics_01
python tools/smoke_baseline.py --mode rl --run-id local_rl_01
python tools/smoke_baseline.py --mode st --run-id local_st_01
python tools/smoke_baseline.py --mode combined --run-id local_combined_01
python tools/smoke_baseline.py --mode train --run-id local_optimizer_01
python tools/smoke_baseline.py --mode train_entry --run-id local_entry_01
~~~

- semantics: two 20-step traces with identical prescribed actions.
- rl/st/combined: one episode each, cap 100 simulated seconds.
- train: 64 steps, diagnostic warmup/batch 8/8, verifies parameter updates.
- train_entry: original DDPGAgent.train(1), one episode capped at 500 steps,
  original warmup/batch unchanged; verifies entry and TensorBoard saving.

Configs remain in RL-MPC-LaneMerging-master/configs/:
combined_default_1.json for the three controller checks; train_default_1.json
for training and equality checks. No config file is edited.
The script records SYSTEM=Windows, diagnostic budgets and original checkpoint
paths. train_entry changes only its log output destination.

Each run writes artifacts/smoke/<run-id>/report.json, resolved_config.json,
working_tree.diff, source_snapshot/, and console.log. Semantics adds traces;
training entry adds training_logs/. Large outputs stay out of Git.
TensorBoard can view artifacts/smoke/p1_final_entry/training_logs, but one
episode is NOT a learning curve or evidence of convergence.

## Acceptance, Git and pause conditions

See reports/p01_compatibility_acceptance_20260926.md for actual evidence.
Pause on nonfinite values, loading failures or unexpected action/reward changes.
Negative episode outcomes are results, not reasons to change seeds.
Formal training remains user-run only after its configuration is approved.
P2 must audit reset/global traffic state; P3 must test whether jerk probes
actually change neighbor responses before large action-conditioned training.

Local feature commits retain source provenance. Do not push the vendored
upstream tree until license/republication permission is documented.
Do not describe the upstream source tag as a reproducible-paper milestone.
