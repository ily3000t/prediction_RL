# P4b bounded history/actor pilot

~~~powershell
conda activate pytorch
Set-Location E:/Prediction_RL
python tools/audit_history_actors.py --config configs/development/p04_history_actor_v1.json --run-id p4_history_local_01
~~~

Run from a clean commit with a fresh short ID. Three original development seeds,
three existing roots per seed, two saved-prefix replays/root, maximum prefix 250
steps; per-seed watchdog 300 seconds. No candidate future rollout or training.
The original source report, P4a labels, policy checkpoint and SUMO network are
hash-pinned. Policy loading only reproduces the P3 initialization environment;
actions come from the recorded prefixes, not fresh DDPG inference.

Outputs: artifacts/p4/<run-id>/report.json, config, worker logs, per-seed reports
and r0.json..r2.json input/history packs with links/hashes to existing labels.
Inspect history_actor_gate and all nine root checks. formal_training_ready
remains false even if the pilot succeeds. Unknown geometry or changed replay
must fail explicitly; do not discard a seed/root to obtain a pass.

Completed evidence: artifacts/p4/p4_history_v1_01/report.json; clean code
88b4da5; 59.67 seconds; history_actor_gate=true, 9 roots, 45 candidate labels
joined, 129 tests passed. All roots have 11 actual history frames; 12 neighbors
selected per root and 3–12 omitted neighbors summarized. formal_training_ready
is still false. See reports/p04b_history_actor_acceptance_20260927.md.
