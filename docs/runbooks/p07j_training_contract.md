# P7j training semantics and reset/coverage audit

This is a bounded precheck, not the60k matched experiment. Run from
E:\Prediction_RL in the pytorch environment with clean committed source.

```powershell
python -B tools/audit_training_contract.py audit --run-id p7j_audit_v1
```

The command runs four tiny optimizer jobs (explicit replay-start/minibatch8),
six frozen-policy episodes, and20 initial-state-only resets. Two subprocesses,
one Torch thread each; failure stops further batches. New short run ID required;
no overwrite or retry of a failed directory. It must not run concurrently with
another project SUMO/learning experiment.

Read artifacts/p7j/p7j_audit_v1/aggregate.json and historical_limits.json.
training_transition_semantics_gate and initial_gym20_parity_gate concern only
the bounded verified scope. historical_20k_replay_coverage_gate stays null because
the original20k runs did not save full replay/state-sampling logs. Do not claim
both full prechecks passed or start60k automatically. Review observed protocol
differences and missing historical evidence with the user first.

Original Slotted Jerk collision/arrival reward and done semantics stay intact.
Requested jerk is the replay action; measured jerk need not match it because
speed/acceleration bounds and SUMO are part of the original environment. Masked
terminal Q closes bootstrap for internal deadline AND legacy outer TimeLimit.
Changing truncation semantics would be a new method version, not compatibility.

The frozen continuing episodes measure overall stream/reset sensitivity, not an
isolated causal effect of warmup/density. They do not replace first-episode
Gym20 evaluation or author50 external comparisons. No favorable scene or seed
selection, no weight updates, no formal test access, no automatic budget growth.
