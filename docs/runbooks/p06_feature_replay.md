# P6a bounded feature integration check

Use a clean source tree and the existing `pytorch` environment:

```powershell
cd E:\Prediction_RL
python tools/audit_prediction_features.py --run-id p6a_features_01
```

Use a NEW short run ID for later repetitions; reports are immutable. The script
binds the accepted P5c report and strict selected ordinary/conditional ensembles,
never trains or opens the test set. It launches isolated sequential SUMO workers:
four arms, each two seed-0 positive-jerk episodes (cap160 each) and one seed-100
mixed-jerk prefix (cap40). Maximum 1,440 simulated steps, no policy evaluation.
The mixed cap ends the diagnostic, not a synthetic terminal transition.
Each worker has a 120-second diagnostic cap; this never modifies observations.

`artifacts/p6/<run_id>/report.json` reports exact original observation/reward/
action/traffic/RNG parity, config/weights/code hashes and child receipts.
Per-worker feature traces are separately retained, including terminal zero blocks.
Timing is descriptive and cannot reject a valid observation. See the frozen
design for all 149 channel names and their semantics.

Passing this check authorizes only an engineering conclusion. Next is P6b:
original continuous DDPG feature wiring with replay-buffer/input-shape guards,
matched B1/B2/B3 architectures and a bounded learning smoke. Formal exploratory
training remains user-run after seed/budget approval. Existing B0 policy weights
cannot be partially loaded into an expanded policy.
