# P4d two-step supervision/resume audit

Run from E:/Prediction_RL in pytorch on a clean committed tree:

```powershell
python tools/audit_masked_training.py --config configs/development/p04_masked_training_v1.json --run-id p4_training_v1_01
```

The command is an engineering smoke, NOT formal ensemble training. It reads all
nine pinned P4b roots, their 45 P4a candidate labels and the existing development
split manifest. No new SUMO rollout or DDPG process is started. Both B2/B3 run
two updates per member trajectory; a restored copy replays the second update.

Outputs under artifacts/p4/<run-id>: resolved_config.json, report.json and
ordinary_s1.pt, ordinary_s2.pt, conditional_s1.pt, conditional_s2.pt. These are
development-only training-state checkpoints, not deployable models. Never
overwrite an existing run; use a new short ID for a recorded repeat.

Check training_interface_gate, all three exact resume fields per model,
supervision counts, source_evidence_unchanged and formal_training_ready=false.
Do not interpret two-step training losses as prediction quality or select
between B2/B3 using this engineering audit.

Regression tests:

```powershell
$taskTemp = 'E:\Prediction_RL\artifacts\pytest_p4d_' + [guid]::NewGuid().ToString('N')
python -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
git diff --check
```

After acceptance, the next task is P4e formal data/training protocol review and
collection interfaces. A production trainer, validation checkpoint selection,
calibration and P5 accuracy/ranking gates are not supplied by this smoke command.
