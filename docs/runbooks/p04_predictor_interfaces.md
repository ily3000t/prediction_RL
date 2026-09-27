# P4c bounded offline interface audit

Use the pytorch environment from E:/Prediction_RL, on a clean committed tree.
No SUMO episode, optimizer update or policy evaluation is started.

```powershell
python tools/audit_predictor_interfaces.py --config configs/development/p04_predictor_interface_v1.json --run-id p4_interface_v1_01
```

This exact run ID must not be reused if its artifact directory already exists.
Use a new short ID for an explicitly recorded repeat; never delete failure
reports to make a rerun look like the first acceptance.

The audit requires the pinned P4b report and all nine root files. It saves
resolved_config.json, report.json and ordinary.pt/conditional.pt under
artifacts/p4/<run-id>. The weights are RANDOM INITIALIZATION for interface checks,
not trained B2/B3 models. Do not load them into DDPG or describe their differences
as evidence of action-conditioned prediction accuracy.

```powershell
$taskTemp = 'E:\Prediction_RL\artifacts\pytest_p4c_' + [guid]::NewGuid().ToString('N')
python -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
git diff --check
```

Read interface_gate, both models' batch/serial errors and exact checkpoint
roundtrip fields, source_evidence_unchanged and formal_training_ready=false.
An accepted interface is only permission to implement P4d target alignment and
masked training interfaces. Full datasets/ensemble training are still user-run
after configuration approval; P5 must precede expensive DDPG experiments.
