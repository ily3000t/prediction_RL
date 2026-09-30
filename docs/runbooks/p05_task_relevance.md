# P5c task-relevance diagnostic

Run from `E:\Prediction_RL` with the existing `pytorch` environment. No SUMO,
training, actor reselection, calibration or DDPG is executed. Models stay frozen.

Config: `configs/development/p05_task_relevance_v1.json`.
Definitions: `docs/tasks/p05_task_relevance.md`.

## Maintainer audit and prepare

```powershell
python tools/diagnose_task_relevance.py audit --config configs/development/p05_task_relevance_v1.json --run-id p5c_task_v1_01
python tools/diagnose_task_relevance.py prepare --config configs/development/p05_task_relevance_v1.json --audit artifacts/p5/p5c_task_v1_01/report.json --run-id predictor_v1_task1
```

Audit uses 9 existing development roots. Prepare checks training receipt inventories
and raw traces/labels, without running models. Disk verification can take longer
than model inference. Neither operation changes or overwrites original data.

## User-run training-data diagnostic

Use the exact request hash printed by prepare and recorded in the handoff. Do not
paste the placeholder literally:

```powershell
python tools/diagnose_task_relevance.py run --request artifacts/task_diagnostics/predictor_v1_task1/request.json --confirm-request-hash <printed-request-hash>
```

Outputs: `artifacts/task_diagnostics/predictor_v1_task1/report.json` and `roots.json`.
The train split covers 256 source episodes / 713 reached roots, retaining empty
supervision and terminal events. Missing reference roots are not resampled. Root
and episode denominators remain visible. These are in-sample diagnostics only.

Inspect:

- `summary.methods`: distance ordering agreement with recorded return/collision/
  arrival, counts of ties, censoring and no geometry, root/episode averages;
- `tasks`: original rewards, components, events, horizon and terminal reasons;
- `coverage`: all-vehicle nearest bumper / same-path-gap identities, distinguishing
  selected, omitted-root and newly arriving actors, not collision responsibility;
- `critical_cells`: actor/time/lane and errors at the truth-critical location,
  including conditional-response differences rather than only global ADE;
- `wrong_order_pairs`: candidate pair errors linked to task outcomes;
- `reward_unverified_steps`: time-limit audit gaps, never imputed.

No result automatically releases policy features or DDPG. Do not rerun to select
a favorable result. A failure/started output stays preserved; inspect before any
separately named request. Old P5a/P5b evidence and test split remain untouched.
