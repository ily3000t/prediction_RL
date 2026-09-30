# Four-arm DDPG integration smoke

Use a clean tree in the existing pytorch environment:

```powershell
cd E:\Prediction_RL
python tools/smoke_feature_ddpg.py --run-id p6b_ddpg_01
```

Use a new short run ID for a repetition. The script binds the accepted P6a report,
strict frozen ordinary/conditional predictors, code/dependency hashes and the
standalone `configs/development/p06_ddpg_smoke_v1.json` configuration. Results are
immutable under `artifacts/p6/<run_id>`; eight-stage old project pipelines are not
involved. Four fresh SUMO processes execute sequentially, at most2,252 steps total.

Outputs: request/report, per-arm resolved configuration, episode traces, original
TensorBoard loss/return logs, immutable evaluation-only policy/critic state dicts.
The shortened warmup/minibatch are diagnostic overrides, not formal defaults.
The CPU device is diagnostic; no claim about CPU/GPU training speed is made.

Inspect `engineering_gate`, Adam updates, actor/critic changes, frozen predictor
check, matched expanded initialization and checkpoint roundtrip. Do not rank
methods by these tiny training episodes. No formal training/evaluation command is
released yet; P7 must separately freeze budget/seeds and development comparisons.
