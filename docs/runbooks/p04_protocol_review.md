# P4e review-only plan preparation

From E:/Prediction_RL in pytorch, with a clean committed tree:

```powershell
python tools/prepare_predictor_protocol.py --config configs/formal/p04_response_dataset_v1_draft.json --run-id p4_plan_v1_01
```

No formal experiment is launched. The tool verifies prerequisite reports,
upstream source and reference weight hashes without loading the policy. It reads
environment versions, including SUMO --version only. There is no run, approve,
collect or train subcommand.

Outputs under artifacts/p4/<run-id>: planning_report.json, plan.json,
resolved_protocol.json, split_manifest.json, resolved_upstream_base_settings.json.
Existing output directories are refused; use a new short ID for a recorded repeat.

Require status=review_only, plan_validation_pass=true, execution_authorized=false,
formal_training_ready=false. The blockers are intentional. Jobs are proposed,
not executed. Test jobs remain locked until predictors/calibration freeze.

Review docs/design/p04_collection_training_proposal.md and confirm/change the
episode budget, reference coverage, epoch/selection budget and calibration scope.
Do not pass the generated upstream settings to main.py: TASK is source metadata.

After review, implement the resumable collector and verify future metadata reads
on existing DEVELOPMENT seeds first. Formal collection/training remain user-run
after acceptance. Production training, calibration, P5 geometry/ranking and DDPG
feature interfaces are still pending.
