# P5b shared-ego candidate proximity ranking

Run from `E:\Prediction_RL` in the existing `pytorch` environment. This reuses
the frozen ordinary/conditional three-member ensembles. No retraining, SUMO
execution, calibration change, DDPG or test release is performed.

Configuration: `configs/development/p05_candidate_ranking_v1.json`.
Definitions/limitations: `docs/tasks/p05_candidate_ranking.md`.

## Maintainer preparation (clean committed source)

```powershell
python tools/diagnose_candidate_ranking.py audit --config configs/development/p05_candidate_ranking_v1.json --run-id p5b_ranking_v1_01
python tools/diagnose_candidate_ranking.py prepare --config configs/development/p05_candidate_ranking_v1.json --audit artifacts/p5/p5b_ranking_v1_01/report.json --run-id predictor_v1_rank1
```

The audit uses only the existing 9 development roots. Preparation verifies and
hashes inputs without evaluating the full validation set. Both refuse output
directory reuse; pick a new short run ID only after explaining any failed attempt.

## User-run validation

Use the exact confirmation hash printed by preparation (the accepted handoff
report records it). Do not paste this placeholder literally:

```powershell
python tools/diagnose_candidate_ranking.py run --request artifacts/ranking/predictor_v1_rank1/request.json --confirm-request-hash <printed-request-hash>
```

The request binds 177 reached validation roots from the original 64 episodes,
with all five candidates and three prediction methods. Empty supports remain
counted. It evaluates frozen model inference, not training; elapsed runtime is
reported, not a deadline that changes predictions.

Results: `artifacts/ranking/predictor_v1_rank1/report.json` and `roots.json`.
Inspect `summary.ego_parity_gate` first, then both `support_views`, method metrics,
paired differences, metric denominators, near-tie spreads, empty/truncated roots,
omitted/new actor counts and termination counts in root rows. Lower error/regret
is better; higher ordering/tie-hit rate is better. Do not select a favorable view
after observing the results or treat proximity ranking as driving success.

Negative results finish with `status=complete`; `ddpg_ready=false` remains
mandatory. If interrupted, preserve the failed/started run and inspect it before
preparing a separately named request. Never overwrite artifacts or replace seeds.

P5a point-error and calibration evidence stays at
`artifacts/offline/predictor_v1_diag2/`; no prior output is overwritten.
