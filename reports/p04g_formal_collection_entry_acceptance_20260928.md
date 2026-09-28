# P4g user-run collection entry acceptance — 2026-09-28

Implementation commit: `bd9e3c174e9a4f5d7029e952a300464520af8713`.
Branch: `codex/p04-formal-collection`. All measured audit/preparation calls began
with clean committed source. Old E:/WCDT_ACCVP and upstream source were not modified.

## Implemented and verified

- Shared fresh-episode collector constructs history, masked actor inputs, five
  candidate label packs and future geometry without historical labels as inputs.
- Existing P4f audit now observes that shared engine and checks new data against
  accepted P3/P4 evidence. Formal workers do not load historical labels.
- Independent full protocol file; a test verifies all experimental values match
  the prior draft after removing release identity/status/execution controls.
- Prepare validates current shared-engine acceptance and makes no formal job run.
- Run requires the exact request hash and explicit allowed splits; worker also
  checks parent authorization, active lock and job membership. Test cannot run.
- Completed artifacts are immutable and fully checked on reuse. Failures stop,
  remain visible and are not silently retried. Missing roots are not replaced.
- Aggregation separates source/data validity from model effect: even all-collision
  or zero-reached-root outcomes are retained and require coverage review.

Full unit suite: **273 passed in 6.17 s** (31 new tests relative to P4f).
An initial test-only Windows slash mismatch was corrected; the full suite then
passed. Git staged whitespace check and explicit file/sensitive-string review
passed. No weights, logs or raw results were staged.

## Actual bounded shared-engine audit

```powershell
python tools/audit_response_collection.py --config configs/development/p04_response_collection_v1.json --run-id p4_collect_v2_01
```

| Check | Result |
| --- | --- |
| Development simulator seeds | 0, 1, 100 only |
| Roots / unique candidates / executions including repeat | 9 / 45 / 90 |
| Original traces, rewards, observations, histories and labels | Exact |
| Forward/reverse future geometry | Exact |
| Selected-neighbor valid actor-time cells | 8,988, unchanged |
| Complete supervision tensor fingerprint | Exact P4d fingerprint |
| Collector gate | true |
| Measured total time | 263.587 s, engineering audit only |

Supervision fingerprint:
`904c3ab7a40a2e3a22674ef373577ac8fda7fc3fdd62817a9ffe2cf942eacae2`.

Actual `--resume`: complete, all three episodes reused, executed_episodes=[],
elapsed 1.778 s; no simulator rollout. Original P4f v1 artifacts remain untouched.

Audit artifacts (relative to project):

- `artifacts/p4/p4_collect_v2_01/invocations/398fb383670c/report.json`
  SHA256 `b815a5d0f38c1844996ef1beefdcfb6e2fcc5d58677487e96270e17a8629874c`.
- `artifacts/p4/p4_collect_v2_01/invocations/edd6597b8a73/report.json` (resume)
  SHA256 `490cd43870af7b8cb891eb633e490037e14cc4c142802f504b55f93b7d293ba9`.

## Actual formal request preparation — no execution

```powershell
python tools/collect_response_dataset.py prepare --config configs/formal/p04_response_dataset_v1.json --collector-report artifacts/p4/p4_collect_v2_01/invocations/398fb383670c/report.json --run-id response_v1
```

Result: `prepared_not_executed`, `collection_started=false`,
`formal_training_ready=false`. Train/validation/calibration budget is 256/64/64 =
384 episodes. Test budget 128 remains locked. A separate read-only load/selection
check accepted the request and counted 384 selectable jobs, **0 test jobs and
0 formal episode directories**. No formal command was executed.

- Request: `artifacts/data/response_v1/request.json`.
- Canonical confirmation hash:
  `ed1a90bfeff0c3ab01fa4841ca7a8fa1588e77799aaf1087175916e36210f9a6`.
- Request file byte SHA256:
  `816da45299f0297edbbd5b5e6aa8682e1d41c2bf1878f3c50d0c13c850a8bc28`.
- `preparation.json` SHA256:
  `8418dc824d0d839c97d1043d6fe234cd3d72dbe54905d54c1a5739b4484d26ff`.

The canonical hash confirms parsed request content; it is intentionally distinct
from the pretty JSON byte hash. Request includes full resolved protocol/manifest,
upstream/reference checkpoint/evidence/source bindings, Python/OS/package/SUMO
versions. Each future invocation records its actual commit, command, selected
splits, jobs, failures and reused artifacts. The published plan's disabled jobs
stay immutable; manual invocation authorization is a separate receipt.

## Remaining limitations

This is engineering acceptance on inspected development scenarios plus mocked
formal orchestration, not an observed formal dataset or a prediction-effect gate.
Fresh formal seeds may expose geometry/replay/runtime failures; the 300-second
worker guard stops/preserves them rather than substituting data. A complete
episode may have no usable supervision; counts require review before training.
Partial episode retry/repair and test release are deliberately not implemented.
Full predictor training/calibration, P5 ranking validation and DDPG feature
integration remain pending. Do not start model training from smoke weights.

No remote push, source publication or runnable-model milestone tag. Follow
`docs/runbooks/p04_formal_collection.md` for the exact user-run command and limits.
