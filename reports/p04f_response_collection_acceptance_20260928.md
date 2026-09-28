# P4f response collector acceptance — 2026-09-28

Scope: bounded development-only engineering verification. No formal collection,
predictor training, DDPG training or new test seeds. No change to upstream or the
old E:/WCDT_ACCVP project. The P4e formal protocol remains an unapproved draft.

Implementation commit: d40dbf38500837dcd815b91da3cad81262179dce
on codex/p04-response-collector. Both measured invocations began with a clean tree.

## Actual verification

Full tests: **242 passed in 5.61 s**, including 32 new collector/storage tests.
Git staged whitespace check passed. Staged review contained only seven intended
source/config/test/task files; no weights, raw results or secrets were staged.

Command:

```powershell
python tools/audit_response_collection.py --config configs/development/p04_response_collection_v1.json --run-id p4_collect_v1_01
```

| Check | Observed result |
| --- | --- |
| Seeds | 0, 1, 100; existing development episodes only |
| Root/candidate/branch executions | 9 / 45 unique / 90 including repeats |
| Discovery and reference-policy roots | Exact accepted P3 roots |
| Original observation, reward, execution/termination traces | Exact across all candidates |
| History/actor inputs and labels | Exact accepted P4b/P4a artifacts |
| Future metadata, forward versus reverse repeat | Exact |
| Observed future frames | 777 per unique-candidate set; no padding as observations |
| Terminal unique branches | 28; remaining 17 reach the horizon |
| Selected-neighbor valid actor-time cells | 8,988; unchanged |
| Aggregate supervised batch fingerprint | Exact accepted P4d fingerprint |
| Collector gate | true |
| Total elapsed | 295.464 s (~4.92 min), not a deployment benchmark |

Measured in-worker collection durations: seed 0 = 91.08 s, seed 1 = 77.06 s,
seed 100 = 120.99 s. These include repeat/replay audit work and must not be
interpreted as predictor inference latency or a formal collection time guarantee.

The observed terminal outcomes were retained, not filtered to make acceptance
pass. Labels remain censored on terminal frames; the new read-only geometry
sidecar does not relax masking or feed future information into model inputs.

## Verified resume

```powershell
python tools/audit_response_collection.py --config configs/development/p04_response_collection_v1.json --run-id p4_collect_v1_01 --resume
```

Result: complete, collector_gate=true, executed_episodes=[],
reused_episodes=[0,1,100], measured aggregate/resume time 1.694 s. No worker/SUMO
simulation was launched by the resume branch (SUMO --version is still queried).
The initial report and completed episode artifacts were not overwritten.

Unit tests also cover modified/missing/extra artifact rejection, changed request
binding, partial/failed episode refusal, stale/contended locks, atomic no-clobber
writes, explicit missing roots, terminal geometry, source CRLF normalization and
parent failed-run receipts. They do not claim recovery from every OS/power failure.

## Artifact provenance

All raw evidence stays under artifacts/p4/p4_collect_v1_01 (Git ignored).

- request.json SHA256:
  8a3b7fd0ab99872aed6fcf856810e7397fa224d29c1647c89717c1d64e52c409
- Canonical request hash (distinct from pretty JSON byte hash):
  1c39350ad6ae23b72e17d476104a1919e79195d945948f97a222532dbd470544
- invocations/4671b46793ad/report.json SHA256:
  fae7eb7faa612048439a0e0ad02fd20f268d7db77102a98f8c5e35b2bba167ee
- invocations/1b1c09708cc0/report.json SHA256 (verified resume):
  3aa641fdce078271583a6aefd6a41d252630f34682176cedc367aa017bc52f3e
- Supervision fingerprint:
  904c3ab7a40a2e3a22674ef373577ac8fda7fc3fdd62817a9ffe2cf942eacae2

Environment: Windows-10-10.0.26200-SP0, Python 3.10.16 in
E:/Programs/EnvAnaconda3/envs/pytorch, PyTorch 2.5.1, SUMO 1.22.0.
Full dependency/SUMO executable metadata and all input hashes are in request.json.
Original resolved settings are stored independently for each seed.

## Boundaries / next gate

This accepts the collection mechanism and development-only verified reuse, NOT
the 512-episode proposal, a production executor, trained prediction quality or
closed-loop method effectiveness. The audit deliberately requires historical
accepted evidence; it cannot collect new formal seeds by changing a CLI argument.

Next work is an independently versioned user-run formal collection executor after
sampling/protocol review, followed by full ordinary/conditional three-member
training/calibration and P5 offline acceptance. Test collection stays locked until
predictors/calibration freeze. No long experiment was started, no source/license
publication was attempted, and no runnable-model milestone tag was created.

Runbook: docs/runbooks/p04_response_collector.md.
