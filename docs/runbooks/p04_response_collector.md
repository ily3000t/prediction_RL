# P4f development response collection

This is a bounded engineering audit, not the formal dataset executor. The CLI
rejects all seeds except 0, 1 and 100 and cannot execute the P4e formal draft.
It uses pinned accepted P3/P4 artifacts to check exact parity, so it is not a
standalone fresh-machine data-generation command.

From E:/Prediction_RL in pytorch, with clean committed source:

```powershell
python tools/audit_response_collection.py --config configs/development/p04_response_collection_v1.json --run-id p4_collect_v1_02
```

Use an unused short run ID for a deliberately recorded repeat. Each seed has
three fixed geometric root targets, five jerk probes and two branch executions
in opposite candidate orders. A candidate acts for one original control tick,
then zero jerk for the remainder of the 25-step horizon. Reference root discovery
uses the pinned author DDPG and is independently repeated. Root histories are
also replayed twice. No policy/predictor optimizer or ST intervention is run.

Accepted run: artifacts/p4/p4_collect_v1_01. Verify/reuse it without more SUMO:

```powershell
python tools/audit_response_collection.py --config configs/development/p04_response_collection_v1.json --run-id p4_collect_v1_01 --resume
```

The request binds resolved collection configuration, split manifest, environment
versions, original source/checkpoint/evidence SHA256 and all project Python source
text. Python source hashes normalize CRLF/LF (matching Python's source handling);
config hashes use canonical parsed JSON. Raw artifacts/weights use byte hashes.
Any actual source/config/environment drift requires diagnosis, not cache reuse.
Each invocation records its actual Git SHA and command separately, so a docs-only
commit does not invalidate compatible data.

Layout (all ignored by Git):

- request.json: immutable resolved request, no formal authorization.
- e000, e001, e002: seeds 0, 1, 100, respectively.
- eNNN/settings.json, discovery.json: original resolved upstream settings and roots.
- eNNN/r0..r2/history.json and labels.json: separate observed inputs and future labels.
- eNNN/rN/c0..c4.json and c0..c4_repeat.json: original traces and future geometry.
- eNNN/rN/accounting.json: per-root counts/parity results.
- eNNN/complete.json: request-bound receipt and full artifact inventory.
- invocations/<id>/report.json: immutable success/failure/exec/reuse report.

Future metadata contains lane ID, lane position, length and width for actors
actually present after each observed transition. It stays OUTSIDE history/model
observations. Terminal geometry is retained for audit but trajectory_usable=false;
no unobserved post-terminal frames are fabricated. Existing P4a label censoring
and P4d selected-neighbor masks are unchanged. The compatibility label pack has
no embedded history; history.json is the separately joined observed input.

Require status=complete and collector_gate=true in the latest invocation report.
The accepted audit has 9 roots, 45 unique candidates, 90 branch executions and
the exact earlier P4d tensor hash. A verified resume has executed_episodes=[] and
reused_episodes=[0,1,100]. Repeats are not additional independent samples.

Failure behavior:

- A missing predeclared root is recorded without replacement. Because this audit
  compares already accepted development coverage, any changed coverage fails it.
- NaN/geometry/parity errors stop the run and preserve partial files/failure data.
- Completed episodes may be reused only after the full inventory passes checks.
- Failed/incomplete episodes are not silently retried; preserve them for diagnosis.
  A reviewed explicit repeat uses a new run ID, never an overwrite or seed change.
- A process crash can leave writer.lock. There is no automatic stale-lock removal.
  First establish that no parent/worker/SUMO process is using the run and preserve
  the failure evidence before any manual recovery. Do not delete an active lock.
- The 300-second per-worker watchdog is an engineering failure cap, NOT a policy
  deadline. It never substitutes stale predictions or changes an observation.

Next: review/freeze the P4e formal sampling/training protocol and implement its
separate user-run executor with split-release controls. Do not simply relax this
audit's seed guard or repurpose its inspected episodes as training/test data.
Formal collection, predictor training/calibration and P5 acceptance are pending.
