# P2 environment acceptance — 2026-09-26

Status: PASS for the bounded environment contract audit. This is not a
performance experiment, model training or proof of safe driving.

## Source and changes

Parent milestone: 3f0cf22 (P1 local compatibility).
Implementation commit: e3dc21f.
Audit commit: 1e45e3c2e4bf260249ff88fd24423e2fc6f9d77e.
Feature branch: codex/p02-environment-contract.
The complete RL-MPC-LaneMerging-master tree is unchanged from P1 in this stage.
No old E:/WCDT_ACCVP files were edited, staged or committed.

New adapter: src/prediction_rl/envs/upstream.py.
New integration auditor: tools/audit_environment.py.
No DDPG training hook was changed and no new package was installed.

## Verified results

- 41 unit/regression tests passed, including P1 tests.
- Eight real-SUMO checks passed; eight isolated workers, twelve bounded episodes.
- First development audit: artifacts/p2/p2_contract_01/report.json,
  95.92 seconds; dirty source snapshot retained.
- Clean confirmation: artifacts/p2/p2_clean_01/report.json,
  90.24 seconds, working_tree_dirty=false at commit 1e45e3c.
- git diff --check passed; upstream source/config diff is empty.

| Case | Seed | Reference steps | Adapter steps | Outcome |
| --- | --- | --- | --- | --- |
| Constant +5, continuous two-episode stream | 0 | 45, 35 | 45, 35 | Exact trace equality |
| Constant -5, time limit | 0 | 500 | 500 | Exact trace equality |
| Repeating -5,-2.5,0,2.5,5 | 100 | 165 | 165 | Exact trace equality |
| Independent process repeat of stream | 0 | adapter 45, 35 | repeat 45, 35 | Exact equality |
| Changed-seed stream | 1 | not a performance comparator | 46, 38 | Initial traffic differs |

Equality includes observations at the original ALL policy dtype, reward,
done, original info, full per-vehicle position/speed/acceleration/lane snapshots
and the Python traffic scheduling delay. No numerical equality tolerance was
used on these traces. These cases do not constitute independent statistical
samples supporting a method-effect claim.

All eight checks: streaming equality, mixed-action equality, timeout equality,
same-seed repeatability, seed differentiation, 500-step cap, continuing traffic
across resets, and the original extra terminal simulation step.

## Important findings and decisions

1. Preserve upstream continuous traffic across training resets. Do not reseed
   each episode or reset control.delay there. Independent scenes use separate
   run initialization/processes.
2. Preserve the original timeout branch: at 500 policy steps it removes ego
   and advances SUMO again. Terminal simulation increment is 0.4 seconds rather
   than a normal 0.2-second decision period.
3. In that terminal transition the reconstructed speed command is 0.0, while
   post-step measured speed/jerk are null because ego was removed. Null must
   not be replaced by zero or a stale value.
4. Requests, helper-reconstructed speed commands and actual SUMO measurements
   are distinct fields. Command reconstruction is explicitly not wire-level
   instrumentation. No action intervention or shield was added.
5. Unknown config fields, invalid actions/observations, mid-episode reset,
   concurrent sessions and conflicting imported upstream modules are rejected.
   Runtime faults invalidate the adapter rather than fabricating a transition.

## Audit environment and fixed inputs

Existing pytorch Python 3.10.16, torch 2.5.1, NumPy 2.2.6, Gym 0.26.1;
SUMO 1.22.0 with TraCI 1.25.0. The P1 Cython extension remains enabled.
Reference config: RL-MPC-LaneMerging-master/configs/train_default_1.json.
SHA256: 537423fe0bb27ea19bff87caaa4ab2388789bd2cac95c1cce3d9832beb142d8a.

The run seed override and SYSTEM=Windows are explicit in each resolved config.
No optimizer, predictor or policy checkpoint is used in P2. Full source,
configuration and trace hashes are stored with reports. Outputs remain ignored
by Git; only this small summary and the tools/tests/docs are committed.

## Remaining boundaries and next stage

The integration cases exercise arrival and timeout, not an intentionally
generated collision. Collision reason precedence is unit-tested, but the
upstream detector's physical validity is not certified. Cross-version SUMO
equivalence, broad scenario coverage and checkpoint-training convergence remain
outside this stage.

Proceed next to P3: deterministic snapshot/restore, repeated identical branches,
branch-order independence, and separate ego-motion versus neighbor-response
effects under jerk interventions. Save both SUMO state and Python/RNG state.
Do not collect a formal dataset or train a predictor until P3 acceptance.

No push performed. User reported uploading the prior Git state; that does not
by itself establish the upstream license. Existing publication restrictions
remain in effect until permission is documented.
