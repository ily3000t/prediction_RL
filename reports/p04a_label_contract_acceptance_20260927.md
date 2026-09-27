# P4a label/split contract acceptance — 2026-09-27

Status: P4a label contract PASS; full P4 incomplete; training_ready=false.
No SUMO episode, formal data collection, predictor training or DDPG training
was launched. Only existing P3 v3 branch JSON was read and converted.

## Provenance

- Implementation commit: 028f8133ff50a0e3fa5d9df3fe80897e104f46ff.
- Input: artifacts/p3/p3_policy_v3_01/report.json, SHA256 pinned by
  configs/development/p04_label_audit_v1.json.
- Output: artifacts/p4/p4_labels_v1_01/report.json.
- Output report SHA256: 7f4e239d40872262189bf527b366764bd818f69c680e8debe252b595bcbd27c2.
- Conversion launched from a clean commit. Source/output file hashes, resolved
  config, Python executable/version, OS, command and original simulator metadata
  are retained. Raw packs are Git-ignored; source P3 files are unchanged.
- Unit/regression tests: 109 passed in 2.41 seconds before the audit.

## Results

| Quantity | Result |
| --- | ---: |
| Episode groups | 3 |
| Roots | 9 |
| Distinct candidate branches | 45 |
| Repeated copies included as new data | 0 |
| Branches ending in upstream arrival | 20 |
| Branches ending in upstream collision | 8 |
| Nonterminal at 25-step horizon | 17 |
| Collision event value usable | 25/45 |
| Arrival event value usable | 37/45 |
| Valid actor-time trajectory cells | 14,297/21,625 |

All 45 branches are kept, including collisions and roots without neighbor
response. All three episodes remain development-only. Candidate counts and
actor-time cells are not independent statistical samples or policy success rates.

Collision labels: 8 observed positives, 17 complete-horizon negatives and
20 masked/censored after early arrival. Arrival labels: 20 observed positives,
17 complete-horizon negatives and 8 masked/censored after early collision.
The latter labels cannot silently become negative examples. This conservative
first version does not extrapolate outcomes after an original terminal event.

Trajectory masks exclude terminal frames, absent root actors and unobserved
tail time. Zero padding is never considered real motion. New future actors are
listed separately and cannot determine root input membership. Labels store
coordinates relative to the fixed root ego origin, not the future ego.

Each source root matches its parent report; forward and reverse branch copies
match; every requested jerk plan and time grid is validated. Source/action/time
errors would fail conversion, not remove examples. Explicit episode manifests
reject cross-split reuse and duplicate candidate rows.

## Scope and limitations

This is a schema/label fixture, not a training dataset. P3 did not persist the
continuous root history required by the encoder. Therefore history=null,
history_available=false and training_ready=false are explicit in every pack.
Other blockers: actor adapter not implemented; formal split manifest not approved.

Raw trajectory labels still cover all root actors. They are variable-sized and
are not falsely presented as the proposed K=12-neighbor model input. The working
input design (ego separate, 11 history frames, root-only actor selection,
mandatory slots and masked omitted summaries) is documented but not implemented.
No predictor, ensemble training, B2/B3 ranking or model-effectiveness claim is made.

Formal data should retain responsive and nonresponsive roots. It needs independent
approved episode groups and coverage checks; these inspected development episodes
cannot be recycled as holdout evidence. Event censoring coverage must remain
visible to training/evaluation, and a ranking target is a separate design step.

## Next step

Proceed to P4b: record continuous history during exact prefix replay, implement
and test the upstream road-geometry adapter and deterministic actor/mask/summary
construction, then perform a bounded pilot. Keep the old project read-only.
After that, implement the ordinary and action-conditioned predictor interfaces
with matched inputs and strict save/load tests. Formal collection/training stay
user-run after configuration approval.

P4a can be merged as an independently tested feature. This does not mark all of
P4 accepted. No push while upstream publication permission is unresolved.
