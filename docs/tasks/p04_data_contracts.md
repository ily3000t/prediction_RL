# P4a task: explicit labels and episode grouping

Goal: turn audited P3 branches into validated development label fixtures while
preventing episode leakage, duplicate repeat samples and false future negatives.

Precondition: P3 v3 accepted at main f69a032. Work on codex/p04-data-contracts.
Allowed: new independent dataset module, read-only converter, tests and docs.
Forbidden: modifying upstream environment/reward, old project, P3 outputs,
creating formal datasets, training, or claiming full P4 completion.

Acceptance: unit tests; convert all 3 episodes / 9 roots / 45 distinct candidates;
keep nonresponsive/colliding branches; explicit development split; future/event
masks; source/output hashes; training_ready=false with concrete missing inputs.

Artifacts: artifacts/p4/<run-id>/. Commit code/config/tests before clean audit;
commit small acceptance summary separately. No push while license unresolved.

Pause: corrupted source, discontinuous trace, unknown termination, missing
candidate family or inconsistent metadata. Never drop bad examples to pass.

Handoff: docs/runbooks/p04_label_audit.md. Next is P4b history/actor adapter, not
formal collection or training. Implementation boundary is in the design document.

Actual acceptance: code 028f813, 109 tests passed; clean offline audit
artifacts/p4/p4_labels_v1_01 passed for all 45 candidates. training_ready remains
false for documented missing history, actor adapter and approved formal splits.
Summary: reports/p04a_label_contract_acceptance_20260927.md. This slice can merge;
full P4 completion is not claimed.
