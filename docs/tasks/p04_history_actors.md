# P4b task

Objective: observed history and deterministic dedicated actor/summary adapter,
with exact P3 replay parity and stable input shared across same-root candidates.
Preconditions: P3 accepted; P4a masks/splits accepted at main 70692fc.
Branch: codex/p04-history-actors.

Allowed: independent geometry/adapter module, bounded history extension, tests,
hash-pinned pilot and docs. Prohibited: upstream/old-project edits, new rewards,
formal data/training, candidate outcome filtering, native snapshot promotion.

Acceptance: masks/ordering/overflow/unknown-lane unit tests; 9/9 original roots
match P3 signature and prefix trace hash; histories and encodings repeat exactly;
all 45 existing labels join; no future data in adapter API. Keep formal readiness
false and report omitted actor/group coverage.

Artifact path: artifacts/p4/<run-id>/. Source/config/tests/docs committed before
pilot; small result report separately. Preserve failed pilot evidence. Merge
only this verified slice; do not mark all P4 complete or push unresolved upstream.

Stop on geometry/parity/input failure. Handoff: docs/runbooks/p04_history_actors.md.

Actual result: implementation 88b4da5; 129 tests passed; clean bounded pilot
p4_history_v1_01 completed in 59.67 seconds. All nine roots match P3, all repeated
histories/encodings match and all 45 candidate labels join. Independent checks
verified omitted counts over 99 frames and selected actor IDs in every label
family. Report: reports/p04b_history_actor_acceptance_20260927.md.
This slice can merge; next is P4c model interfaces, not formal training.
