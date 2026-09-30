# P5c: task relevance before policy feature design

## Goal / prerequisites

Use existing development and TRAINING data to relate the fixed P5b distance
proxy to original rewards/events, locate critical-cell prediction errors and
inspect unselected/new actor coverage. This is in-sample diagnosis, not new
generalization evidence. Accepted P5a/P5b outputs remain immutable, including
negative ranking results and the empty-parity root. No P5b green gate required.

## Scope and prohibitions

Add a standalone diagnostic/config/tests/runbook. Reuse frozen CV, ordinary and
conditional predictors; no training, new event head, reward, action, seed,
threshold, actor selection, ST or Risk changes. Validation/calibration/test are
not new analysis splits; old validation reports are read solely for provenance.
No online features or DDPG are released by this task. Main-agent execution is
limited to the three existing development episodes / nine roots. The user runs
the complete training-data analysis after request preparation.

## Frozen definitions

1. Read recorded trace rewards directly. Verify Slotted Jerk components against
   original configuration: collision -10, arrival +10; otherwise -.1*dt minus
   .1*measured_jerk²*dt, plus the recorded invalid-action penalty. Preserve actual
   runtime branch logic. If a timeout removed ego and measured jerk is unavailable,
   mark its decomposition unverified; do not infer a fictitious measured jerk.
2. Store undiscounted observed branch returns, length, elapsed time and terminal
   reason. These are fixed-continuation prefixes, NOT learned-policy Q values.
   Return ordering uses pairs where both terminate or both have a full 5-second
   horizon. Other prefix-length comparisons remain explicitly incomparable.
   A completed return can be negative; there is no method-effect success gate.
3. Collision/arrival use original event value/mask separately. Compare only pairs
   with both event labels known and different. Censored means unknown, not false.
   Task ties, missing geometry and proxy ties are counted, never turned into wins.
   A no-trajectory immediate collision retains its observed event and reward.
4. CV/B2/B3 retain the P5b score and common retrospective support. Compare each
   score's pair ordering with return, collision preference and arrival preference.
   Include observed distance scores as a diagnostic reference, not an online
   oracle or new best method. No new scalarization combines these outcomes.
5. Rebuild labels from raw traces and verify receipt inventories, repeated
   branches and future/base geometry agreement. Read all observed nonterminal
   vehicles, including omitted root actors and new arrivals. Record the globally
   closest front-bumper actor and minimum same-path gap, actor role, time and
   lanes. These identify geometric proxies, NOT causal collision partners.
6. Locate the truth minimum-distance actor/time for each candidate on unchanged
   common support. Record each method's neighbor position error and candidate
   response-difference error there. Record erroneous ordered pairs and associated
   candidate IDs/reasons; root records join these to critical actors, times/lanes.
   Future metadata is retrospective attribution only, never a predictor input.
7. Aggregate root-equal and episode-equal task agreement, eligible denominators,
   terminal types, coverage roles and lane-stratified error counts. Pair counts
   are descriptive; roots/candidates within an episode are not independent trials.
   Do not claim causality or statistical significance from this diagnostic.

## Implementation / verification

Strict independent JSON config, short immutable run IDs, canonical tracked JSON
hashes, raw artifact hashes, historical Git source verification, clean source,
recorded environment/command. Verify training receipts without inference during
prepare; inference loads at most 16 root families at a time. Run rejects modified
sources/input receipts, unknown split, missing confirmation and output overwrite.

Tests: reward branches and mismatch, missing timeout jerk, terminal event retention,
censoring/ties, omitted/new/selected actors, localization and observation isolation,
split restrictions, confirmation/overwrite, request preparation without inference,
negative-result completion and strict configuration. Then whole regression suite.
Commit implementation before bounded real-data audit; separately commit acceptance
and handoff evidence. No push while upstream licensing remains unresolved.

## Pause / next decision

Reward or provenance mismatch is an engineering failure: preserve artifacts and
repair the diagnostic, not upstream rewards. Lack of task association or critical
actor coverage is a research finding: do not tune metrics on validation to win.
After user-run diagnosis, review whether a minimal deployable continuous feature
contract is justified; authorizing that contract and small closed-loop exploration
is a separate step. Offline ranking need not win every metric, but failures must
remain visible. Model architecture/budget changes require a targeted explanation.

Commands and artifacts: `docs/runbooks/p05_task_relevance.md`.
