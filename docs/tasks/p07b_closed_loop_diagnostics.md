# P7b read-only closed-loop diagnosis

Goal: explain P7 collision, non-completion and online input distribution without
changing any model, reward, action, seed, policy checkpoint or predictor.
Prerequisite is the complete pinned P7 request/aggregate and all parent receipts.

All 12 policies, scenes 200/210/219 (first/middle-index n//2/last of the original
roster), 36 deterministic fresh-worker replays; no outcome-based scene selection.
CPU one Torch thread, two isolated workers, at most 18,000 transitions. No training
or exploratory action noise is added. These remain development scenes, not new
confirmation evidence. Long diagnostics are user-run.

A sidecar calls the original feature provider exactly once and repeats only pure
observed-state encoding. B0/B1 have a separate shadow actor/history audit that
never reaches the policy. Record 20/169 observation plus exact ALL time channel
and mask, actor/history/summary tensors and masks, IDs, roles, omissions, root
frame, lane position, speed/acceleration, requested jerk, reconstructed command
speed, post-step measurements, projected reward jerk and terminal reason.
Terminal predictor blocks stay zero; no terminal inference or stale audit input.
Observation i is consumed by action i and step i produces observation i+1.

Build a descriptive reference from the existing frozen eligible train-root
roster, reading history.json only. Future label files are not parsed. This is an
in-sample input reference, not predictor accuracy validation; the sealed test
stays closed. Physical statistics respect history/summary masks; prediction
channels respect root role/pool presence. Compare global ranges and matched
road-lane/full-vs-partial-history cohorts. Missing cohorts or valid training
support stay unavailable, not zero or a fabricated OOD judgment.
Min/max and p01/p99 comparisons are descriptive, not calibration or safety gates.

Low-speed <=0.1 m/s and command-gap >0.001 m/s are fixed descriptive thresholds.
Commands are reconstructed by the existing helper, NOT intercepted wire commands.
Low-speed duration is a 0.2-second sample-grid summary, not a new stop/failure rule.
History cells overlap: ranges and sample fractions do not imply independent data.

Exact action/reward/outcome parity against each same-checkpoint P7 episode is an
engineering requirement. Model/replay/optimizer remain unchanged. A negative
driving outcome is valid; mismatched replay or malformed artifacts are invalid.
No seed deletion, threshold loosening, failure overwrite or automatic retry.

Only new diagnostics code/config/tests/docs may change. Old P7 files are checked
against Git provenance AND current normalized fingerprints; newly added sidecars
do not justify changing old executable files. New P7b requests bind all current
source/runtime/reference hashes. Do not edit or replace old P7 hashes. The old
P7 executor intentionally requires its original full source inventory; to replay
it as an original run, use its recorded source checkout. P7b instead verifies its
complete historical parent explicitly, then binds the new recording source.

Bounded audit: same four arms, training seed 2 / scene 200, maximum 2,000 steps.
It deliberately covers the already observed long/failed paths for instrumentation,
not unbiased effect evidence. Full diagnosis retains all training seeds.
Deliver tests + bounded audit + immutable reference/request + docs + atomic commits.
Do not advance to formal P8 automatically.
