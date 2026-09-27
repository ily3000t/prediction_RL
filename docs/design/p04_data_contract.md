# P4 data and input contract — development v1

## Scope and implementation boundary

P3 accepted exact prefix replay and development interaction coverage. P4a
implements episode-group split validation, masked future labels and a read-only
conversion audit on those existing traces. It does not implement the full P4
collector, actor adapter, predictor, calibration or ensemble training.

The 3 P3 episodes / 9 roots / 45 distinct candidates are already inspected
development evidence. They must not become validation, calibration or test data.
The converter fixes split=development and training_ready=false. It refuses a
changed source report hash, unknown config keys, dirty source commit, missing
candidate family or branch-repeat disagreement. All input/output hashes are
recorded. A successful label audit is not a training-readiness gate.

## Episode and root identity (implemented)

Episode identity groups environment snapshot hash, simulator seed and episode
index; it deliberately excludes the root time, candidate and reference policy.
All prefixes from an episode stay together across methods. Root identity adds
the audited root signature. Candidate repeats verify determinism; they are not
new examples. Duplicate candidates and roots assigned to multiple episodes fail.

Manifest has explicit development, train, validation, calibration and test lists.
No random row-level split or silent percent-based fallback is permitted. Each
row must match its episode's declared split. Formal IDs/seeds and manifests
remain to be approved before collection; this implementation creates only a
development manifest. Formal multi-version scenarios need one approved grouping
identity rather than using changed code hashes to bypass episode grouping.

## Candidate and trajectory labels (implemented)

The full requested plan is one original 0.2-second jerk probe plus 24 zero-jerk
steps. Requested jerk is distinct from reward-projected jerk, reconstructed
speed command and measured speed. All are retained separately; no action is
clipped or rewritten by the converter. This is a 5-second specified simulator
intervention, not an arbitrary continuous-action safety certificate.

Root ego and all root-neighbor IDs define label rows. No future actor is allowed
to determine input slots. Full raw label arrays are variable in actor count;
they are NOT yet the fixed-capacity model input. Future arrivals are listed for
coverage audit, not retroactively added to the root observation.

Features are [x, y] relative to the FIXED root ego position, scalar speed and
scalar acceleration. These are not coordinates relative to the future ego,
velocity vectors or oriented vehicle-surface gaps. No geometry safety labels
are inferred from these point positions.

Masks:

- future_mask[actor,time]: actor observed on a nonterminal future frame.
- transition_mask[time]: actual upstream transition recorded, including terminal.
- Event value/mask: upstream_collision or upstream_arrival observed within the
  requested horizon. A positive event is known when observed. An absent event
  is known negative only when the complete physical time horizon is observed;
  earlier competing termination leaves it censored (mask=false).
- All terminal-frame trajectories are conservatively masked because original
  termination can remove ego or other actors. Terminal events remain recorded.
- Missing actors, post-terminal time and unobserved actions are not zero-motion
  labels. Zero trajectory padding is usable only with mask=false; audit fields
  use null for missing measurements.
- An incomplete nonterminal rollout is an error, not legitimate censoring.
- Original timeout's extra tick is recorded but never assigned to a regular
  trajectory-grid cell. No upstream termination semantics are changed.

These events retain upstream definitions. They are not independent Risk scores,
calibrated viability, proof of safety or model ranking labels. Terminal censoring
can be informative: downstream learning/evaluation must report event-mask coverage
and not interpret missing labels as negatives. A competing-risk formulation or
new ranking target requires a separate documented design.

## History and actor input design (next implementation, not yet available)

Working pilot design: 11 frames at 0.2-second spacing spanning 2.0 seconds,
including the root. Early history and actor absence carry explicit masks, never
duplicated/synthetic observations. Ego has a separate stream; K=12 refers to
neighbor physical slots, NOT ego. Extra summary tokens do not count as vehicles.

Actor membership and ordering are determined using root/past state only and
shared by all candidate branches of that root. Preserve mandatory target-lane
front/rear and critical conflict actors; deduplicate IDs. Stable vehicle-ID
tie-breaking applies. Road-relative gap/TTC/conflict ordering needs an audited
upstream lane/route geometry adapter; do not equate global x with path progress.
Mandatory overflow must be explicit, never silently truncate a required actor.

Omitted-actor summaries belong in a dedicated prediction adapter. Proposed
groups are ramp, main approach/conflict and main downstream, with an explicit
unknown-lane policy. Use observed count, gap/closing/arrival summaries only when
their geometry definitions are validated. Empty groups have false token masks;
padding cannot attend as a real actor. Current P3 snapshots lack vehicle extents
and continuous history, so this layer is deliberately not faked from P3 labels.

Ordinary B2 and conditional B3 share episode splits, actor selection, masks,
history, prediction horizon, three-member ensemble and policy summary layout.
B2 neighbor prediction must not access candidate plans; B3 may. Future labels
and observed execution corrections cannot enter online input. GRU/one interaction
attention layer and candidate encoder dimensions will be fixed in the model
implementation review, not selected from formal test outcomes.

## Next P4 slices and acceptance

1. P4a: label/split contract and existing-trace audit (this slice).
2. P4b: exact replay history collection and deterministic geometry/actor adapter;
   bounded pilot only, masks/permutation/overflow/leakage tests.
3. P4c: ordinary/conditional predictor and three-member save/load/batching tests;
   no old checkpoint reuse or imports from E:/WCDT_ACCVP.
4. Review formal data/seed manifest and training commands with user. P5 remains
   required before expensive DDPG experiments.

No formal training CLI is provided by this P4a slice. No upstream reward, control,
shared WcDT class, old-project file or P3 artifact is modified.
