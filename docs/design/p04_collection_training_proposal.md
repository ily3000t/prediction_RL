# P4e collection/training proposal — review required

Status: DRAFT, not approved and not executable. Sample sufficiency and prediction
quality have not been established. The planning CLI does not provide a production
collector/trainer or implement calibration. User review precedes formal execution.

## Frozen baseline and scope

Keep train_default_1, original Slotted Jerk reward, continuous jerk and original
termination. The pinned author greedy DDPG supplies only the reference prefix.
Each probe applies one 0.2-second jerk step then 24 zero-jerk steps. B2/B3 share
all five candidates, histories, labels and splits; no ST, Risk network, new reward
or commitment is introduced.

The accepted replay backend currently supports the FIRST episode of a fresh
seeded worker. This dataset restriction does not change upstream DDPG training's
continuing-traffic resets. Distribution mismatch with eventual policy visitation
is a limitation: the author reference and three geometric positions may miss
important states encountered by an untrained or different policy.

## Proposed dataset

| Split | Simulator seeds (inclusive) | Episodes | Maximum roots | Unique branches |
| --- | --- | ---: | ---: | ---: |
| Existing development | 0,1,100 | 3 inspected | 9 existing | 45 existing |
| Train | 11000–11255 | 256 | 768 | 3840 |
| Validation | 12000–12063 | 64 | 192 | 960 |
| Calibration | 13000–13063 | 64 | 192 | 960 |
| Test, sealed | 14000–14127 | 128 | 384 | 1920 |

New total: 512 episodes, at most 1536 roots / 7680 unique branches. Each branch
is repeated once for exact replay verification: at most 15360 branch executions,
not independent statistical samples. Prefix discovery/replay adds cost; no full
runtime estimate is claimed before a collector benchmark.

All roots/candidates of an episode stay in one partition. The episode namespace
is the previously audited environment identity: a protocol/model rename must not
reclassify inspected traffic as fresh test data. These seed ranges are proposals,
not previously run seeds selected by performance. Future DDPG evaluation requires
separately reviewed seeds and must not reuse the predictor test set.

## Root sampling and accounting

Select the first nonterminal threshold crossing: ramp 20 m before end, ramp
internal midpoint, downstream 5 m after start. Discovery is capped at 250 steps.
Do not select locations by candidate success/collision or prediction improvement.
Unreachable targets and early reference termination remain in the manifest;
never replace seeds or positions to fill a quota.

Initially use one isolated worker process, fresh prefix replay and restored RNG/
control state. Native SUMO snapshot continuation remains unaccepted. Replay,
geometry or provenance errors stop the run and preserve failed evidence. Future
resume must check full job/config/source hashes, not rewrite failures.

Store root signatures, observed 11-frame history, original root traffic and the
12-neighbor adapter. Preserve each branch's original execution audit, requested
versus executed controls, future traffic and censored trajectory/event labels.
Also record separate read-only future lane position, lane ID, length and width
for vehicles still present. Terminal/missing actors stay missing. These queries
must pass unchanged base-trace/reward/action parity before production collection.
This supports later geometry labels without guessing from XY or recollecting
data; it does not itself define safety or ranking labels.

Review per-split/location missing roots, collision/arrival and censoring counts,
actor coverage/overflow, response magnitude and repeat equality before training.
No unsupported numeric coverage threshold or automatic training-ready flag is
invented here. Insufficient coverage triggers a design review, not favorable seed
replacement. Broader reference/traffic coverage requires a new dataset version.

## Proposed predictor training

Keep the accepted 64-wide model, three separately trained members each for B2/B3,
paired fresh initialization seeds 4101,4102,4103. These are ensemble members, not
DDPG optimizer replicates. No smoke/WcDT warm start or bootstrap.

- CPU, one Torch thread initially; GPU training/resume requires separate testing.
- Batch 16 complete root families; paired member-seed/epoch root permutations.
  A production trainer must save sampler/RNG state, not reuse the smoke's fixed
  batch resume claim for stochastic training.
- Adam lr=0.001, betas=(0.9,0.999), eps=1e-8, weight decay 0, gradient norm clip 1.
- Maximum 100 epochs, validation every epoch, early-stop patience 15 and absolute
  min_delta=0.0001. Upper bound: 4800 optimizer steps per member if all roots exist.
- Save the lowest finite validation root-equal scaled MSE, exact ties earliest.
  min_delta governs patience, not whether a genuinely lower checkpoint is saved.
- Same budget/selection rule for B2/B3; attained stopping epochs may differ.
- P4d scales [10,3,5,2], equal channel weights, masked cell then candidate then
  root averaging. Report excluded empty branches/roots. This is root-equal, not
  episode-equal: missing-root patterns must remain visible.

B2 learns a plan-marginal neighbor response from the same full candidate families;
B3 receives the candidate plan. Trajectory loss does not establish viability.
Calibration/test scores and later DDPG results cannot select checkpoints.

## Proposed calibration and sealed test

This method is not implemented. For each frozen ensemble, set coordinate scale
s=max(population ensemble std,0.1*channel_scale). One eligible calibration episode
contributes the maximum |truth-mean|/s across its available roots/candidates/
physical neighbors/times/channels. Use order ceil((n+1)*0.90) of n eligible scores;
if the order exceeds n, the interval is infinite, not silently clamped. For
64 eligible episodes, this is order 59. Empty episodes remain counted/excluded,
never replaced. Calibrate B2/B3 separately by the same rule.

Bands mean +/- q*s target nominal 90% coverage; this is not a demonstrated
guarantee. Report empirical episode coverage, width, eligibility, subgroup counts
and censoring. Exchangeability, reference-policy sampling and informative missing
tails limit interpretation. Scope is observed NONTERMINAL coordinates only, not
collision probability, arbitrary continuous-action response or online safety.

Collect/open test trajectories only after model selection and calibration rules/
parameters are frozen and hashed. Do not tune epochs, roots or losses on test.
P5 also needs simple prediction references and predeclared same-root ranking
using shared ego rollout/geometric summaries; that definition remains pending.

## Review and implemented boundary

Confirm/revise budgets and ranges, author-policy/three-root coverage, epoch and
selection settings, and calibration scope. Design approval is not run approval.

The implemented planner writes a draft manifest, 512 disabled collection jobs,
six disabled training jobs, resolved protocol, and literal upstream defaults plus
original overrides without executing config.py. No run or approve command exists.
The upstream base settings retain TASK=TRAIN_DDPG as provenance; NEVER feed that
planning artifact directly to upstream main.py. Future workers must separately
record their assigned seed/host in a fully resolved run configuration.
