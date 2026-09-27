# P4d supervision and bounded training interface v1

Scope: engineering checks using the already inspected development roots. No
formal data collection, predictor fitting, checkpoint selection, calibration or
DDPG training is authorized by this stage. The trainer is named DevelopmentTrainer
and refuses more than two updates per member path; it is not a production trainer.

## Join and leakage boundaries

Join history and label packs by (episode_id,root_id), then gather future physical
neighbor targets by the root adapter's actor_ids. Do not assume that label array
order is slot order. Require all five candidate IDs, validate their frozen plan
values, and reject duplicate/missing families. Root traffic, root-frame actor
coordinates, features, masks and declared time horizon must agree.

Same-episode split assignment is checked against the explicit manifest, including
all development/train/validation/calibration/test lists. Adjacent roots and their
branches cannot leak across partitions. Reading validation/test targets for
evaluation is allowed by alignment, but the bounded trainer accepts development
only. The current three inspected episodes are never promoted to train or test.
Formal data collection and split approval remain unimplemented.

The observation dictionary remains the P4c fixed whitelist. Targets and future
masks are separate SupervisedBatch fields, never passed to the predictor. Ego
and omitted summary tokens receive no neighbor trajectory target; future arrivals
cannot alter the root actor selection. Missing actor/terminal/post-terminal cells
are masked. A terminal-frame mask violation is an error. Unknown events are not
invented negatives; existing collision/arrival labels are preserved in source
packs but are NOT trained by this trajectory-only objective.

## Explicit loss

For each valid actor/time cell, average squared residuals over four channels,
after division by [10,3,5,2] for root-relative x, y, speed and acceleration.
All channel weights are 1. These fixed development choices are not fitted on any
dataset and must be reviewed/frozen before formal training.

Reduction is: mean over valid actor/time cells within each candidate; equal mean
over supervised candidates within each root; equal mean over supervised roots.
This prevents longer nonterminal branches or roots with more target cells from
dominating merely through cell count. It is root-equal, NOT episode-equal; a
future formal collector must freeze root sampling/count policy as well.
Fully censored candidates/roots are excluded AND explicitly counted; an entirely
unsupervised batch raises without any optimizer update. No zero target or safety
label is substituted. Nonfinite input/output/labels remain errors even if masked.

B2 and B3 use the same full candidate families and supervision weighting. B2 has
no action-plan input, so it learns a plan-marginal response under the declared
candidate distribution, rather than a zero-jerk-only target. B3 may distinguish
the candidate branches. The ordinary comparator's ambiguity is intentional and
must be disclosed; it is not fair to secretly train it on fewer branches.
Trajectory supervision alone may underrepresent early-terminal unsafe outcomes;
it cannot substitute for event/feasibility/ranking validation in P5.

Each ensemble member receives its own loss and independent Adam state. There is
no optimization of the ensemble-mean prediction. Current smoke uses one full
9-root batch, Adam lr=0.001, betas=(0.9,0.999), eps=1e-8, no weight decay,
gradient norm clipping at 1, CPU float32 and one Torch thread. Loss/gradient
finiteness for all members is checked before their updates.

## Training state and exact resume

P4d has a separate versioned training-state format, explicitly marked
development_smoke_only. It does not relabel/update P4c untrained checkpoints.
Saved state contains complete model tensors, three Adam moment sets and steps,
model/loss/optimizer contracts, initialization seeds, source provenance, and a
fingerprint of the exact observation/plan/target/mask batch plus ordered roots,
episodes, split and manifest. Changed data or split cannot silently resume.

Loading is weights_only=True with exact keys, shapes, dtypes, finite values,
fixed scaling buffers, optimizer parameter groups, nonnegative second moments
and step consistency checks. Partial checkpoint loading is forbidden. Atomic
short temporary files publish without overwriting existing evidence.

The smoke compares uninterrupted step 1->2 against step 1->save->load->2, including
all model tensors, Adam moments and resulting inference. This has two updates per
member trajectory plus a replay of the second step on a restored copy. The fixed
batch model has no dropout, shuffling, stochastic sampling or scheduler; its only
randomness is recorded initialization. This is NOT a guarantee of resumability
for a future shuffled/GPU training loop, which must save its own RNG/sampler state.

No acceptance threshold is based on loss decreasing. Finite updates and exact
resume establish engineering validity, not generalization, convergence, calibrated
uncertainty or B3 advantage. Neither smoke checkpoint is approved for control.

## Remaining work

Next: formal collection/training protocol design and review (P4e), including
episode coverage, root sampling, splits, budgets, selection rules, loss weights,
calibration design and collection entry points. Full training remains user-run
after approval. The shared ego-plan/geometric summaries and P5 action-ranking
checks are also pending; do not jump directly to expensive DDPG experiments.
