# P6a prediction features — DRAFT, not frozen or executable

Superseded after user authorization by `p06_prediction_features_v1.md`.
This document remains the historical proposal; numeric implementation is v1.

## Motivation and boundary

P5c shows maximum bumper distance is weakly aligned with original branch reward.
Conditional prediction helps critical responses at the merge connection but is
worse than CV at some other locations. Preserve both findings. Do not change
the predictor architecture/reward yet or claim independent generalization.

Retain the original observation as an unchanged prefix; append a fixed-layout
block for the five original jerk probes. DDPG retains continuous jerk and the
original actuator. No argmax snapping, action mask, ST, commitment, reward shaping
or wall-clock-dependent observation substitution.

## Proposed minimal information families

Exact fields/time grid/dimension require review and freezing before implementation:

1. Relative position and speed at a small fixed set of future times for root-known
   target front/rear actors. These roles already exist in the adapter. Keep each
   identity fixed over a candidate rollout, never select by true future outcomes.
2. A small pooled proximity descriptor over all selected physical actors, retaining
   other modeled interactions beyond the front/rear pair. It remains a continuous
   descriptor, not a safety flag, reward, or chosen action.
3. Root-known actor/history masks and omitted-actor status. Never import future
   disappearance, terminal or common-support masks from P5b/P5c into observations.

Prefer fixed-time relative states over adding all TTC/DRAC/event features at once.
TTC/conflict-time needs a path, zero-speed/nonclosing and missing-data contract;
XY alone does not supply future lane IDs. Do not silently invent them.

No learned event/risk head. Ensemble disagreement is not calibrated probability.
If included, define per-member nonlinear summary and ensemble aggregation order;
old wide calibrated bands are not action-validity flags. Its extra input dimensions
must be justified, not automatically included as another feature bundle.

Do not initially add engineered jerk-cost or ego-progress reward-proxy channels
just to improve task agreement. These are analytic control information, not learned
neighbor prediction. If added later, give them to matched controls and separate
their contribution from the predictor's contribution.

## Comparison design

- B0: exact original DDPG observation/interface.
- B1: matched expanded policy with all added prediction channels zero.
- B2: ordinary ensemble plus the shared feature/ego-rollout adapter.
- B3: conditional ensemble plus the identical adapter.
- CV: mandatory offline reference check. Adding a CV closed-loop training arm is
  recommended by the strong existing results but needs budget approval; it is not
  silently added to the original four-arm training plan.

B1/B2/B3 share hidden layers, input layout, initialization scheme, training and
checkpoint selection budget. B0 is an original-interface reference, not a matched
parameter-count control. B2/B3 share actor slots, histories, normalization, masks
and ego plans; only neighbor prediction differs. Never give B2 the candidate plan
to enforce capacity matching. No test-set fitting or feature selection.

## Engineering-only acceptance

1. Inputs are observed history, current map and frozen predictions. Poisoning labels,
   future actor IDs or terminal masks cannot change generated features.
2. Dimensions/order are fixed; absent actors, short history and near-zero velocity
   have defined finite values/masks. Invalid model values explicitly invalidate a
   run; do not silently clip into a fabricated safe state or use stale predictions.
3. Extracting state/predicting does not advance SUMO. Each step delegates one action.
4. Action replay with fixed seeds preserves original observations, rewards, executed
   controls and termination with augmentation on/off. Different learned policies
   need not choose identical actions; this is an implementation parity test.
5. Reset clears history. Terminal states do not infer for removed ego. Time-limit
   and original continuing-traffic semantics remain unchanged.
6. B2 neighbor predictions are plan-invariant although candidate-relative geometry
   can differ through ego plans. Batched/per-probe features agree. Weights remain
   frozen with gradients disabled and original SUMO time grid unchanged.
7. B1 is exactly zero in the added block. Do not silently partially load a B0 policy
   with incompatible input dimensions. Predictor reuse is separate from policy reuse.

## Next authorization

Freeze a minimal numeric interface first, then implement unit tests and bounded
development replay without learning. Small closed-loop exploration has a separately
approved seed list, budget and outcomes. Judge original reward and actual driving
results, not whether a new offline scalar wins. A short smoke proves integration,
not method efficacy. Test stays sealed during this design step.

This draft authorizes no formal training, new data, architecture changes, or
outcome-selected location-dependent switching. Negative outcomes remain reportable.
