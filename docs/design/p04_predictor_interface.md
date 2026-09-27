# P4c development predictor interface v1

This is an engineering interface, not a fitted model or a P5 prediction gate.
No original SUMO, reward, action, DDPG or old-project code is changed/imported.
All evidence roots are development-only. No optimizer is run.

## Matched ordinary/conditional interfaces

B2 ordinary and B3 conditional each contain three independently initialized
members. They use the same physical inputs, feature scales, actor masks, hidden
width (64), shared per-actor history GRU, one 4-head interaction attention layer,
and response head. No temporal/actor slot position embeddings are introduced;
ego has a role embedding and the three summary groups have type embeddings.
Missing history cells do not update GRU state. A relative time index locates
observed frames/holes on the fixed grid. The root ego is always a valid attention
key; root-empty actors/summaries are key-masked. Physical slot outputs are masked.

The dedicated summary adapter consumes seven values, seven validity bits and
relative time before its own shared GRU. Group history may exist while the group
is empty now; such a group is not a root attention key. Masked finite values
cannot influence predictions. Nonfinite values, even in masked cells, fail.
Variable IDs, role maps and root_actor_metrics are audit data, never channels.

The plan is one original 0.2-second jerk intervention followed by 24 zero-jerk
steps. Five probes are [-5,-2.5,0,2.5,5]; no action clipping or reinterpretation.
The predictor validates the full 25-element sequence, but only the first scalar
needs encoding because the suffix is frozen. A future different reference plan
requires a new contract, not a silent configuration change.

B3 encodes the actual intervention. B2 replaces it with a constant zero reference
before plan encoding; no candidate value reaches its network, and its neighbor
trajectory is repeated across candidates. This explicit information ablation
matches the network layout/parameter count, NOT the effective number of learned
degrees of freedom: the zero-input plan weight in B2 cannot learn a slope.
This small intentional ablation must be disclosed; do not claim identical
effective capacity or use unused legacy WcDT decoders. Both heads retain the same
learned constant bias context. B2 can later combine its ordinary neighbor forecast
with each candidate ego plan in a shared geometric summary layer; that layer is
not implemented here. Same member initialization seeds give paired starting
weights, but members will require separate training, not cloned final weights.

## Output and scaling

Each member returns [root,candidate,12,25,4], ensemble stacks a leading 3 axis.
These are single-mode physical NEIGHBOR response trajectories in the fixed root
ego XY frame with absolute speed/acceleration. Ego is a separate observed token,
not one of the 12 targets. Summary tokens have no individual trajectory targets.
The root actor mask denotes modeled slots, not future survival or safety. No
oracle future-valid mask is supplied at inference. There is no multimodal head.

Physical inputs use fixed SI divisors [100,10,30,5,100,5,2,1,1,1]; summaries use
[12,100,10,10,30,30,1]. These are development unit conversions without clipping,
not train/test fitted statistics. A root-state residual trajectory head scales
its four outputs by [10,3,5,2]. Untrained outputs can violate physical bounds;
they must not enter control. No probability, calibrated feasibility, confidence
interval, event head or action ranking claim is made. Ensemble population std
is only pointwise disagreement and is zero on padded actor outputs.

Scene encoding is performed once per root batch per ensemble member. Candidates
share encoded context and are decoded together. Batch equality is float32
allclose(atol=1e-5,rtol=1e-5), not a cross-device bitwise determinism claim.
Checkpoint roundtrip on the same CPU computation is required to be exact.

## Serialization and boundaries

Checkpoints store state_dict tensors plus JSON-safe provenance, complete config,
feature order, plan/scale version, mode and member initialization seeds. Loading
uses weights_only=True and exact keys/shapes/dtypes/fixed scale buffers. Wrong
mode, missing members, nonfinite weights and changed schemas fail; no partial
or WcDT checkpoint migration exists. New files publish atomically by hard link
without overwrite on the local filesystem. Checkpoints are ignored by Git.

P4c only permits training_status=untrained_interface_smoke. Trained checkpoint
provenance and loss/label alignment belong to P4d and must be explicitly added.
Remaining P4 work includes supervised target alignment, masked losses, leakage-
safe formal collection, training/checkpoint selection, and calibration. No P5
or DDPG comparison may be claimed complete based on interface checks.
