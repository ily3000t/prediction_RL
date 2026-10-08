# Same-root candidate response-difference objective v1

## Scope

The sole scientific change is the conditional predictor's training objective:
`trajectory_MSE + lambda * response_difference_MSE`. No architecture, actor
adapter, output head, online feature layout, reward, jerk interface, DDPG network,
commitment, Shield, calibration rule, test split or simulation protocol changes.
The original ordinary and absolute-MSE conditional ensembles remain references.
This is paired difference regression, not InfoNCE or real-world causal discovery.

## Alignment and reduction

`align_supervision` binds all five candidates to one root, one ordered physical
actor roster and the FIXED root ego coordinate frame. Summary tokens are inputs,
not physical trajectory targets. Only the intersection of the two branches'
actor/time masks is eligible. Missing actors and terminal/post-terminal labels
are never replaced with observed zero trajectories.

For every unordered candidate pair, compare predicted response differences with
label differences. Normalize the four channels by the existing SI scales
`[10, 3, 5, 2]`, with unchanged equal channel weights. Average eligible actor/time
cells within each pair, valid pairs within each root, then roots with absolute
trajectory supervision. A trajectory-supervised root with no common pair has
zero difference contribution, remains in that root denominator, and is counted.
An entirely trajectory-empty batch remains an error, never an optimizer update.
Coverage counts distinguish valid pairs, possible pairs, common actor/time cells,
roots with pairs, roots without pairs and the root denominator.

The existing offline diagnostic excludes pair-empty roots from its pair-error
mean. Therefore its mean need not equal the training difference component when
pair-empty roots exist. Report their respective denominators rather than quietly
equating these estimands. Validation comparisons reuse identical frozen masks.

## Interpretation and weight

With identical valid support across M candidates, errors `e_i = prediction_i -
target_i` decompose into a shared mean error and candidate-centered error V:

`L_absolute = squared_mean_error + V`

`L_difference = 2*M/(M-1) * V`

For five candidates the factor is 2.5. The development weight `lambda=0.2`
increases the candidate-centered error weight from 1 to 1.5 on this common
support. It is a predeclared bounded objective trial, not a tuned optimum or a
claim of extra labels. Nonidentical branch masks alter the weights and prevent
using this identity as an exact global masked-loss equality.

Keep absolute supervision: a shared position bias cancels from difference loss
but is still wrong for geometry. If true responses are identical, the correct
response difference is zero; action identity alone is not a reason to separate
predictions. Terminal censoring means this loss does not fully supervise
collision, arrival, viability or reward differences.

## Training, selection and identity

Each of the three conditional members is trained independently from the same
frozen initialization seeds, data, root shuffling and optimizer/budget settings
as the original trial. Do not optimize only the ensemble mean. Ordinary mode
has identical outputs for all candidate plans, so its difference loss has no
useful model gradient; the new trainer rejects that mode.

Checkpoint selection and patience still use validation root-equal trajectory
MSE. Difference loss and combined validation objective are diagnostic fields,
not new selection criteria. This keeps the training objective as the single
changed scientific variable. Early stopping may produce different actual epoch
counts; report those counts, not just the common maximum of 100 epochs.

`lambda=0` directly returns the legacy objective tensor, with no added zero
gradient graph. Tests verify exact scalar/gradient equality, two-epoch parameter
and Adam equality, sampling/CPU RNG equality and unchanged selection. The new
checkpoint envelope intentionally has a new version even for lambda=0; semantic
state equivalence is not equality of serialized checkpoint file hashes.

Difference checkpoint identity binds its loss contract, weight, data fingerprints,
configuration, request digest and all three member selections. Old consumers
reject this version. No weights-only resume, silent partial load or rewriting
of historical requests/artifacts is permitted.

## Evidence boundaries

The new CLI prepares a request, USER-runs three predictor members, and produces
paired validation diagnostics against the two frozen ensembles. Negative results
complete normally. Calibration and test are not used for selection. This does
not authorize DDPG training, prove method effectiveness, or claim calibrated
uncertainty. Any subsequent policy comparison needs explicit predictor binding
and matched from-scratch policy training; an old B3 policy is not the formal
policy for a changed predictor observation mapping.
