# P4i training review and calibration entry — 2026-09-29

Implementation commit: `ba6301a2f1890747b8998d07b29752f3849677b5`.
Branch: `codex/p04-predictor-calibration`. Training, data, configurations and
upstream/old-project source remain unchanged. No formal calibration, predictor
test, SUMO or DDPG evaluation was launched in this stage. No push or milestone tag.

## User-completed predictor training

Actual training commit: `4c93feae29c5ecd3b4f310a2b8467d65373335cc`, clean tree.
UTC start/end: 2026-09-29 01:59:36.865454 / 02:10:20.952709 (~10 min 44 s).
All six members completed the registered 100-epoch budget, 4500 optimizer steps
each. None stopped early. No calibration/test data was used for selection.

| Predictor | Initialization seed | Best epoch | Best validation scaled MSE | Last validation | Last train (pre-update) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 4101 | 90 | 0.02975039 | 0.030178 | 0.031840 |
| Ordinary | 4102 | 95 | 0.02987178 | 0.029905 | 0.031345 |
| Ordinary | 4103 | 99 | 0.02952640 | 0.031377 | 0.030878 |
| Conditional | 4101 | 98 | 0.02029044 | 0.020316 | 0.021110 |
| Conditional | 4102 | 99 | 0.02122127 | 0.022048 | 0.021588 |
| Conditional | 4103 | 96 | 0.02127532 | 0.022184 | 0.021856 |

Arithmetic mean of the three independently selected member losses: ordinary
0.02971619, conditional 0.02092901; descriptive relative reduction 29.57%.
This is NOT the loss of the averaged ensemble, a held-out test comparison, three
independent policy replicates, statistical significance or a closed-loop gain.
It is optimistic model-selection validation evidence worth proceeding with.

Both training and validation losses fell substantially. There is no obvious
large reverse-direction train/validation divergence in the recorded curves,
but the metrics use different timing (within-epoch pre-update versus end-epoch),
so their gap alone is not an overfitting test. Conditional models still improved
late: best epochs 96–99, versus ordinary 90–99. Full convergence is not established.
The current frozen budget is retained rather than increased ad hoc. Additional
training, if later justified, needs a new reviewed version and paired B2/B3 budget.

## Read-only training acceptance

- Historical source hashes checked against actual training Git blobs.
- Request/preparation/dataset-review/previous audit bindings and frozen jobs checked.
- All six immutable member inventories checked, including epoch checkpoints.
- Entire selection histories replayed; strict best-loss/earliest-tie rule verified.
- Each selected.pt tensor equals its selected epoch's model tensor.
- Both exported ensembles exactly contain the three selected member states.
- Strict state-dict-only CPU loader verifies mode, dimensions, dtype, finiteness,
  fixed buffers, feature contract, selection lineage and artifact SHA256; no partial load.
- No current writer lock; test collection remained unopened.

## Calibration implementation and actual tests

The P4e/P4g rule is unchanged: max absolute standardized residual per episode,
population ensemble std floored by 0.1*[10,3,5,2], nominal 0.90, observed nonterminal
coordinates only. Float32 model outputs use float64 mean/std/residual accumulation.
All candidates and roots of the same episode count as one calibration unit.
For n=64 eligible episodes, the one-based order is 59; small n requiring rank>n
produces explicit unbounded intervals, not silently clipped quantiles.

Separate loader/entry disallows fitting on train/validation/test. Prepare performs
data alignment/hashing only, not prediction or fitting. Run requires exact request
confirmation, clean committed source and unchanged prerequisites. Fitted sample
coverage/coordinate width are explicitly not test evidence. Negative/uninformative
bands can complete successfully; engineering errors preserve failure reports.

Full regression: **336 passed in 11.76 s** (31 new tests). Tests cover finite-sample
order, ties, empty episodes, strict masks/splits, population std/floor, partial or
tampered weights, selected-history validation, no-fit preparation, manual hash
confirmation, immutable successful output and preserved failure without partial freeze.
Explicit staging/whitespace/sensitive-string checks passed; no weights/raw data staged.

Actual bounded audit used only the existing 9 roots from 3 development episodes
and the frozen trained models. Batched versus single-root standardized scores
passed predeclared atol=1e-5 plus rtol=1e-5 (not bitwise equality): maximum absolute
difference ordinary 5.6283e-5, conditional 2.2157e-5. Both correctly returned an
unbounded nominal-90% quantile with only three development episodes; this was an
interface check, not formal calibration. No gradients or optimizer calls were made.

## Immutable artifacts

| Relative path | SHA256 |
| --- | --- |
| `artifacts/training/predictor_v1/invocations/648e96637f5e/report.json` | `b47ba0002ee9e86b5cf7e1692a892e13b971e09ff0e506b17b2aeb4aff941485` |
| `artifacts/training_reviews/predictor_v1_review/review.json` | `01fe480aa2a36fac7f600cea0228eb62c3af560c7008fae44cdf51ea7f001a9d` |
| `artifacts/p4/p4_calibration_v1_01/report.json` | `e9f80486c5f6f8fd5f57bdc62a5410adc52f6eb8d3c89fac6a06a2b578bf8683` |
| `artifacts/calibration/predictor_v1/request.json` | `503fe57309813a6ae667885e31b135bda0b68a6385b597aa4ffd9bd450b97cbf` |

Ordinary ensemble SHA256:
`939242f358315bd00ce89978422e4b55b8059f41837463d2e80f661c63817e8d`.
Conditional ensemble SHA256:
`daddeda8569b88c2a3482a0e8c739995785ee0f8216dda6525c86d2ce120e25f`.
Calibration request canonical confirmation hash:
`1321568a60e0f4acf3479026981b140b4a7d919c33c56bcca5a14643ffb3a7bb`.

Review/audit/prepare ran from clean implementation commit. Prepared request
verification passed, calibration roster=64, formal started.json absent,
formal calibration.json absent. Existing source-phase reports remain immutable.

## Pending acceptance

User must run calibration. q/width/coverage results are currently unknown.
Calibration does not establish candidate ranking or safety. P5 still requires
frozen simple references, shared ego rollout/geometry, same-root ranking and
censoring/coverage definitions, plus a separately authorized test release after
model/calibration/evaluation freeze. DDPG remains blocked on that methodological
acceptance, not on obtaining favorable calibration numbers.

Next manual command: `docs/runbooks/p04_predictor_calibration.md`.
