# P6b original DDPG feature integration acceptance

## Implementation and boundaries

P6a channels now enter actual ALL 0.5.3 DDPG States, actor/critic networks and
uniform replay. The original 400/300 network, TimeFeature, action scaling/noise/
clipping, Adam, Polyak targets, discount and cosine schedule remain upstream code.
Original reward, continuous jerk, SUMO control period and termination are unchanged.
Detailed traffic/feature audits are recorded outside replay metadata.

B0 retains original20 observation channels (actor21/critic22 with time/action).
B1/B2/B3 use169 observation channels (actor170/critic171). Strict evaluation-only
state-dict checkpoints bind arm, feature layout, predictor and upstream config hashes
and preset. They cannot silently load into incompatible agents or resume Adam/replay.
Formal training and long evaluation remain user-run; neither was started here.

Source commits:

- `fbcd3d67398a10be64e6b5c6437a89234f617a99`: DDPG integration/tests/smoke.
- `e274ca825b8d076cc048480f08d45b2be21baeae`: historical LF/CRLF binding repair.

The first launch stopped BEFORE creating an output directory: Git checkout had
converted the prior P6a JSON to CRLF. Its recorded LF byte hash still equals the
historical Git blob. The repair accepts only LF/CRLF spelling of that exact blob,
not changed parameters or even equivalent reformatting. Old reports, seeds and
thresholds were not changed. New source was committed before real training smoke.

## Tests actually executed

**439 passed in10.74 seconds**, with fresh artifact test directory and cache disabled.
`git diff --check` passed. Added22 DDPG cases and9 historical-checkout cases.
Synthetic B0 replay proves exact parity against the unmodified ALL preset for:
actions, stored states/masks, online and target weights, Adam steps and cosine state.
Other checks cover expanded matched initialization, reset/terminal conversion,
lean replay info, invalid values, zero terminal bootstrap, strict checkpoint
roundtrip/rejection without partial mutation and inherited original training loop.

## Real SUMO short optimizer smoke

Command:

```powershell
python tools/smoke_feature_ddpg.py --run-id p6b_ddpg_01
```

Report: `artifacts/p6/p6b_ddpg_01/report.json`.
SHA256: `694fe1fc35361a684d2c908becd11d79a3a797963838eaa444fa4b5392c0ea31`.
Status complete; engineering gate true; clean source `e274ca8`.
Started22:46:28, finished22:48:08 Asia/Shanghai; elapsed99.33 seconds.
SUMO1.22.0, ALL0.5.3, Torch2.5.1, NumPy2.2.6, Gym0.26.1, TraCI1.25.0.
CPU/one Torch thread; all arms share original run seed0 (optimizer/simulator
seed use the upstream coupled run-seed convention).

| Arm | Actual SUMO steps | Completed episodes | Actor/critic Adam updates | Weight roundtrip |
| --- | --- | --- | --- | --- |
| B0 original | 69 | 1 | 62/62 | Exact |
| B1 zero channels | 158 | 2 | 152/152 | Exact |
| B2 ordinary ensemble | 102 | 1 | 95/95 | Exact |
| B3 conditional ensemble | 101 | 1 | 94/94 | Exact |

Total430 steps; cap2,252. Nominal requested64 frames is checked between complete
episodes by upstream ALL; different actual lengths are expected, not a new cutoff.
Both actor and critic changed finite parameters in every arm. B2/B3 predictor
weights and gradients remained frozen. B1 channels remained exactly zero and
terminal blocks were zero. TensorBoard return/loss logs were retained per arm.

B1/B2/B3 initial policy SHA:
`89db2cde60f27fdaff905205802657e32d154f91f23f28303ccaf67fecfb9eaa`.
Initial critic SHA:
`b2321b0f4774eb16319886b5baa2e9bd797eae626a88a7f17ca3ecafbf03003e`.
Both hashes matched across all expanded arms in separate processes.

All episode outcomes are retained: B0/B2/B3 each had one arrival; B1 had an arrival
and a collision. These tiny training trajectories are not paired evaluation or a
method ranking. The collision is not an invalid implementation result and was
not deleted/reseeded. Engineering success is not evidence that ACCVP is superior.

## Next stage

P6b is accepted: actual optimization can consume the prediction channels while
preserving the original algorithm. This does NOT establish convergence, task gains
or readiness for formal claims. The smoke overrides warmup8/minibatch8 deliberately
activate updates; the ordinary factory retains upstream5000/100 defaults and2e6
schedule horizon. Smoke weights are evaluation-only and must not warm-start formal
runs. P7 must separately freeze small exploration seeds, training/selection budget
and independent development evaluation scenarios before user-run experiments.
No formal/exploration CLI is claimed implemented at this milestone.
