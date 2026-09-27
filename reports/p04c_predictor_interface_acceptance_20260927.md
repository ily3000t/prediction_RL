# P4c predictor interface acceptance — 2026-09-27

Result: **interface_gate=true**. This accepts the engineering interfaces only.
No SUMO episode, optimizer update, formal ensemble training or DDPG training ran.
formal_training_ready=false; P4 as a whole and P5 are not complete.

## Provenance

- Code commit: 3fece2d522bbb343d4e5d706db7b0d046279e75e, clean tree.
- Run: artifacts/p4/p4_interface_v1_01/report.json.
- Report SHA256: 2a668c9ee0a49a114e2a840ebc7bbcecfae982f34491ab26778bf1e27623fda4.
- Config: configs/development/p04_predictor_interface_v1.json.
- Config SHA256: f24b854c7751f81385399ec041943e76bf936a52b63118e7d2eedc5099e6ed81.
- Resolved config SHA256: e7a18ca8d3ccccf63726821cfcf7a431b236b8babbc8127be7e3c06b35e66c10.
- Source: accepted P4b report, SHA256 89d4f73676a1fa6816421731cc30ee4d6f379d3e6d971879ccfdda20cfbb6c8c.
- All nine root files and source report remained unchanged.
- Development simulator seeds: 0, 1, 100; three roots per episode, five plans per root.
- Member initialization seeds: 3101, 3102, 3103. These are NOT optimizer replicates.
- CPU float32, one Torch thread; Python 3.10.16, Torch 2.5.1, NumPy 2.2.6, Windows.
- Full command/environment and individual input/checkpoint hashes are in the raw report.

## Results

| Check | Ordinary B2 | Conditional B3 |
| --- | --- | --- |
| Parameters per member | 66,340 | 66,340 |
| Ensemble members | 3 | 3 |
| Root/candidate coverage | 9 / 45 | 9 / 45 |
| Output finite | pass | pass |
| Candidate batch vs serial maximum absolute error | 0 | 0 |
| Root batch vs serial maximum absolute error | 0.0000152587890625 | 0.0000152587890625 |
| Checkpoint roundtrip output | exact | exact |
| Candidate information path | invariant to plan | connected to plan |
| Scene encodings per 9-root batch | 3 (one per member) | 3 (one per member) |

Root/candidate batching uses the predeclared float32 criterion
abs(a-b) <= 1e-5 + 1e-5*abs(b), not an absolute-only 1e-5 criterion.
The maximum root-batching difference therefore passes without changing tolerance.
Checkpoint roundtrip is exact on the same CPU computation.

The output shape is [3 members,9 roots,5 candidates,12 neighbors,25 times,4
trajectory channels]. Ego is separate; summary groups are not physical output
actors. These 45 candidate evaluations are NOT 45 independent traffic scenarios.
The existing roots came from only three development episodes.

## Tests and source scope

- New prediction tests: 27 passed.
- Full regression: 156 passed (5.02 seconds reported by pytest).
- git diff --cached --check passed after fixing a blank EOF line.
- Tests cover short/empty history, empty actors, overflow summaries, masked-value
  invariance, actor-slot permutation, B2 plan isolation, B3 gradient flow without
  optimizer updates, candidate/root batching, strict checkpoint corruption
  rejection, independent members and preserved caller CPU RNG.
- No upstream source, original reward/action semantics or old-project files changed.

## Interpretation and next gate

The weights are random initialization. Conditional output differences demonstrate
only that the conditioning path is connected, not accurate interaction response.
Finite outputs do not imply physically valid trajectories. No calibration,
ranking improvement, collision reduction or control improvement is claimed.
The ensemble standard deviation is disagreement, not a confidence interval.
The ordinary zero-plan ablation has matching parameter layout but not identical
effective degrees of freedom; this limitation is documented in the interface design.

Next P4d must align existing future labels by selected actor IDs, respect terminal/
missing masks, implement explicitly weighted supervised losses and training metadata,
and test save/resume and data split isolation. Formal collection/training still
requires reviewed configurations and user execution. Calibration and P5 offline
accuracy/candidate-ranking checks must precede an expensive DDPG experiment.

No milestone tag or remote push was made. Upstream publication permission remains
unresolved; local engineering acceptance does not resolve licensing.
