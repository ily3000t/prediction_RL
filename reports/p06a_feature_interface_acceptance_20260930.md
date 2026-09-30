# P6a frozen feature interface acceptance — 2026-09-30

## Scope completed

Implemented observed-only frozen ensemble inference, 149 named prediction
channels, and the blocking B0/B1/B2/B3 observation wrapper. Original input is
20 float32 values; expanded input is 169. Continuous jerk actions and original
reward are untouched. No predictor retraining, architecture change, policy
learning, ST, commitment, new reward or test evaluation occurred.

Frozen design: `docs/design/p06_prediction_features_v1.md`.
Source commit: `3a69ccc2e1e7f635573f4d0c697061b85438671f` (clean).

## Tests actually executed

`python -m pytest tests -q -p no:cacheprovider --basetemp <new artifacts directory>`:
**408 passed in 10.40 seconds**. Includes 15 new tests (parameterized cases):
exact layout/scales, empty actors/masks, omitted count, poisoned future metadata,
per-candidate/batched parity, ordinary candidate invariance, frozen weights/no
gradients/no RNG consumption, invalid prediction rejection, original action/
reward/prefix delegation, reset history clearing, terminal no-inference, faulted
wrapper behavior and bounded development configuration. `git diff --check` passed.

## Real SUMO replay

Command:

```powershell
python tools/audit_prediction_features.py --run-id p6a_features_01
```

Report: `artifacts/p6/p6a_features_01/report.json`.
SHA256: `18abb76be0d4c7a665ac7ff4f08e4c53ee1622dfa334c90aa296605b9373e06b`.
Status complete; engineering gate true. SUMO 1.22.0, pinned pytorch Python
environment, CPU frozen three-member inference with one Torch thread.
Started 22:10:57, finished 22:14:01 Asia/Shanghai; elapsed 184.14 seconds.
Eight fresh subprocesses; maximum configured budget 1,440 steps, **actual 480**.

| Replay | Per-arm steps | Arms | Outcome |
| --- | --- | --- | --- |
| seed 0, positive jerk, continuing-traffic reset | 45 + 35 | B0/B1/B2/B3 | Original termination preserved |
| seed 100, fixed mixed jerk sequence | 40 | B0/B1/B2/B3 | Bounded prefix, not fabricated terminal |

All six baseline-versus-expanded exact trace comparisons passed: original float32
observations, rewards, requested and reconstructed/measured controls, original
info/termination, traffic snapshots, continuing reset delay and CPU Python/NumPy/
Torch RNG states. B1 appended channels were exactly zero. Terminal blocks were
zero and no terminal prediction was invoked. B2/B3 produced nonzero, different
feature traces while preserving the baseline execution trace.

| Prediction arm | Stream augmentation mean / max | Mixed augmentation mean / max |
| --- | --- | --- |
| Ordinary ensemble | 19.88 / 32.57 ms | 20.09 / 24.31 ms |
| Conditional ensemble | 19.53 / 29.98 ms | 18.75 / 24.51 ms |

These times include state reads/adapter/ego reference/inference/summary and all
returned observations, including reset and zero terminal blocks. They are a small
engineering sample, **not** deployment-runtime acceptance or scaling evidence.
No latency threshold changes an observation. Raw trace/config/checkpoint hashes
and per-worker reports are retained under ignored artifacts, not committed.

## Limits and next stage

The feature geometry is a hypothetical root-based lane-chain reference, including
known downstream extrapolation; not actual future availability or safety assurance.
Root masks cannot predict disappearance or future entrants. Pooling mean trajectories
is not probabilistic collision checking. Existing poor/mixed offline results remain
unchanged; this replay does not establish efficacy or independent generalization.

P6a is accepted. **DDPG is not connected yet**: do not launch original training and
assume it uses the new channels. P6b must wire the upstream continuous DDPG network
and replay buffers, reject incompatible policy checkpoint shapes, preserve reward/
control/optimizer semantics, and complete a short four-arm training smoke. Formal
exploration requires separately frozen seeds/budget and is run by the user.
