# P3 development diagnosis — 2026-09-26

Status: engineering replay checks PASS; interaction-mechanism gate FAIL.
P3 is paused. Do NOT start P4 model/data implementation or formal collection
on the premise that action-conditioned neighbor responses have been established.

## Scope, provenance and evidence

Parent: 1df4595 (P2). Implementation checkpoint: 3ec6e8f.
Branch: codex/p03-counterfactual-diagnostics. Upstream code/configs and P2 adapter
are unchanged; old E:/WCDT_ACCVP remains untouched. No model was trained.
No result was discarded, no seed replaced and no threshold relaxed.

Raw evidence: artifacts/p3/p3_dev_01/report.json and seed_<n>/ subdirectories.
The diagnostic ran once, in 104.07 seconds, with a dirty development source
snapshot and diff saved at launch. It is not a clean formal experiment.
The committed code adds environment metadata logging after that run; this did
not change its simulation/branch logic. No extra run was made to seek a pass.

Post-run environment audit (not backfilled into the immutable report):
Python executable in the existing pytorch environment; torch 2.5.1,
NumPy 2.2.6, Gym 0.26.1, TraCI 1.25.0, Cython 3.0.12, cvxopt 1.3.2,
ALL 0.5.3 and SUMO 1.22.0. No package or system installation in P3.
The original native SUMO snapshot also retains its simulator version header.

## Predeclared intervention and gate

Config: configs/development/p03_mechanism_v1.json.
Original environment config: configs/train_default_1.json under the upstream tree.
Seeds 0, 1, 100; first-episode roots after 0 or 30 zero-jerk policy steps.
Probes are five fractions of the ACTUAL upstream action bounds:
-5, -2.5, 0, 2.5, 5 m/s^3 in this environment.
Each probe acts for one original 0.2-second cycle; then zero jerk continues.
Horizon: 25 steps / 5 seconds, censored at original episode termination.

Zero jerk is NOT zero acceleration: after intervention, acceleration can
persist until upstream/SUMO constraints change it. This reference continuation
is frozen, not a claim about an arbitrary future DDPG policy.

Five branches run forward and then repeat in reverse order at every root.
Thus 6 roots, 60 branch rollouts, each capped at 25 steps, plus prefix replay.
A separate native-save probe is not used to generate these labels.

Response compares absolute positions/speeds of the SAME non-ego root actors
at the SAME simulation time. Ego-relative gaps are not evidence of neighbor
response. Missing actors and terminated tails are censored and counted, not
zero-filled. Newly appearing actors do not count as root-actor response.

Pre-run screening thresholds: at least 0.1 m position difference OR 0.05 m/s
speed difference. Mechanism gate requires all planned roots evaluated and
response observed in at least two development seeds. This is a conservative
development check, NOT a statistical significance claim or safety threshold.

## Results

All six roots replay exactly. All forward/reverse repeated branches match
exactly, including reward, observations, traffic and execution-audit fields.
57 unit/regression tests passed after final metadata/test additions.

| Seed | Prefix steps | Max ego position difference (m) | Max neighbor position difference (m) | Max neighbor speed difference (m/s) | Response |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | 0 | 26.00 | 0 | 0 | No |
| 0 | 30 | 22.08 | 0 | 0 | No |
| 1 | 0 | 26.00 | 0 | 0 | No |
| 1 | 30 | 24.00 | 0 | 0 | No |
| 100 | 0 | 26.00 | 0 | 0 | No |
| 100 | 30 | 26.00 | 0 | 0 | No |

engineering_gate=true; mechanism_gate=false; response_root_count=0/6.
This is an exactly zero observed neighbor difference, not merely a marginal
failure of the chosen response thresholds.

## Why this is not yet a valid interaction-training sample

Measured background speeds across ALL observed branch frames were 7 m/s.
The author config caps normal traffic at 7 m/s; route types use Krauss with
sigma=0, speedDev=0 and speedFactor=1. These settings do not prove that
neighbors can never react, but the measured traces contain no such reaction.

Root coverage is weak:

| Seed/root | Ego x at root (m) | Ego speed (m/s) | Root neighbors | Ego frames on highwayahead_0, across five branches |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | -211.41 | 23.82 | 11 | 0 |
| 0 / 30 | -69.52 | 23.82 | 14 | 49 |
| 1 / 0 | -211.41 | 23.12 | 12 | 0 |
| 1 / 30 | -73.69 | 23.12 | 15 | 45 |
| 100 / 0 | -211.41 | 6.25 | 12 | 0 |
| 100 / 30 | -174.32 | 6.25 | 15 | 0 |

All roots themselves are still on ramp_0. Four windows never reach the
downstream main lane. In the two windows that do, ego begins much faster than
background traffic. This makes insufficient conflict coverage a plausible
explanation, not proof of causality or of universally noninteractive SUMO.

The upstream sets ego speedMode=22 and applies speed commands. We did not
change that execution contract or background response settings to create a
favorable result.

## Native SUMO snapshot is NOT accepted

The native probe enables RNG serialization and 17-digit state precision;
Python RNG, control.delay, raw environment attributes and wrapper counters are
restored. Its configured root and direct continuation match original replay.

After loadState, root traffic and the captured Python signature match, but
the first identical zero-jerk continuation differs:

| Quantity | Uninterrupted | Native-restored |
| --- | ---: | ---: |
| Pre-step ego speed (m/s) | 23.8202610 | 23.8202610 |
| Reconstructed speed command (m/s) | 23.8202610 | 23.8202610 |
| Measured post-step speed (m/s) | 23.8202610 | 22.6202610 |
| Measured post-step acceleration (m/s^2) | 0 | -6 |

native_eligible=false, root_exact=true, continuation_exact=false.
RNG tags were present; therefore merely enabling RNG save is insufficient.

This proves an un-restored execution-state difference exists, not which
specific variable caused it. TraCI control modes/command state and native
internal model state are next diagnostic targets. speedMode loss has NOT been
measured and must not be asserted as the confirmed cause.

SUMO documents default RNG omission and limitations in saving some internal
vehicle-model state:
https://sumo.dlr.de/docs/Simulation/SaveAndLoad.html
The concrete failure above is from the local SUMO 1.22.0 test.

Trusted development backend remains fresh_sumo_prefix_replay_v1: restart the
original simulator, restore run-start RNG, replay the original prefix, then
require exact root/Python/RNG/prefix trace signatures. This is slower than
native restoration and limited to first-episode action-prefix roots here.
It is not an efficient formal dataset collector or a portable snapshot format.

## Next decision, not executed

Keep this v1 negative result. Do not expand the network/ensemble or start
expensive training to compensate for absent label variation.

Recommended next diagnostic version: keep the same seed list and original
reward/action semantics, but choose roots using predeclared road-position or
time-to-merge criteria, independent of observed response/outcome. Report all
eligible and unavailable roots. Roots sampled from author-policy rollouts
could be a separate declared coverage study, not an unrecorded change.

Author-provided train_moderate_1.json (11 m/s traffic) and train_fast_1.json
(15 m/s) exist. They are options for a separately declared environment-coverage
study, not replacements silently applied to the current baseline. Do not
search configurations and retain only those favoring ACCVP.

If broader neutral coverage still yields no neighbor response, reconsider
the B3 neighbor-response claim. Action-conditioned ego geometry/viability may
still vary even when neighbors do not; that is a different claim and may need
much less machinery than a learned conditional neighbor predictor.

P4 remains blocked pending this P3 decision. No P3 success tag, merge to main,
push, formal dataset or model training is performed in this handoff.
