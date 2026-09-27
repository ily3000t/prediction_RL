# P3 v3 author-policy reference acceptance — 2026-09-27

Status: engineering PASS; predeclared mechanism coverage gate PASS.
Next: P4 design review, not formal collection or training.

## Provenance and frozen scope

- Clean implementation: `fde9d8123bb996c030b9648bd98c2c5c3410d88a`.
- Plan: `configs/development/p03_mechanism_v3_author_policy.json`.
- Plan SHA256: `ef05fbdc6155a85d0759f68104e67bf8e0685b69d71e94e25ab6f9cc0cfcf4bd`.
- Author policy: `RL-MPC-LaneMerging-master/pretrained_models/ddpg_default1_extended/policy.pt`.
- Policy SHA256: `548663282e348356f658fe8204b44b7ad589d935b05a1f18fc3d8927b3dd7750`.
- Evidence: `artifacts/p3/p3_policy_v3_01/report.json` and per-seed traces.
- Report SHA256: `de70e24603f1647769e557c2649a27e845e34b581139f8900ccec2c138bec4ca`.
- One bounded run: 220.03 seconds, clean worktree at launch, original CUDA
  inference setting, existing pytorch environment. No package installations.
- 87 unit/regression tests passed before the run.

User approved changing only the reference trajectory used to reach geometric
roots: author DDPG instead of zero jerk. V1/v2 configurations and reports remain
unchanged. Seeds 0, 1, 100; original train_default_1 environment/reward;
three lane-position targets; five bound-derived probes; one 0.2-second probe;
25-step horizon; subsequent zero jerk; response thresholds and gate all remain.
No ST, new reward, policy training, checkpoint search or seed substitution.

## Inference and restoration validation

The pinned author full-module checkpoint is hash-checked before deserialization.
Inference retains original ALL GreedyAgent, TimeFeature (scale 0.001), state
dtype/mask conversion, continuous action scaling and original device selection.
The q network is not needed for greedy actor inference and is not loaded.
No second simulator is opened to construct the observation codec.

At each discovery step, an independent time-feature wrapper invokes the unchanged
upstream DDPGAgent.get_control on raw.previous_state. Its action must equal the
new observation-path action exactly. RNG must remain unchanged. No clipping,
exploration noise, optimization or ST action replacement is added.

Every seed's reference discovery runs twice from the same initial state. Exact
actions, selected traffic, lane positions and root metadata must match. Selected
action prefixes are then replayed independently before candidate rollouts.

| Seed | Steps per reference discovery | Repeated discovery | Upstream action parity | Probe repeats |
| --- | ---: | --- | --- | --- |
| 0 | 55 | Exact | Exact, every step | Exact |
| 1 | 58 | Exact | Exact, every step | Exact |
| 100 | 111 | Exact | Exact, every step | Exact |

There are 448 checked inference steps across both discoveries. Nine roots have
five candidates repeated in reverse order: 45 distinct branch rollouts and
45 repeats. These are not 90 independent scenes. Exact agreement is verified
on this environment/device, not guaranteed across all hardware/SUMO versions.

## Coverage and response results

Targets are unchanged: ramp_0 at 20 m before its end, :mergenode_1_0 midpoint,
and highwayahead_0 at 5 m from start. All roots are first nonterminal threshold
crossings, chosen before any candidate result is evaluated. Discovery budget
is 250 steps per pass; all 9/9 targets were reached without replacement.

| Seed | Target | Prefix steps | Ego speed (m/s) | Max neighbor position delta (m) | Max neighbor speed delta (m/s) | Response |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | approach | 33 | 16.773 | 10.012 | 5.200 | Yes |
| 0 | internal_mid | 48 | 17.795 | 0 | 0 | No |
| 0 | downstream | 55 | 24.095 | 0 | 0 | No |
| 1 | approach | 33 | 16.872 | 11.928 | 7.000 | Yes |
| 1 | internal_mid | 50 | 16.010 | 0 | 0 | No |
| 1 | downstream | 58 | 23.210 | 0 | 0 | No |
| 100 | approach | 71 | 11.895 | 8.080 | 7.000 | Yes |
| 100 | internal_mid | 92 | 9.132 | 4.479 | 4.149 | Yes |
| 100 | downstream | 111 | 7.609 | 6.129 | 5.387 | Yes |

Maxima compare the same root-neighbor IDs at the same simulation times across
the ten candidate pairs. They are absolute motion differences, not ego-relative
gap differences. All zero-response roots are retained.

Unchanged screen: speed difference >= 0.05 m/s OR position difference >= 0.1 m.
Unchanged gate: all planned roots plus responses in at least two development
seeds. Actual: 9/9 roots, 5 response roots, response seeds=[0,1,100].
`status=complete`, `engineering_gate=true`, `mechanism_gate=true`,
`next_stage=P4_design_review`.

## Termination and robustness audit

The original terminal conditions are preserved. Common-time paired horizons:

| Seed | approach | internal_mid | downstream |
| --- | --- | --- | --- |
| 0 | 25 | 15–16 | 8 |
| 1 | 25 | 16–17 | 8 |
| 100 | 17–24 | 10–25 | 3–25 |

Of the 45 distinct branch rollouts, 20 reach original arrival, 8 terminate on
original collision and 17 remain nonterminal at the horizon. These are candidate
interventions with zero-jerk continuation, NOT episodes controlled throughout by
the author DDPG. They must not be quoted as the DDPG baseline's collision rate.
The repeated copies are not counted again.

All ten approach branches in seeds 0 and 1 remain nonterminal for the full
25-step horizon. Their neighbor responses cannot be artifacts of collision
termination. A separate read-only reanalysis excludes a paired frame whenever
either branch has terminated, then recomputes the same same-ID/time absolute
maxima. Every maximum in the response table remains unchanged. No report/gate
was overwritten; this is a post-run robustness check, not a revised criterion.

Root-actor membership mismatch observations summed across pairs are 10, 7, 7
at the three seed-100 roots; zero elsewhere. These are repeated pair/frame
observations, not independent events. Missing actors and terminal tails remain
censored. Unequal horizons constrain comparisons between roots.

## Interpretation against prior diagnostics

| Version | Reference and roots | Response roots | Response seeds | Gate |
| --- | --- | --- | --- | --- |
| v1 | Zero jerk; fixed steps 0/30 | 0/6 | None | Fail |
| v2 | Zero jerk; merge geometry | 3/9 | 100 | Fail |
| v3 | Pinned author DDPG; same merge geometry | 5/9 | 0, 1, 100 | Pass |

V2 established that the simulator can produce neighbor responses; v3 extends
that evidence to all three declared development seeds without changing the
environment. The previously sampled zero-jerk prefixes did not adequately cover
the interactions reached by the author policy. Changing the reference trajectory
also changes encounter timing, acceleration and relative geometry; this study
does not isolate which of those factors drives the response.

The P3 stop condition is resolved at the development-mechanism level. This does
not prove that B3 will outperform B2, that an ensemble is necessary, that labels
are sufficient for learning, or that the resulting controller is safer. Nine
roots from three episodes cannot establish generalization or statistical effect.

## Accepted backend and next boundary

Accept fresh_sumo_prefix_replay_v1 for bounded development labels, with saved
policy prefixes and exact checks. Native loadState remains ineligible after its
v1 continuation mismatch; v3 does not test or fix it. No native snapshot code
is silently promoted. Prefix replay is slower and limited to first-episode
roots here; it is not yet an efficient formal collector.

Proceed to P4 DESIGN REVIEW: freeze root/episode split units, prefix-policy
provenance, intervention/continuation labels, terminal masks, actor capacity
and omitted summaries, then implement a small pilot and model unit tests.
Retain both responsive and nonresponsive roots; do not filter by favorable
candidate outcomes. Ordinary and conditional predictors must share data splits,
history, horizon and capacity. Require P5 ranking/coverage acceptance before
expensive DDPG comparisons. Formal collection/training remain user-run.

Implementation and this acceptance record are separate local atomic commits.
P3 is eligible for a non-squash merge into main after final review; no model
performance success tag or push is warranted. Upstream license remains unresolved.
