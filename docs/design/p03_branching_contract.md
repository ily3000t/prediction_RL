# P3 diagnostic branching contract

## Trusted development restoration

ReplayBrancher is constructed in a fresh upstream_session, before an open
SUMO connection. It freezes run-start RNG/delay and the resolved settings hash.
A ReplayRoot stores an immutable prefix recipe and signatures, with a manager
identity guard. Root objects are in-memory and cannot be used across managers.

For every candidate:
1. Close the previous owned environment.
2. Restore run-start Python/NumPy/torch RNG and traffic delay.
3. Construct a fresh unchanged upstream environment.
4. Replay reset plus the entire first-episode action prefix.
5. Require exact prefix trace and root signatures before executing the probe.

Root signatures include traffic, relevant raw Python state/history,
wrapper counters, RNG and settings. Wall-clock start_time is deliberately
excluded because it is not simulator state. Signatures do not certify hidden
SUMO internal state; replay continuation tests provide an additional check.

This backend does not implement arbitrary multi-episode snapshot restoration,
persistent Python snapshot deserialization or optimized data collection.
No pickle file is loaded. Metadata and root recipes are auditable; artifacts
remain outside Git.

## Native probe

native_snapshot_probe saves SUMO with RNG and high-precision options, saves
Python/environment state in memory, then compares uninterrupted and restored
continuations. It is experimental and never auto-promoted as a data backend.

Observed failure at the first tested native root means save/load success and
equal visible root observations are insufficient. Pending controls and hidden
vehicle state need separate investigation. Keep native_eligible=false.
Do not relax trace equality or compensate with changed driving constraints.

## Intervention and comparison

Candidate values come from actual action bounds using fractions in the plan.
Candidates act for one original decision period; a predeclared common zero-jerk
continuation follows. No candidate snapping of a learned policy, no mask and
no shield are involved. This is a simulator intervention under a specified
continuation, not an identified real-world causal effect.

Every root evaluates the same probes in forward and reverse order. Root
restoration and branch traces must match exactly before interpreting response.
Root actors are paired by vehicle ID and exact simulation time. Use their
absolute coordinates/speeds; changes in ego-relative gaps alone do not qualify.
Unequal terminal tails and missing actors are censored, not filled with zeros.

The mechanism gate is independent of engineering validity. A valid experiment
with no neighbor response finishes status=complete, mechanism_gate=false.
Do not turn a negative mechanism result into a code error or delete it.

## Historical coverage and current stage state

P3 v1 has valid replay diagnostics but no observed neighbor response across
its six roots. V2 adds first nonterminal lane-position threshold crossings
along the fixed zero-jerk reference; selection finishes before probe evaluation.
Every selected root must match discovery traffic after replay. Terminal or
unreached targets remain unavailable, with no substitution. The selector is
independent of response/reward outcomes; original actor/control settings remain.

V2 evaluates all nine targets, with response at three roots in only seed 100.
At that version, the predeclared gate still required two seeds, so P4 was paused. This is
limited coverage, not proof of an absent mechanism. See
reports/p03_merge_region_diagnosis_20260927.md. Native save/load remains disabled.

V3 replaces only the discovery prefix controller with a hash-pinned author DDPG.
It retains GreedyAgent and TimeFeature, validates every action against original
get_control, verifies no RNG consumption and repeats discovery independently.
Saved continuous jerk prefixes, not fresh policy calls, are replayed for every
candidate. After reaching a root, the candidate and zero-jerk continuation are
unchanged. No ST takeover or new policy action space is introduced.

V3 passed: 9/9 roots, 5 response roots in three seeds; all discovery/branch parity
checks exact. P4 design review can proceed. This is not model effectiveness or
formal data sufficiency. See reports/p03_author_policy_acceptance_20260927.md.
