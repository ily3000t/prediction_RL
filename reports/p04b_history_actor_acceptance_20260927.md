# P4b history/actor adapter acceptance — 2026-09-27

Status: bounded P4b engineering gate PASS. Full P4 and formal training readiness
remain incomplete. No new candidate future rollout, formal data collection,
predictor optimization or DDPG training was performed.

## Provenance

- Clean implementation commit: 88b4da5052e3a5f68fe6bbe3a1e503d6d92a686f.
- Plan: configs/development/p04_history_actor_v1.json.
- Run: artifacts/p4/p4_history_v1_01/report.json.
- Report SHA256: 89d4f73676a1fa6816421731cc30ee4d6f379d3e6d971879ccfdda20cfbb6c8c.
- Elapsed: 59.67 seconds; working_tree_dirty=false; 129 unit tests passed before
  the pilot. Three separate seed workers; original pytorch/SUMO environment.
- P3 and P4a source report hashes verified unchanged after the run. Upstream
  environment/control/reward/network and E:/WCDT_ACCVP were not modified.

The pilot uses the same seeds 0, 1, 100 and the same nine accepted P3 roots.
Each saved prefix is replayed twice: 18 short first-episode replays. No root is
selected using candidate outcomes; no failure or nonresponsive root is dropped.
Original DDPG initialization is reproduced before freezing RNG; control during
these replays uses the saved continuous jerk actions, not fresh policy inference.

## Results

Every root's signature, full original prefix trace hash and base traffic match
P3 exactly despite additional read-only metadata queries. Extended histories and
actor encodings match exactly on both repeats. Runtime lane lengths match the
pinned network. All roots have 11 observed frames spanning 2.0 seconds.

| Seed | Target | Prefix steps | Selected neighbors | Omitted neighbors | Active summary groups |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | approach | 33 | 12 | 3 | 1 |
| 0 | internal_mid | 48 | 12 | 4 | 1 |
| 0 | downstream | 55 | 12 | 5 | 1 |
| 1 | approach | 33 | 12 | 3 | 1 |
| 1 | internal_mid | 50 | 12 | 5 | 1 |
| 1 | downstream | 58 | 12 | 6 | 1 |
| 100 | approach | 71 | 12 | 8 | 1 |
| 100 | internal_mid | 92 | 12 | 10 | 2 |
| 100 | downstream | 111 | 12 | 12 | 2 |

There are 108 selected and 56 omitted root-neighbor occurrences across nine
roots, not 164 independent vehicles/scenarios. Every root has overflow beyond
12 physical neighbors; summaries retain observed aggregate information about
those omitted actors. This is not equivalent to retaining individual trajectories.

Output dimensions are fixed: neighbor history 12x11x10, ego history 11x10,
summary history 3x11x7, with actor/time/token/scalar masks. Multiple mandatory
roles referring to one actor are deduplicated. All five candidates of each root
share one input hash, preventing candidate-specific input membership differences.

Additional read-only checks after the pilot:

- All 99 saved history frames: sum of per-group omitted counts equals observed
  non-ego actors minus root-selected actors present at that frame.
- All selected actor IDs exist in each of the 45 joined candidate label packs.
- Label pack SHA256 values still match the accepted P4a report.

The label packs remain unmodified; new packs reference them by ID/path/hash.
The pipeline does not invent historical measurements from those future labels.

## Unit and boundary coverage

129 total unit/regression tests passed. New tests cover lane-chain continuity,
front-bumper/leader-length gap calculation, cross-path ETA versus same-path TTC,
empty actor and summary masks, short-history padding, historical absence,
root-disappeared actors, deterministic ordering/ties, mandatory overflow,
unknown lanes, invalid extents/nonfinite states, nonmonotone/gapped history and
input non-mutation. Low-density/padding and mandatory-overflow behavior are
synthetic tests; the nine real roots all have more than 12 neighbors and full
history, so real-world/general scenario coverage is not claimed.

## Important semantic limits

SUMO reference positions are front bumpers. The adapter uses a continuous
lane-chain coordinate ending at the common downstream lane start, not global x.
Same-path gap/TTC and cross-path merge-end ETA are simple observed-state proxies.
They are not oriented-box collision checks, risk calibration, the first physical
conflict surface or an action-conditioned safety guarantee. No Risk model or
Shield was added. The detailed definitions are in the design contract, with the
official SUMO retrieval reference.

Selection is root-only and candidate-independent. Dynamic actor IDs, role maps
and per-actor geometry metrics are audit metadata, not extra unbounded predictor
input channels. The predictor must consume the named fixed arrays and masks.
Omitted summaries are three dedicated adapter tokens; they are not vehicles
with fabricated trajectory supervision.

## Handoff

history_actor_gate=true; root_count=9; joined_candidate_count=45;
formal_training_ready=false; next_stage=P4c_predictor_interfaces.

Next implement ordinary/conditional predictor interfaces with identical scene
inputs, masked history/attention behavior, candidate batching and strict
three-member checkpoint save/load. Do not begin formal collection or optimization
until model/data/split configurations are reviewed. This development sample
cannot be promoted to independent validation or test evidence.

The P4b slice can merge independently, retaining atomic commits. Do not mark
all P4 complete, claim predictor improvement or publish unresolved upstream code.
