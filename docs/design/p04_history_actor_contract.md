# P4b history and dedicated actor adapter — development v1

This layer is independent of upstream networks and the old ACCVP/WcDT project.
It consumes only observed root/past traffic; there is no candidate outcome,
future trajectory or reward argument. The same root input is shared by all five
candidate probes and will be shared between B2 and B3.

## Read-only history

capture_with_history replays recorded original actions from reset, adding
read-only lane position, length and width queries after each step. It retains
at most 11 frames including the root: 2.0 seconds at 0.2-second spacing.
The original base trace is hashed separately, so extra reads must reproduce
the exact P3 root signature, traffic and prefix trace hash. Terminal roots are
not replaced. The bounded pilot replays each of nine saved roots twice, checks
history/encoding equality and joins the existing P4a labels; no candidate future
is simulated again and no new scenario/root is selected.

Short history is left-padded with zeros and false masks. Missing historical
actors are not copied backward. A vehicle gone by the root cannot become a
physical slot, but its observed past can appear in past omitted summaries.
No post-root time is consumed. Formal multi-episode collection is not implemented.

## Geometry and limits

The five-lane upstream network is pinned by SHA256. Lane lengths and connectivity
are parsed from merge.net.xml and runtime lengths are checked against TraCI.
Unknown lanes and invalid extents/positions cause explicit errors.

SUMO position and lane position use the front bumper, not vehicle center. See
[official retrieval semantics](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html).
The adapter defines a lane-chain coordinate with zero at highwayahead_0 start:
upstream lane lengths are subtracted, downstream position is positive. This
coordinate is continuous along each route; it is NOT a two-dimensional collision
surface or the first point of lateral overlap in the merge junction.

Compatible paths are the same approach group, or an approach and the shared
downstream. Longitudinal bumper gap subtracts the leading vehicle length;
CV TTC uses that gap and positive closing speed, with zero for overlapping
longitudinal extents. These remain constant-speed route-chain proxies, not
guaranteed collision times. Different upstream approaches get NO same-path gap
or TTC. Their constant-speed times to the common merge endpoint supply an ETA
difference proxy. Upstream stopped actors and downstream actors have no positive
future endpoint ETA. Undefined metrics use None/masks, never an arbitrary zero.

No oriented-box surface distance, Risk model, Shield or safety guarantee is
introduced. Front-bumper Euclidean distance is only a final deterministic
priority tie-breaker and is explicitly named as such.

## Root-only selection

K=12 counts physical NEIGHBORS; ego is a separate sequence. Summary tokens do
not consume physical vehicle slots. Roles are selected in this stable order:

1. Target main-path front, by nonnegative merge-coordinate difference.
2. Target main-path rear, by closest negative difference.
3. Nearest cross-path ETA difference, when both endpoint ETAs are defined.
4. Lowest defined same-path CV TTC.
5. Earliest defined neighbor arrival at the merge endpoint.

IDs are deduplicated; equal metrics use lexical vehicle ID. Mandatory actors
exceeding capacity raise MandatoryOverflow. Remaining slots sort by endpoint
ETA, defined CV TTC, absolute merge-coordinate difference, front-bumper point
distance and ID. Undefined priority metrics sort last. This is a fixed
candidate-independent proxy selector, not a learned or per-candidate selector.

## Tensor layout and masks

| Field | Shape | Meaning |
| --- | --- | --- |
| ego_features | 11 x 10 | Separate ego history |
| actor_features | 12 x 11 x 10 | Stable root-neighbor slots |
| actor_history_mask | 12 x 11 | Actor observed at that past time |
| actor_mask | 12 | Valid physical slots at root |
| summary_features | 3 x 11 x 7 | Omitted observed actors by group/time |
| summary_history_mask | 3 x 11 | Group nonempty at that time |
| summary_feature_mask | 3 x 11 x 7 | Each scalar defined |
| summary_mask | 3 | Group nonempty at root |

Physical features: XY relative to fixed root ego origin, speed, acceleration,
merge-end progress, length, width and three group one-hot fields. Raw SI units
are retained; train-only normalization is a later model concern.
Dynamic IDs, mandatory role maps and root_actor_metrics are audit metadata,
not unbounded additional model channels. A predictor should consume only the
named fixed arrays/masks; no future label pack is accepted by this adapter.

Summary groups: ramp (including ramp internal lane), main approach (including
main internal lane), main downstream. At each history time, summarize observed
vehicles not in the root-selected physical IDs; never include ego. Features:
count, minimum defined same-path bumper gap, minimum defined CV TTC, minimum
defined endpoint ETA, mean relative speed, maximum absolute relative speed,
and global neighbor-capacity-overflow indicator for that time. Undefined minima
are zero with false scalar masks. Empty groups have all-zero features and false
token masks. A root-empty group cannot attend as a current actor even if a past
group was present. Summary counts preserve observed omitted vehicles, not their
full individual trajectories and not knowledge of future arrivals.

Future actor labels must later be gathered by these stable IDs; omitted summary
tokens are not vehicles and have no invented individual trajectory target.

## Stage boundary

This is a bounded development input contract. History/actor gate success does
not imply dataset coverage, predictor convergence or B3 advantage. All existing
roots remain development-only and formal_training_ready=false. Next is P4c
ordinary/conditional model interfaces, masked sequence/attention tests, three
member checkpoint save/load and batched-vs-single-candidate equality. No formal
collection/training until split/config review; P5 still guards DDPG expenditure.
