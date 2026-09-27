# P3 v2 merge-region diagnosis — 2026-09-27

Status: engineering replay PASS; mechanism coverage gate FAIL. P4 remains paused.
Unlike v1, v2 observes action-dependent neighbor motion, but only in one of the
three fixed development seeds. This establishes a local simulator response,
not broad coverage, predictor utility or closed-loop improvement.

## Provenance and unchanged conditions

- Clean source commit: `78d4bb14b1af851ae6924ae3f38e23b0f13f1977`.
- Branch: `codex/p03-counterfactual-diagnostics`; main remains P2 `1df4595`.
- Plan: `configs/development/p03_mechanism_v2_merge_region.json`.
- Plan SHA256: `c842bfe85b4b5054a066e8ad35c8528dfd5940d756cd6c489c85f5674397abdf`.
- Evidence: `artifacts/p3/p3_merge_v2_01/report.json` and per-seed traces.
- Report SHA256: `07027d5537750bc0b17492e25ca800c7900f1c4f8559cb4b7b17f43d2bc17563`.
- One bounded run, 249.32 seconds; `working_tree_dirty=false`.
- 77 unit/regression tests passed before the run.

Original author `train_default_1.json`, reward, DDPG interface and SUMO controls
are unchanged. Seeds remain 0, 1, 100. Five probes remain -5, -2.5, 0, 2.5,
5 m/s^3, derived from actual upstream bounds. Each acts for one original
0.2-second step, followed by zero jerk, over at most 25 steps (5 seconds).
Zero jerk after the intervention can preserve nonzero acceleration; it is not
a constant-speed continuation. Original terminal conditions still apply.

No training, formal collection, old-project edit, package change or push.
Raw results are ignored by Git. Native save/load is disabled in v2 because
the previous continuation parity failure is unresolved. Exact fresh-process
prefix replay remains the only trusted development backend.

## Result-independent root selection

Before evaluating any probes, follow one zero-jerk reference trajectory and
select the first nonterminal state reaching each lane-position threshold:

| Target | Actual SUMO lane | Threshold |
| --- | --- | --- |
| approach | ramp_0 | 20 m before lane end; position 181.92 m |
| internal_mid | :mergenode_1_0 | half of 52.18 m; position 26.09 m |
| downstream | highwayahead_0 | 5 m from lane start |

Discovery is capped at 250 original steps. No response, reward, collision or
success outcome enters root selection. Missing targets must be reported, not
replaced. All nine planned targets were reached. Discrete-step overshoot is
recorded rather than repositioning the ego vehicle.

| Seed | Target | Prefix steps | Actual lane position (m) | Overshoot (m) |
| --- | --- | ---: | ---: | ---: |
| 0 | approach | 30 | 182.922 | 1.002 |
| 0 | internal_mid | 40 | 28.642 | 2.552 |
| 0 | downstream | 46 | 5.046 | 0.046 |
| 1 | approach | 31 | 183.355 | 1.435 |
| 1 | internal_mid | 41 | 27.678 | 1.588 |
| 1 | downstream | 48 | 7.869 | 2.869 |
| 100 | approach | 114 | 182.527 | 0.607 |
| 100 | internal_mid | 151 | 26.865 | 0.775 |
| 100 | downstream | 176 | 5.941 | 0.941 |

## Results

All selected roots exactly match their discovery traffic after prefix replay.
At every root, all five probes repeated in reverse order match exactly,
including observations, rewards, traffic and execution audit. There are
45 distinct candidate rollouts plus 45 exact repeats, not 90 independent scenes.

The table reports maxima over the ten probe pairs, shared root-actor IDs and
aligned simulation times. Neighbor quantities are absolute motion differences.

| Seed | Target | Max ego position delta (m) | Max neighbor position delta (m) | Max neighbor speed delta (m/s) | Response |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | approach | 22.080 | 0 | 0 | No |
| 0 | internal_mid | 8.400 | 0 | 0 | No |
| 0 | downstream | 3.600 | 0 | 0 | No |
| 1 | approach | 22.080 | 0 | 0 | No |
| 1 | internal_mid | 9.600 | 0 | 0 | No |
| 1 | downstream | 2.880 | 0 | 0 | No |
| 100 | approach | 26.002 | 9.480 | 7.000 | Yes |
| 100 | internal_mid | 23.990 | 13.728 | 5.209 | Yes |
| 100 | downstream | 24.000 | 14.042 | 5.214 | Yes |

Unchanged response screen: position delta >= 0.1 m OR speed delta >= 0.05 m/s.
Unchanged mechanism gate: all planned roots evaluated AND response in at least
two development seeds. Actual: 9/9 roots evaluated, 3/9 with response, response
seeds=[100]. Therefore `status=complete`, `engineering_gate=true`,
`mechanism_gate=false`, `next_stage=pause_for_mechanism_diagnosis`.
Three roots from the same reference episode do not count as three independent
seeds. These small development counts are not statistical effect estimates.

## Trace checks and censoring

At seed 100 / approach, after 20 steps, the same vehicle `traffic_18400`
travels at 7 m/s under the -5 probe but has stopped under the zero probe.
The ego is still on the ramp in the former branch and on the internal merge
lane in the latter. Both branches remain nonterminal over all 25 steps.
This difference cannot be explained solely by ego-relative coordinates or
unequal termination. It is a measured simulator neighbor response; its exact
SUMO right-of-way/car-following cause has not been separately isolated.

For the internal and downstream roots, the -5 versus +2.5 pair gives neighbor
speed differences of 5.209 and 5.214 m/s respectively at step 25. Both branches
in each cited pair remain nonterminal. Thus the response conclusion does not
depend on the separate +5 internal branch, which collides at step 25, or the
+5 downstream branch, which arrives at step 25. Those outcomes remain recorded.

Paired common-time horizons, in steps:

| Seed | approach | internal_mid | downstream |
| --- | --- | --- | --- |
| 0 | 24–25 | 15–16 | 10 |
| 1 | 24–25 | 16–17 | 9–10 |
| 100 | 25 | 25 | 25 |

The seed-100 internal root has four root-actor membership mismatch observations
summed over probe pairs (not four independent vehicles or events); all other
roots have zero. Absent actors and unequal tails are censored, never filled
with zero motion. Fast-ego later roots have shorter observation windows due
to original termination, limiting direct comparisons of response opportunities.

## Interpretation and next decision

V1 observed 0/6 response roots; v2 observes 3/9. In v1, seed 100 was sampled
too early to reach the merge in the prediction window. Geometric coverage
reveals responses without altering traffic, reward, seeds or thresholds.
This rules out the blanket interpretation that this environment never lets
neighbors respond to ego actions.

It does NOT establish sufficient interaction diversity for B3 training.
The reference ego speed is about 23.82/23.12 m/s in seeds 0/1, versus 6.25 m/s
in seed 100, while the author config caps normal traffic at 7 m/s. Speed and
relative traffic geometry, together with terminal censoring, are plausible
coverage explanations, not separately identified causes.

Recommended next review: a separately versioned, bounded coverage diagnostic
using the author-provided DDPG reference policy, retaining the same three seeds,
geometric targets, probes, horizon and thresholds. Freeze checkpoint hash,
deterministic inference and prefix generation before running; report every
missing/negative root. This asks whether the reference control policy explains
weak interaction coverage. It has NOT been implemented or run here and requires
approval of that P3 scope change. Do not search checkpoints or keep only favorable
scenarios. Author moderate/fast traffic configs would be another independent
environment study, not a silent substitution.

Do not enter P4 or train a larger predictor to compensate for this gate failure.
If declared broader coverage still lacks diverse responses, reconsider whether
ordinary neighbor prediction plus analytic candidate ego geometry is sufficient;
that would change the B3 research claim and needs an explicit design decision.

P3 branch remains unmerged; no success tag and no push. V1 config and evidence
are preserved unchanged. See the runbook for exact reproduction commands.
