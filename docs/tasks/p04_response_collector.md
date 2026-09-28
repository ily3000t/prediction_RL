# P4f bounded response collector

Goal: implement and verify observed history + five candidate responses with a
future lane-position/vehicle-size sidecar and immutable episode reuse.
Prerequisites: P3 exact replay and P4a-d accepted development artifacts.

Allowed: project data helpers, development-only CLI/config, tests and docs.
Forbidden: old-project/upstream changes, executing the P4e draft, new seeds,
formal collection or training, rewards/action changes, native snapshot fallback.

Acceptance: three existing seeds, nine existing roots, 45 unique candidates and
90 repeated branches. Independently rediscover roots; exact original transitions,
history/inputs/labels and P4d tensor fingerprint; repeat future metadata exactly.
Test missing-root accounting, immutable writes, exclusive locking, failed episode
preservation and checksum-verified no-simulation resume.

Outputs: ignored artifacts/p4/<short-id>, one immutable request, e000..e002
receipts/data, append-only invocation reports/logs; tracked small acceptance only.
Git: codex/p04-response-collector; source/test commit before SUMO acceptance,
separate evidence/docs commit then non-squash merge. No push/tag.

Pause on any parity or geometry mismatch. Do not replace a seed, omit a root,
weaken a hash check or overwrite failures. Formal execution remains locked until
protocol review and a separate production executor stage.

Handoff: see docs/runbooks/p04_response_collector.md after acceptance.
