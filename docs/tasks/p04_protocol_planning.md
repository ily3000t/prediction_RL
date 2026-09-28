# P4e planning slice

Goal: expose dataset/training decisions before expensive collection.
Prerequisites: accepted P3 and P4a–d engineering checks, clean committed source.

Allowed: independent draft config, strict schema/seed checks, deterministic job
and episode manifests, nonexecuting upstream-default resolution, tests and docs.
Forbidden: changing upstream reward/control, old-project writes, data/model runs,
self-approval, test access, silent unknown-field ignoring or publication.

Acceptance: no split overlap/development promotion; stable episode identity under
protocol rename; deterministic short job IDs and budget counts; no authorized
jobs; locked test release; invalid configs rejected. Resolve original reward/
action settings without executing config.py. Plan validity is not training readiness.

Artifacts: configs/formal/*_draft.json, ignored artifacts/p4 planning outputs,
small acceptance report in reports. Feature branch codex/p04-protocol-planning;
code commit before clean-tree planning, separate acceptance commit afterward,
non-squash local merge after checks. No remote push or runnable milestone tag.

Pause for user review of budgets, sampling, selection and calibration. Do not
silently implement an executor that runs this draft. The handoff command and
review procedure are in docs/runbooks/p04_protocol_review.md.
