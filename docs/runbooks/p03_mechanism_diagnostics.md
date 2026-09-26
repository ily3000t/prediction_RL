# P3 bounded diagnostic runbook

Current result: mechanism gate failed. This command reproduces the bounded
diagnostic, not a recommendation to repeatedly rerun until a favorable result.
Do not start a formal dataset or predictor training.

~~~powershell
conda activate pytorch
Set-Location E:/Prediction_RL
$taskTemp = 'E:/Prediction_RL/artifacts/pytest_p3_' + [guid]::NewGuid().ToString('N')
python -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
python tools/diagnose_branches.py --config configs/development/p03_mechanism_v1.json --run-id local_p3_01
~~~

Use a fresh short run ID. The script rejects existing output folders.
The standalone human-maintained config contains the complete diagnostic plan;
its upstream_config entry identifies an original reference input, not an
extends hierarchy. Worker resolved_config.json records all upstream defaults
plus explicit run seed and OS setting. Unknown plan fields are rejected.

Budget: 3 seeds, 2 fixed first-episode roots each, 5 probes repeated in reverse
order. Each of 60 candidate rollouts is capped at 25 steps. Prefix replays and
SUMO warmup add work. The native snapshot test runs at one root. Worker watchdog
is 300 seconds; it aborts a diagnostic, never changes observations.

Outputs: artifacts/p3/<run-id>/report.json, resolved_diagnostic_config.json,
source_snapshot/, working_tree.diff, seed_<n>.log, and per-seed report/config.
root_<steps>_branches.json retains original and reversed branch traces,
censoring counts and absolute response comparisons.
seed_0/native_probe/ retains sumo.xml and direct/restored comparison.

Current evidence: artifacts/p3/p3_dev_01/report.json.
Expected status for that result: complete; engineering_gate=true;
mechanism_gate=false; next_stage=pause_for_mechanism_diagnosis.
Do not judge success by process exit code alone: a negative valid experiment
is supposed to aggregate successfully.

For a new coverage design, create a new version after review; do not overwrite
p03_mechanism_v1.json or prior results. No next-stage training CLI exists yet.
