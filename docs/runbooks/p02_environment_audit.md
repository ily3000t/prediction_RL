# P2 bounded environment audit

Run in the existing pytorch environment. No package installation is needed.
Upstream code, network files, SUMO and compiled st_cy must remain available.

~~~powershell
conda activate pytorch
Set-Location E:/Prediction_RL
$taskTemp = 'E:/Prediction_RL/artifacts/pytest_p2_' + [guid]::NewGuid().ToString('N')
python -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
python tools/audit_environment.py --run-id local_p2_01
~~~

Choose a new short run ID for every invocation. Existing output folders are
rejected. Run serially without another training/SUMO process in this workspace.

The fixed audit starts eight independent workers, totaling twelve episodes.
Each episode is capped at 500 policy steps; many terminate much earlier.
Workers have a 120-second wall-time diagnostic watchdog. This can mark a test
invalid and abort it; it NEVER changes policy observations or serves as a
deployment timeout. No optimizer or model checkpoint is used.

The preserved reference config is:
RL-MPC-LaneMerging-master/configs/train_default_1.json.
Explicit diagnostic seeds are 0, 1 and 100, not selected based on performance.
Each worker writes a full resolved configuration; no config inheritance is
introduced and unknown upstream settings are rejected by the session loader.

The test matrix is fixed:

| Case | Seed | Jerk sequence | Episodes |
| --- | --- | --- | --- |
| raw_stream / adapted_stream / repeat_stream | 0 | constant +5 | 2 each |
| other_seed | 1 | constant +5 | 2 |
| raw_limit / adapted_limit | 0 | constant -5 | 1 each |
| raw_mixed / adapted_mixed | 100 | repeating -5,-2.5,0,2.5,5 | 1 each |

Outputs under artifacts/p2/<run-id>/:

- report.json: source identity, plan, child summaries and all eight checks.
- working_tree.diff and source_snapshot/: retain dirty development source.
- <case>.log: worker stdout/stderr.
- <case>/resolved_config.json: full upstream settings used.
- <case>/traces.json: observations, rewards, flags and traffic states.
- <case>/execution_audits.json: requested versus measured execution for adapters.
- <case>/report.json: seed, steps, config/traces hashes and runtime versions.

Development dirty runs retain source evidence. The accepted rerun is
artifacts/p2/p2_clean_01/report.json, launched from clean commit 1e45e3c.
Reports are immutable. Do not overwrite a failure or change seeds to pass.
These are semantic regression checks, not model-effectiveness experiments.

See docs/design/p02_environment_contract.md for reset/termination semantics.
Next is P3 snapshot branching and neighbor-response diagnostics; no predictor
or DDPG training command is authorized by this audit.
