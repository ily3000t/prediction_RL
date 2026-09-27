# P4a offline label audit

This converts existing P3 v3 traces; it does not run SUMO or train a model.
Source evidence is pinned by configs/development/p04_label_audit_v1.json.

~~~powershell
conda activate pytorch
Set-Location E:/Prediction_RL
python tools/audit_dataset_labels.py --config configs/development/p04_label_audit_v1.json --run-id p4_labels_local_01
~~~

Use a clean committed checkout and a fresh short run ID. Existing directories
are rejected. Output: artifacts/p4/<run-id>/report.json, resolved_config.json,
split_manifest.json, index.json and r00.json... per-root candidate label packs.
All source traces are read-only. Labels, raw traffic and logs stay outside Git.

Check status=complete and label_contract_gate=true. training_ready MUST remain
false: P3 roots do not include the continuous history required by the encoder,
the fixed actor adapter is not implemented and formal splits are not approved.
Do not use this fixture as a formal training/test dataset or count repeats twice.

See docs/design/p04_data_contract.md for masks and the next implementation slices.
