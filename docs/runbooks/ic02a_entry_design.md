# IC02a 离线任务入口核查

本入口只读原源码、冻结资产指纹及已有 IC01 记录；不导入 SUMO，不运行模型，不实现新 reset。设计里的 smoke 是**未批准提案**，本脚本不会执行它。

## 单元及回归测试

```powershell
cd E:\Prediction_RL_worktrees_course_learning
$icEntryTestParent = 'E:\Prediction_RL_worktrees_course_learning\artifacts\ic_tests'
New-Item -ItemType Directory -Path $icEntryTestParent -Force | Out-Null
$icEntryTestBase = [System.IO.Path]::GetFullPath((Join-Path $icEntryTestParent ('entry_' + [guid]::NewGuid().ToString('N'))))
if ((Test-Path -LiteralPath $icEntryTestBase) -or -not $icEntryTestBase.StartsWith($icEntryTestParent + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected test target' }
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B -m pytest -q -p no:cacheprovider --basetemp $icEntryTestBase tests/test_curriculum_entry_design.py tests/test_curriculum_interaction.py tests/test_actor_adapter.py tests/test_dataset_contract.py tests/test_response_collection.py
```

## 离线核查（需干净工作树）

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/audit_curriculum_entry.py --baseline-root E:\Prediction_RL --run-id ic02a_v1
```

该 run ID 只能使用一次，不覆盖旧产物。完成后查看本工作树：

- `artifacts/ic/ic02a_v1/report.json`：源码事实、设计、输入哈希、有限 smoke 的预算和待确认决定。
- `artifacts/ic/ic02a_v1/episode_mapping.json`：旧 train episode 与 root 的关联；不是新的可执行任务池。

`static_inspection_gate=true` 仅意味着静态源码及映射检查通过。`runtime_parity_verified`、`new_entry_implemented`、`new_entry_approved` 与 `training_ready` 仍为 false。

## 下一步

请先审核 `docs/design/curriculum_task_entry_v1.md`：拟新增 `seeded_fresh_gym20_task_v1` 与最多 12 个工程 episode/144 个控制步的 smoke。批准后另建明确执行请求；当前不要修改配置的 authorized 字段，不要运行旧训练入口或猜测不存在的 smoke 命令。
