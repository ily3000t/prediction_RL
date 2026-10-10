# IC01：纯离线训练侧交互核查

状态：该入口只分析旧 train 数据，不生成新 SUMO 分支，不运行预测器/DDPG，也不发放课程训练许可。先通过单元测试并提交分析源码/配置，再运行不可覆盖的审计。

## 分析前固定的开发规则

- 关键车辆：root 时刻的目标前/后车、5 秒内的最近跨路径到达车辆、5 秒内的最低同路径 TTC 车辆；采用原 actor adapter 的根状态角色及 125 m 传感范围，不看分支收益或方法胜负。
- 使用已审计五条 lane 的公共道路进展坐标，以 `highwayahead_0` 起点为零；不是候选 ego-relative 坐标或局部 lanePosition。
- I：所有共同有效的关键邻车单元等权平均 `abs(delta_v)/c_v + abs(delta_s)/c_s`。只看邻车，终止帧不计入，不补零。
- 归一化：train 中正的 root 平均响应量的中位数；某通道没有正支持时仅用单位尺度便于展示，并明确记录 fallback，不把它当成学到的尺度。
- 支持：候选对至少共同观察 5 个 tick，即 1 秒；至少 80% 候选对满足该条件才允许暂归 C1。另报告完整 5 秒支持，不能把短窗口结论扩展为全时域保证。
- 原 collection 要求重复分支逐字节相同。重新核验每个实际重复文件的 SHA256；相同时同一网格差为零，是已验证的重复噪声观测，不是重新运行仿真的结果。若旧重复证据不完整或改变，工程审计停止。
- 分层边界只用 train root 的上下三分位；不强迫各层样本数相同。C0 需低 I、宽松 D 且完整共同时域；C1 需高于重复噪声的 I、合格支持和至多 3 个关键 actor；其他保留 C2。
- 每层至少 10 个不同源 episode 的非退化检查只是本轮开发提醒，不是正式统计样本量要求。root 分层尚不是 episode 任务分层，原目标任务分布 pi 不从这些 root 的比例推断。

## 测试

```powershell
cd E:\Prediction_RL_worktrees_course_learning
New-Item -ItemType Directory -Path artifacts/ic_tests -Force | Out-Null
$icTestBase = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) ('artifacts/ic_tests/verify_' + [guid]::NewGuid().ToString('N'))))
if ((Test-Path -LiteralPath $icTestBase) -or -not $icTestBase.StartsWith((Join-Path (Get-Location) 'artifacts\ic_tests\'), [System.StringComparison]::OrdinalIgnoreCase)) { throw '测试目录已存在或不在预期范围' }
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B -m pytest -q -p no:cacheprovider --basetemp $icTestBase tests/test_curriculum_interaction.py tests/test_actor_adapter.py tests/test_dataset_contract.py tests/test_response_collection.py
```

## 分析（工作树必须干净）

以下是已完成 `ic01_v1` 的实际调用。不要再次使用相同 run ID；仅在明确需要重新审计时选择新的短 run ID，旧结果保留。

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/audit_curriculum_interaction.py --baseline-root E:\Prediction_RL --run-id ic01_v1
```

同一 run ID 不覆盖、不自动重试；失败保存 failure.json。更正分析实现后另建短 run ID，并保留失败产物。

## 产物

位于本工作树的 `artifacts/ic/<run-id>/`：

- `started.json`：完整解析配置、commit、环境、命令及来源。
- `input_inventory.json`：实际读取并验证的文件指纹；分支未来只访问 train。
- `roots.json`：关键 ID/角色、D、I、共同支持、噪声、截尾及暂定层。
- `task_candidates.json`：root-task 映射候选，不是可执行训练池；含不可达 root 和未确定的 pi。
- `report.json`：所有源场景完整汇总，工程 gate 与训练许可分开。层退化或响应不足也是正常完成的结果。

先检查 C1 响应是否真实、C0/C1 是否足够及支持缺口，再进入 IC02。无论本次数据结果如何，`training_ready` 都保持 false，直到任务入口与完整任务分布已冻结并经用户确认。

## 已完成的 IC01 验收

- 分析源码提交：`9dc84f556a14a596884d3722fcf325dfde32dec2`。
- 256 个 train episode、713 个已到达 root，另有 55 个参考前缀未到达的 root，单独保留原因。
- C0/C1/C2 = 0/211/502；工程 gate 为 true，但训练许可为 false。
- 485 个 root 有正邻车响应；重复分支逐字节一致，观测噪声下限为零。未运行新仿真或预测器推理。
- 105 项单元及既有 actor/标签/采集回归测试通过。
- 下一步先确认任务入口与覆盖，不放宽边界补出 C0、不启动 DDPG。

完整验收及限制见 `reports/ic01_training_interaction_acceptance.md`。C1 的 root 数量不能直接用作任务数量，root 的层比例不能用作原目标任务分布 pi。

下面的只读检查可复核 C0 的两个独立缺口，不会重新计算阈值或覆盖报告：

```powershell
$icRows = @(Get-Content -LiteralPath artifacts/ic/ic01_v1/roots.json -Raw | ConvertFrom-Json)
$icReport = Get-Content -LiteralPath artifacts/ic/ic01_v1/report.json -Raw | ConvertFrom-Json
$icBound = $icReport.summary.classification.thresholds
[pscustomobject]@{
    long_distance_and_low_closing = @($icRows | Where-Object {
        $_.root_context.difficulty.remaining_merge_distance_m -ge $icBound.remaining_merge_distance_m -and
        $null -ne $_.root_context.difficulty.maximum_closing_mps -and
        $_.root_context.difficulty.maximum_closing_mps -le $icBound.maximum_closing_mps
    }).Count
    low_response_and_full_horizon = @($icRows | Where-Object {
        $null -ne $_.I -and $_.I -le $icBound.I_low -and $_.interaction.full_common_horizon
    }).Count
}
```

两项实际输出均为零；这说明问题不只是单个门槛略微偏严，而是现有 root 采集位置、状态关联与支持窗口不能直接构成所提议的基础任务层。
