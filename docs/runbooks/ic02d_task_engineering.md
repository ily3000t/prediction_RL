# IC02d：三任务工程运行

## 当前状态：旧请求已失败，以下命令仅保留历史

`ic02d_eng_v1` 已尝试，worker 的残留文件 wildcard 错误匹配父进程 `started.json`，在 SUMO 启动前退出。实际 0 会话/控制步/分叉；错误已离线修复，399 项测试通过。原请求和失败不可覆盖，当前源码 commit 也已不同，**不要再次执行以下 run 或通过换名重试**。修复版新命名请求需要用户确认；详见 `reports/ic02d_engineering_preflight_failure.md`。

## 已实现的历史入口与预算

只在 `E:/Prediction_RL_worktrees_course_learning` 执行。先提交测试过的 collector，使当前 codex 分支干净；旧基线只读。预算固定且已获用户批准，不是完整课程池或训练。

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/collect_curriculum_tasks.py prepare --baseline-root E:\Prediction_RL
```

使用实际 prepare 输出的 canonical request hash（不是 request 文件字节 SHA）执行：

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/collect_curriculum_tasks.py run --request artifacts/ic/ic02d_eng_v1/request.json --confirm-request-hash <prepare输出哈希>
```

串行三个 worker，train seeds 11000/11008/11016，每 task 最多两次发现、三个检查点各两次捕获和十次分叉。包括重启和前缀的完整上限：114 个 SUMO 会话、12,234 控制步、23,862 raw tick、90 分叉；不是只计候选未来步。没有 B2/B3 推理或训练，没有 DDPG。

结果不可覆盖：顶层 `report.json` 或 `failure.json`，父进程 `postflight.json`；各 task 有 discovery/repeat、root history/labels/候选与重复 sidecar/evidence、全部原始 session tick、worker postflight、complete receipt 和日志。参考作者权重只用于发现道路检查点，参考前缀不成为未来训练初始化。

`--resume` 仅在顶层还无 complete/failed 报告时，复用同 hash/source 下完整校验的 task；开始但不完整的 task拒绝重跑。不要直接调用 worker、更换 run ID 或重置 receipt。若失败，保留目录，先分析原因和已消耗预算，不自动重新采集。

完成后检查：重复发现/历史/候选 exact；实际总计符合上限；worker tensor/RNG/files 和父进程 RNG/files 收尾有效；unavailable/terminal 支持如实保留。I 仅用旧尺度描述，不估新阈值，不赋 task 层；`training_ready=false`、pi=null。下一步分析与剩余 29-task pilot 需要独立批准。
