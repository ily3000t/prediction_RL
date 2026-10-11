# IC02e：剩余 29 个训练侧任务刻画

用户在三任务工程通过后，批准固定剩余 29-task 的命名请求。种子是原步长 8 清单的 11024–11248；已完成的 11000/11008/11016 只读复用，不重跑。原配置和历史报告不改。新授权在 `provenance/ic02e_pilot_authorization.json`，独立无嵌套配置为 `configs/development/ic02e_remaining_task_pilot_v1.json`。

本阶段不改变科学采集语义，直接调用 IC02d 的 TaskBrancher/collect_task 与 sealed-evidence 验收；核对工程报告、receipt、科学源码及资产。最多 1,102 新会话、118,262 控制步、230,666 raw ticks、87 root 槽位、870 分叉，包含发现、所有前缀重放与等待。单 worker、Torch 单线程，无 B2/B3 推理或训练、无 learner/DDPG，旧项目只读。

在工作树干净且原子提交之后，于 `E:/Prediction_RL_worktrees_course_learning`：

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/collect_curriculum_pilot.py prepare --baseline-root E:\Prediction_RL
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/collect_curriculum_pilot.py run --request artifacts/ic/ic02e_pilot_v1/request.json --confirm-request-hash <本次prepare输出哈希>
```

产物为独立 `artifacts/ic/ic02e_pilot_v1/`，request/config/source/环境/receipt/hash 完整绑定。report/failure 不可覆盖；`--resume` 只接受未终结父请求下已完整验证的 task，started 无完整 receipt 不能自动重跑。任何实际失败保存 postflight、预算与 traceback 后停止，不换 seed、不自动补采。原生终止、不可达与不完整未来如实保留，工程预算 cap 不是新 done。

完成后组合原三任务和新增 29 项进行纯离线覆盖分析。32 个任务不是 96 个独立 root 或 960 个独立分叉样本。保持 classifier=null、pi=null、training_ready=false；不能直接套旧 root 阈值定义 task，不能因为初态 I=0 归 C0，也不能按 B2/B3 表现分层。任务规则、原目标分布与调度仍需后续冻结和确认；本阶段不批准训练、合并、推送、打 tag。
