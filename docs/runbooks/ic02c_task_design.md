# IC02c：纯离线任务设计审计

## 已实现入口

仅在 `E:/Prediction_RL_worktrees_course_learning` 执行。先审查并提交代码/配置/文档，使当前 codex 分支干净；基线 `E:/Prediction_RL` 保持原 main/HEAD 干净。

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/audit_curriculum_task_design.py --baseline-root E:\Prediction_RL --run-id ic02c_design_v1
```

命令只读冻结 baseline 引用、训练侧旧报告和已封存入口 trace，不启动 simulator、加载模型、推理或训练，不打开 test。输出目录必须新建且不可覆盖；实际已运行后不要通过换 run ID 冒充新数据。

输出：`artifacts/ic/ic02c_design_v1/report.json` 和 `task_proposals.json`。报告保存 commit、实际命令、解析配置、来源指纹、环境和完整预算。`engineering_gate=true` 只表示本次**离线设计审计**有效；`runtime_authorized=false`、`training_ready=false`、task classifier/pi=null 保持不变。

## 非执行提案

三任务工程提案：11000/11008/11016，最多 9 个参考 root、90 个候选分叉、114 次新 SUMO 会话、12,234 控制步、23,862 raw tick。32-task 总覆盖 pilot 需要工程审查后另获批准；不是本命令自动执行的下一步。

本阶段没有 collector `prepare/run/worker` 或课程 DDPG 入口，不写尚未实现的运行命令。详细定义、成本和暂停条件见 `docs/design/curriculum_task_characterization_v1.md`。本阶段仍不改分层阈值、在线特征、原奖励、候选动作、预测器或旧结果。
