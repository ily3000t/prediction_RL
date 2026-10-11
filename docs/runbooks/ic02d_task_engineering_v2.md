# IC02d 修复版：相同三任务工程验证

## 当前状态：已完成，以下命令仅记录本次执行

`ic02d_eng_v2` 已通过工程验收，不要再次 prepare/run 或覆盖其 request/report。实际 114 会话、4,186 控制步、15,700 raw ticks、9 roots、90 分叉；详见 `reports/ic02d_engineering_v2_acceptance.md`。剩余任务另行批准，不通过重命名重置已消耗预算。

用户在旧请求零 SUMO 失败及 allowlist 修复后，批准同三个训练 seed、同上限的新请求。旧 `ic02d_eng_v1` 及原配置、授权、失败记录保持不变。新版本仅修改执行标识及绑定的授权文件；IC02c 科学设计、原环境、奖励、动作、五候选、参考策略、后续零 jerk、预测时域、旧归一化尺度均未修改。

新请求在准备、worker 启动及收尾时核对旧失败文件哈希和零执行目录。`started.json`/`launch.json`/`worker.log` 是允许的父进程准备文件；已开始的 worker 仍不能自动重试。root ID 沿用 IC02d 科学刻画 v1 命名空间，不代表复用旧运行结果；每项产物由独立 v2 request hash 和 job receipt 绑定。

只在 `E:/Prediction_RL_worktrees_course_learning`，测试并提交新版本后运行：

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/collect_curriculum_tasks.py prepare --baseline-root E:\Prediction_RL
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/collect_curriculum_tasks.py run --request artifacts/ic/ic02d_eng_v2/request.json --confirm-request-hash <本次prepare输出哈希>
```

本次 run 不应重复执行。串行种子 11000、11008、11016，最多 114 新 SUMO 会话、12,234 控制步、23,862 raw ticks、9 roots、90 分叉。新配置 `configs/development/ic02d_task_engineering_v2.json`；独立授权 `provenance/ic02d_engineering_v2_authorization.json`。没有 B2/B3 模型推理或训练，没有 learner/DDPG；其余 29 个 pilot task 未获批准。

产物位于 `artifacts/ic/ic02d_eng_v2/`：顶层 request/preparation/started、report 或 failure、postflight；各 task 封存原始事件、重复发现、历史、候选响应与 label、预算及 tensor/RNG/files postflight。验收在不追加仿真步的情况下从持久化数据重算；不可达检查点和 terminal 截尾如实保留，不替换 seed、不降低 gate、不补零未来。

工程通过不等于 C0/C1/C2 可划分。此阶段不拟合新阈值、不赋层、不计算 pi，`training_ready=false`。完整三任务结果需先审阅，再决定是否请求其余 29 个任务；不自动训练、合并、推送或打 tag。任何开始后失败均保留并暂停；新请求仍需另行批准。
