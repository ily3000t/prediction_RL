# IC02b：fresh Gym20 worker 入口一致性 smoke

## 范围

用户在明确的 IC02b 问题后回复“继续”，批准入口原型、本工作树 Cython 构建及最多 12 个工程 episode / 144 控制步的 smoke。见 `provenance/ic02b_smoke_authorization.json`。这不批准课程训练、新任务分层或额外反事实分支采集；C0 缺口与 target pi 不因入口通过自动解决。

`FreshGym20TaskEnv` 是单任务原型：worker 拥有原 upstream session、SUMO 与 RNG；父进程不导入上游 globals，不随任务播种。每个新实例只有原首回合 reset，20 秒等待、原 ego 插入、原 observation 和 continuous jerk。后续训练接入仍未实现；不能声称已验证真实 learner/replay 持续状态。

`control_cap` 只禁止额外工程 step；最后原生 done=false 时保持 false，不合成失败或 truncation。IPC liveness watchdog 是异常检测，触发即运行无效，没有 stale、观测替换或返回有效终止。课程元数据不进入网络输入。

## 本地构建

在 **课程工作树**的 `RL-MPC-LaneMerging-master` 内，使用原 `setup.py`：

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B setup.py build_ext --inplace
```

不能借用基线目录模块、静默禁用 Cython或改动原 config。生成 C / build / pyd 受 Git 忽略；请求记录 setup/pyx/source、依赖和实际本地扩展哈希。新本地编译字节不宣称与旧扩展字节一致，两条 smoke 路径均使用本地扩展。

## 离线验证

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B -m pytest -q -p no:cacheprovider tests/test_curriculum_fresh_entry.py tests/test_curriculum_entry_design.py tests/test_curriculum_interaction.py tests/test_actor_adapter.py tests/test_dataset_contract.py tests/test_response_collection.py tests/test_probe_features.py tests/test_upstream_contract.py tests/test_terminal_events.py tests/test_terminal_event_recovery.py
```

受限临时目录环境中，另传一个新的、经过绝对路径校验的 `--basetemp` leaf，不对现有结果目录使用它。离线测试覆盖入口、既有特征及 terminal 回归；实际数量记录在验收报告，正式执行前必须审查并提交源码/配置。

## 有界执行

从 `E:\Prediction_RL_worktrees_course_learning` 的干净研究分支执行；以下 CLI 已实现，但命令存在不表示 smoke 已完成。

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/smoke_curriculum_entry.py prepare --baseline-root E:\Prediction_RL --run-id ic02b_v1
```

准备命令输出实际 request 路径和 `confirm_request_hash`。使用该哈希运行：

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/smoke_curriculum_entry.py run --request artifacts/ic/ic02b_v1/request.json --confirm-request-hash <prepare输出的实际哈希>
```

固定 train seeds 11000/11001/11002；每 seed 的 original direct 与 worker-entry 各重复两次，共最多 12 回合。每回合原 12 个固定连续 jerk，共最多 144 控制步；自然结束时提前停止，不换 seed、补步或改 done。等待/插入最多 1,212 tick，原期限 cleanup 独立记录。

两种冻结三成员 predictor 各最多 156 次当前状态特征调用。direct 在独立子进程执行原 `AuditedJerkEnv`；worker-entry 在环境 worker 执行同一原类。通过 CURRENT packet view 分别复用原 `PredictionFeatureEnv` 的普通/条件特征路径；只有 observed history 传给 predictor，不传整条 trace 或未来。worker 的父进程特征调用在下一个控制请求前完成，SUMO 不前进。

## 产物和验收

`artifacts/ic/<run-id>/request.json` 保存解析配置、来源 commit、命令、基线引用、源码/配置/模型文件哈希与环境版本。每回合有 start receipt、日志、全量当前交通/观测/reward/done/raw events/history/预测摘要和不可覆盖 complete receipt。

`report.json` 需要全部 12 格、重复与跨路径精确一致性、父进程 RNG 不变、冻结权重和合法真实 tick/feature 调用计数。仅 latency 与 PID 排除于语义比较；不放宽数值阈值。终止/cleanup 的传输另有离线 fixture，短 live smoke 不声称实际覆盖全部自然终止类别。

失败保存 `failure.json`，不改 seed/标准；不自动重跑失败或不完整 episode。`--resume` 仅复用同源码请求下已经完整校验的回合，不补跑已经开始但未完成的回合。成功/失败顶层报告都不可覆盖；不得用换 run ID 绕过本次批准总预算。

通过入口验证后仍需单独设计 task-level D/I 参考刻画、三层覆盖、目标 pi 与数据预算，并经用户批准。当前不提供也不运行课程 DDPG 命令；不合并 main、不推送、不创建可运行课程标签。
