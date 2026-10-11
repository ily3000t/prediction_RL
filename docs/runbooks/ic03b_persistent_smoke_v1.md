# IC03b：持久 learner 的有界真实工程 smoke

用户在明确提出四组各两个回合、最多八个 SUMO 会话/4,000 控制步、零优化更新后确认“继续”。授权记录为 `provenance/ic03b_smoke_authorization.json`。本命令不启动正式课程训练，不打开 predictor test，不创建新的反事实分叉。

独立配置：`configs/development/ic03b_persistent_smoke_v1.json`，无 extends。初始化 seed 固定 0，私有 sampler seed 固定 17003；四条路径为 B2/B3 × Mix/Curriculum。仅使用 IC02h 完整 32-task 提案的 smoke 标签，不发布 live training manifest 或更改旧 null 字段。

每条路径从头创建一个原 ALL DDPG learner，保留原 CPU/.0002、warmup 5,000、batch 100、replay 1,000,000 等 preset；运行两个原生完整回合。不会从历史 policy 加载权重。每条 learner 最多 1,002 次 act，因此无需改 warmup 就不会更新 optimizer；额外 update guard 在异常更新发生前拒绝运行。网络与 target hash、空 optimizer state、零 minibatch 均在 postflight 核对。

真实 native worker 仅在 reset 创建，上一 worker 终止 store 完成且关闭后才创建下一项；历史从一个当前 frame 开始。任务进入/step/冻结预测必须保持 learner RNG 不变；原探索噪声正常改变 learner RNG，不误要求整个 learner 运行 RNG 恒定。课程标签不进入 observation 或 State.info。

严格检查每回合 101 预热/插入 tick、实际原控制 tick、必要 deadline cleanup；独立重算原 Slotted Jerk 奖励，deadline 使用移除 ego 前的第一 control tick 加速度。终止没有额外 penalty。碰撞/到达/期限失败均保留，不用 smoke 策略性能作为工程 gate。

来源检查绑定 IC02h 四项固定字节哈希及其**全部** input_hashes；源码清单仅接受 IC03 的两项集成模块和 IC03b 的明确新模块。旧 gates、配置、报告不修改。predictor 严格完整加载并只读冻结；本地 Cython、SUMO binary、包版本、上游配置与原空间 trace 同样进入 request。

在干净 `codex/interaction-curriculum-v1` 工作树、pytorch 环境中：

```powershell
python -B tools/smoke_persistent_curriculum.py prepare --baseline-root E:\Prediction_RL
python -B tools/smoke_persistent_curriculum.py run --request artifacts/ic/ic03b_smoke_v1/request.json --confirm-request-hash <prepare打印的hash>
```

只允许固定新 run ID；不可覆盖，无自动 retry、无 incomplete resume。失败时保留 parent/worker failure、日志和已消耗预算，停止。不能修改代码后使用相同请求重跑；任何修复版运行必须另行记录未完成原因并获准剩余预算，不能自动再获得八个回合。

输出为 `artifacts/ic/ic03b_smoke_v1/`：request/preparation/started、逐 arm 的环境与 native/replay 日志、原生回合 summary、不可变 complete 收据、最终 report 或 failure。工程通过后仍不启动四组×三个 seed×120k；IC04 请求冻结与人工批准是独立下一步。
