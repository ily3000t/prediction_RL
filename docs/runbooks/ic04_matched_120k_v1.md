# IC04：交互强度课程四组匹配实验

## 环境与范围

在 Windows 的现有 `pytorch` Conda 环境运行。本次使用：

```powershell
conda activate pytorch
Set-Location E:\Prediction_RL_worktrees_course_learning
python -B -c "import sys,torch; print(sys.executable); print(torch.__version__)"
```

应显示 `E:\Programs\EnvAnaconda3\envs\pytorch\python.exe`。SUMO、TraCI、ALL 0.5.3 和本工作树的 Cython extension 沿用已通过 IC03b 的环境；不要重新安装、关闭 Cython、切换 GPU 或改依赖来绕过校验。原主目录 `E:\Prediction_RL` 只读；本工作树保持 `codex/interaction-curriculum-v1` 分支，不合并或推送。

这是已多次检查的 **开发集实验**，不是独立测试。工程 gate 只检查代码、数据与执行证据，负效果不会导致聚合失败。

## 冻结设计

独立配置：`configs/development/ic04_matched_120k_v1.json`，不 extends。原 IC02h、IC03b 配置及失败报告不修改。

| 项目 | 冻结值 |
| --- | --- |
| 方法 | B2-Mix、B2-Curriculum、B3-Mix、B3-Curriculum |
| optimizer seeds | 0、1、2，四组全部从头训练 |
| 私有 sampler seeds | 对应 17003、17004、17005；不消耗 learner/探索/交通 RNG |
| 每个模型请求步数 | 120,000；原生完整回合结束后最多超出 499，另外记录 |
| checkpoint 节点 | 20k、40k、60k、120k；不按开发结果挑最好节点 |
| 总请求训练预算 | 12 个模型，1,440,000 控制步 |
| 固定开发评价 | 200–219，每个节点/模型各 20 回合；总计 960 回合，仍只有 20 场景 |
| 训练/评价并发 | 1 / 2 个 job，各自 Torch threads=1，CPU |
| 原算法 | ALL 0.5.3 DDPG、学习率 0.0002、warmup 5,000、batch 100、replay 1,000,000 |
| 预测器 | 原 B2/B3 三成员 ensemble，只读冻结；不训练预测器 |
| 奖励/终止/动作/特征 | 原 Slotted Jerk、原终止和 bootstrap、连续 jerk、原探针和 169 维输入；TimeFeature 由原 body 追加 |
| 执行 | `simulation_blocking_exact_v1`，不因计算耗时丢弃结果 |

新的任务清单从完整 32 项 train-only IC02h 提案生成，L/M/H=11/10/11，阈值不变。新的 `task_manifest.json` 明确赋层，旧提案仍保留 null 标签；完整标签和真实未来只用于离线任务组织，不进入 online observation、reward 或动作选择。

| 实际控制步阶段 | L | M | H |
| --- | ---: | ---: | ---: |
| 0–20k | 0.60 | 0.30 | 0.10 |
| 20–60k | 0.20 | 0.20 | 0.60 |
| 60–120k | 0.34375 | 0.3125 | 0.34375 |
| Mix（全程） | 0.3385416667 | 0.2729166667 | 0.3885416667 |

阶段由实际控制 transition 决定，下一合法 reset 才生效；不中断回合、不清空 replay。Mix 匹配的是名义按阶段长度加权的 reset 质量，不是实际回合数或 transition 曝光完全相同。后期恢复有限池目标，不是旧持续交通分布，也不表示 replay 更新分布立即恢复。

原 ALL 的 agent 调用还包括 reset/terminal 调用，不能把 agent `_frames_seen` 当作实际控制步。不同组回合长度不同，原算法的实际优化更新数也可能不同；输出同时记录实际 controls、agent calls、minibatches 和 loss/update 次数，不声称 120k controls 保证优化更新次数完全相等。

所有组使用已批准的 `seeded_fresh_gym20_task_v1`：每回合独立原生 SUMO/Gym20 会话、原 20 秒预热、无作者参考动作前缀。持久 learner/optimizer/target/replay 保留，只有 terminal 被原 replay.store 接收并核查后才创建下一会话。这个协议会有每回合新进程/预热成本；没有通过改变 reset 语义来减少成本。旧持续交通 B3 仅历史参照，不能替代 B3-Mix。

## 1. 准备请求：不开始训练

若 Codex 已完成本轮准备，直接使用交接中打印的 request/hash，**不要重复 prepare**。否则在干净工作树中执行：

```powershell
python -B tools/train_curriculum_ddpg.py prepare --baseline-root E:\Prediction_RL
```

输出：

```text
curriculum_request=...\artifacts\ic\ic04_train_v1\request.json
confirm_request_hash=<本机生成的完整 SHA256>
```

准备会核查只读原主目录、预测器、原科学源码、四路径原生 smoke 和旧失败的不可变证据；保存完整配置、环境、Git SHA、源码/资产哈希、新任务清单及 12/960 项完整 roster。不会启动 SUMO 或加载模型做推理。

以下 `<REQUEST_HASH>` 必须换成上面打印的 64 位哈希。准备后到实验结束不要修改源码、配置、依赖、Git HEAD 或冻结资产。请求严格绑定干净 commit，不能随意 rename 实验绕过预算或失败。

## 2. 用户启动匹配训练

```powershell
python -B tools/train_curriculum_ddpg.py run --request artifacts/ic/ic04_train_v1/request.json --confirm-request-hash <REQUEST_HASH> --stage train
```

一次顺序执行全部 12 个模型，每模型继承原 ALL 的完整回合 train loop，在固定节点保存评价权重；原探索、优化、均匀采样不变。不会复用旧 policy，也不会将权重文件冒充完整 optimizer/replay/RNG 断点。训练 job 中有 `worker.log`、TensorBoard、`episodes.json`、`losses.jsonl.gz`、`replay.jsonl.gz`、真实 `minibatches.jsonl.gz`、每回合原生证据、四个 `n*.pt/json` 和完整收据。

训练日志和 `episodes.json` 是带探索噪声的训练表现，不能称为冻结策略验证曲线。

## 3. 用户启动冻结开发评价并自动聚合

全部训练封存完成后：

```powershell
python -B tools/train_curriculum_ddpg.py run --request artifacts/ic/ic04_train_v1/request.json --confirm-request-hash <REQUEST_HASH> --stage evaluate
```

在相同 20 个场景上以无探索的冻结策略执行全部 960 回合。评价必须与训练的完整 observation/TimeFeature、预测器和动作契约一致；无训练更新、无 replay 插入。原生 raw events 与返回 terminal 原因不一致时停止并标记工程失败，不重算原奖励、伪造正常回合或换 seed。

完成后自动生成：

- `artifacts/ic/ic04_train_v1/aggregate.json`：四节点、全 seed、四个配对增量、逐场景/训练 seed 的差中之差 G、非等距节点梯形近似归一化 AUC、实际曝光与原优化损失计数。
- `artifacts/ic/ic04_train_v1/development_curves.json`：真正的冻结策略开发曲线，不是训练 returns。
- `artifacts/ic/ic04_train_v1/aggregate_receipt.json`：输出校验。

主终点固定 120k。G 为正不独立证明正确利用信息；同时检查到达、碰撞、期限失败、jerk、间距和三个 seed。三 seed/二十开发场景的证据有限，不宣称独立测试显著性；不自动延长训练、修改网络或增加 author50/泛化评价。

额外间距列按原 `MergeGeometry.same_path` 与道路 progress 定义记录前/后 bumper gap；缺失分别报告可用 episode 数，不补零。跨路径投影 TTC 不作为即时安全证明。历史列 `minimum_observed_front_bumper_distance_m` 实际是车辆参考点的欧氏距离，保留兼容性并在每个结果中明示，不能称为真实车身表面最小距离。

## 4. 状态、完整复用与故障交接

```powershell
python -B tools/train_curriculum_ddpg.py status --request artifacts/ic/ic04_train_v1/request.json --confirm-request-hash <REQUEST_HASH>
```

需要复用已经完整完成的 job 时，可在对应 run 命令追加 `--resume`。它只接受整个 job 的完整严格收据；不会恢复任何已开始但不完整的训练或评价，不会根据 `n20000.pt` 继续优化。报告全部失败/碰撞均保留；工程中断请提供对应 `worker.log`、`failure.json` 和 invocation report，先诊断，不删除目录重跑。

自动聚合已生成后不要再调用 aggregate 覆盖；未生成且全部评价完整时可手工调用同一 CLI 的 `aggregate`。原始权重、日志和大数据在忽略的 artifacts 下，不进入 Git。运行耗时由真实机器、回合长度与每回合原生会话开销决定，当前没有可支持准确 ETA 的整程基准。
