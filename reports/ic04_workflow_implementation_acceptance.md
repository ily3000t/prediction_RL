# IC04 四组匹配工作流源码验收

## 交付与执行状态

新增 `tools/train_curriculum_ddpg.py`，命令为 prepare、run（train/evaluate）、status、aggregate 和父进程内部 worker。独立配置 `configs/development/ic04_matched_120k_v1.json`；完整流程、环境与人工运行命令见 `docs/runbooks/ic04_matched_120k_v1.md`。

这一验收只证明冻结协议、执行编排、合成优化路径和既有原生证据的只读重放通过；**没有启动新的 SUMO 会话、真实 predictor inference、120k 训练或新增开发评价**。原 B3 主目录、历史产物与 Git 保持只读。

## 设计保持

- 全部 32 train-only tasks，L/M/H=11/10/11；原提案/失败不改变。新的 live manifest 单独保存任务标签、完整 D/覆盖和来源哈希。
- B2/B3×Mix/Curriculum，optimizer seeds 0/1/2；原三成员预测器冻结。私有 sampler seeds 17003/17004/17005，仅为可重复的独立随机流，不是额外优化器 seeds。
- 每模型 120k 请求控制步，完整原生回合边界最多 +499；20k/40k/60k/120k 的首次完整边界存权重。12 模型总请求 1.44M，主终点 120k。
- 固定 Gym20 新进程首回合，20 个场景 200–219，960 回合。没有 author50、泛化、test、新预测器训练或预算扩展。
- 保留原 ALL DDPG train loop、原 reward/action/terminal/bootstrap、原参数/均匀 replay/探索。新 task sampler 和审计不进入 observation、reward、执行动作。
- 旧 scientific src、offline/smoke guards 与源文件指纹不变。工具层新 typed scope 的构造器单独绑定确认的 IC04 请求，继承原 store/sample/terminal lifecycle；没有修改旧 gate 或将 offline purpose 改名开放训练。
- CPU：训练并发 1、评价并发 2、各 Torch threads=1。每回合 fresh native 会话是已批准的新训练协议，旧持续交通 B3 不能作严格 Mix 对照。
- 完整 job 在 child 退出、finally 清理和 stdout 文件关闭后由 parent 严格封存。所有日志参与收据；不重试失败或不完整 job，评价权重不作为训练恢复状态。
- 原生合法负结果正常聚合；工程错误必须保留并停止，不换 seeds/thresholds。按训练 seed/场景逐对计算增量与 G；固定非等距节点梯形近似 AUC，报告三个 seed 的离散程度。

## 实际验证

- 新增 **29 passed**，最终复核 17.13 秒：冻结配置/完整 roster、旧 null 标签与新任务清单、私有 RNG、边界超出和预算拒绝、原 guard 不被开放、真实优化合成等价性、全部负结果聚合、G/AUC、非法/不完整结果拒绝、收据封存与禁止部分恢复。
- 原 ALL preset 的合成参考/审计路径各 20 回合、5,000 控制步，达到原 warmup 后产生原优化更新；动作、replay State/action/reward/mask、policy/critic、target、scheduler、optimizer 和 Python/NumPy/Torch RNG 逐位一致。不是 120k 结果，也不含真实交通仿真。
- 只读重放已有两条封存 native 轨迹（到达和碰撞），新 EpisodeRecorder 通过原 raw/native 终止与 reward/tick 验证；没有新会话或反事实分叉。
- 全项目最终离线回归 **1,324 passed，2 warnings，62.02 秒**（此前一轮 67.93 秒同样通过）。两条 warning 来自原作者 ALL watch 的已知 `torch.load(weights_only=False)`，不是新的训练错误。
- CLI `--help`、四个 Python 文件的 AST 语法检查和 `git diff --check` 均通过。不能将静态/合成验证描述为真实训练已完成。

## 结果解释限制

I 是参考策略 approach 条件下邻车响应潜力，不是任意学习策略下的在线信息价值或安全等级；D 与覆盖独立保留，不根据新策略输赢分层。Mix 是名义 reset 配额匹配，实际层曝光需从 transition/replay/minibatch 审计解释。

原 ALL 的 reset/terminal agent 调用继续保留，因此回合长度不同可引起优化更新次数不同；actual control budget、agent calls 和真实 minibatches/optimizer loss 次数分别记录。不通过改 warmup/update loop 消除此差异，也不宣称四组优化次数完全相等。

每回合新原生进程/预热成本保留，当前不提供未经整程测量的 ETA。主结论仍需人工运行全部 12 模型与 960 回合后判断：一般课程收益还是动作条件特异性，以及最终安全/效率/舒适性是否保留。开发场景不是新的独立测试，G 或特征敏感性不能单独证明因果机制。

新间距列按原道路 progress/same_path 定义，并报告缺失数量；不把跨路径 TTC 当即时风险。旧 `minimum_observed_front_bumper_distance_m` 列是参考点欧氏距离，其命名局限已在输出与运行文档明确，不当作真实表面距离。
