# P7n：一次固定 120k 匹配预算验证

## 目标与前置条件

用户已确认：完成 60k Gym20 外部比较与 P7m 信息利用诊断后，允许一次
固定 120k 的 B0–B3 四组匹配实验。一般预测已有开发收益，条件化增量尚未成立；
这不是为了把 B3 调成胜出，也不批准 P8、正式测试或无上限训练。

## 修改边界

新增独立配置、请求执行器、扩展节点聚合、测试与文档。
旧源码、历史配置和结果保持不变；不修改上游或 E:/WCDT_ACCVP。
冻结预测器复用，DDPG 四组全部从头训练，不加载旧 policy/critic 作 warm-start。

## 冻结实验

- B0 baseline、B1 zero、B2 ordinary、B3 conditional；训练种子 0/1/2 全部保留。
- 每实例请求 120k，共 12 实例 / 1,440,000 请求控制步。
- 新运行保存 20k、40k、60k、120k 四节点；120k 固定为主终点。
- 原完整回合边界最多超出 499 步；同一实例不中断 optimizer/replay/SUMO/RNG。
- 原奖励、连续 jerk、输入、MLP、预测器、DDPG preset、2M scheduler、Gym20 均不变。
- 同一 200–219 开发场景，三训练种子、四节点，共 960 评价回合，但仅 20 个场景。
- train 串行、evaluate 两 worker；不加入 Shield、commitment 或新 reward。
- 结果再差也正常聚合；不剔除 seed，不挑选最优 checkpoint，不自动延长。

## 实现与验收

复用不可变 P7k train_job / evaluation_job / training_metadata 执行 kernel，
不动态替换其 globals、修改历史 loader 或隐式转换协议。新请求版本、路径、
配置和 source hash 明确区分 P7n。旧 kernel 的 p7k 日志/原始事件标签仅为
实现标签，不能据此把新运行当成旧结果；模型身份绑定新 request SHA。

增加 120,499 步的 non-wrapping replay 预检，保留原 1M 容量。
单元测试验证四个独立边界的精确行为/RNG/optimizer/target/replay 等价性；
验证真正的 120k 请求、完整 roster、负结果和原聚合统计一致性。
真实 SUMO smoke 使用**原先独立冻结的 P7k 工程设计**：单训练种子、
16/32/64 节点、warmup/batch=8。它位于新 P7n 输出目录，不能作为 120k 方法证据。
四训练实例每个最多 563 控制步，十二冻结评价每个最多 500 控制步。

## 产物与 Git

功能分支 codex/p07n-fixed120k-matched；实现与验收记录分别 Conventional Commit。
不推送、不新增 tag。大产物位于 artifacts/p7n/<short_run_id> 且 Git 忽略。
正式大小的 12 实例训练与 960 回合评价均由用户手动执行。

## 暂停条件与交接

源码、runtime、预测器、decision receipt、原始事件或初始场景不一致则停止。
半完成训练目录不能通过 --resume 从只有权重的快照续训或自动重试。
完整结果出来后明确判断：停止条件化路线、修改有定位证据的模块，或另行讨论
正式冻结；不自动改到 200k、不切换有利评价协议。
命令见 docs/runbooks/p07n_fixed120k_matched.md。
