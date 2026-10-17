# 交互强度课程研究本地归档：暂停，有限收益，未采用

归档日期：2026-10-17（Asia/Shanghai）。本记录保存 IC04 的固定结果与决策，不是新训练授权、论文独立确认实验或正式发布。

## 1. 归档身份与决策

| 项目 | 固定记录 |
| --- | --- |
| 状态 | `paused_limited_benefit_not_adopted`；不替换原 B3，不合并 main |
| 研究分支 | `codex/interaction-curriculum-v1` |
| 工作树 | `E:/Prediction_RL_worktrees_course_learning` |
| 只读基线 main | `75051acc06f6ea2ab6dda9be678f191b63d04569` |
| 实验准备、训练及评价来源提交 | `0ca3219b6ae543fc3fce73e8b60c82847144c539` |
| 本地归档标签名称 | `archive/interaction-curriculum-v1-20261017`，指向包含本文件的文档归档提交 |
| 完整配置 | `configs/development/ic04_matched_120k_v1.json` |
| 请求 | `artifacts/ic/ic04_train_v1/request.json` |
| 请求规范化 digest | `cae07a10837737b4fa4a7a63b370870354ec210ff83b1aaf0296068d2031be5c` |
| 发布状态 | `blocked_license_unresolved`；本次不推送、不发布上游源码或权重 |

本研究检验参考条件下的邻车动作响应强度 L/M/H 任务组织是否帮助 DDPG 利用冻结 B3 信息，不将交互强度等同控制难度或信息价值。B2/B3 三成员预测器冻结，原 reward/terminal/bootstrap、连续 jerk、预测摘要、网络、原优化器与均匀 replay 不变。

最终决策：课程对 B3 有方向一致但有限的最终回报和舒适性收益；没有证明更快学习、最终训练种子稳定性提高或可靠的动作条件信息利用机制。因此保留研究分支与全部结果，不替换主线、不调比例或预算寻找更好的结果，也不自动继续信息敏感性诊断。

## 2. 已完成的四组匹配实验

B2/B3 × Mix/Curriculum 四组、训练 seeds 0/1/2，全部从头训练；每模型请求 120k 控制步，总实际 1440376 控制步，均按原生完整回合边界结束。节点固定 20k/40k/60k/120k，主终点固定 120k，不挑选最好节点或成功种子。

四组统一使用 `seeded_fresh_gym20_task_v1`：每回合新的原生 SUMO/Gym20 会话、20 秒预热、无参考策略前缀，持久 learner/optimizer/target/replay 不重置。它不是旧持续交通训练协议；旧 B3 只能作历史参照，不能代替新 B3-Mix，也不能直接与差分分支数值排名。

有限训练池 32 项，L/M/H=11/10/11；0–20k 的采样质量为 60/30/10%，20–60k 为 20/20/60%，60–120k 恢复有限池均匀目标。Mix 使用整个预算的名义加权质量。各阶段只在原生回合边界切换，不截断回合或清空 replay。

评价使用无探索冻结策略、Gym20 新进程首回合，场景 200–219，共 960 回合。仍只有 20 个多次分析的开发交通场景与三个训练种子；不是 960 个独立场景，也没有打开独立测试集。

120k 主终点（回报 ± 为三个训练 seed 的均值标准差）：

| 方法 | 平均回报 ↑ | 自然到达 | 碰撞 | 平均绝对实测 jerk ↓ | 全回合平均仿真时长/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| B2-Mix | 8.3730 ± 0.1348 | 60/60 | 0/60 | 0.4737 | 11.4533 |
| B2-Curriculum | 8.2692 ± 0.3928 | 60/60 | 0/60 | 0.4923 | 11.5200 |
| B3-Mix | 8.1440 ± 0.1149 | 60/60 | 0/60 | 0.6095 | 11.1900 |
| B3-Curriculum | 8.3541 ± 0.1556 | 60/60 | 0/60 | 0.4882 | 11.5200 |

B3 的课程回报增量为 +0.2102（约 +2.58%），三个 seed 分别 +0.2221、+0.1507、+0.2576；jerk 下降约 19.91%，平均时长增加 0.33 秒。最终回报标准差反而由 0.1149 增至 0.1556，不能声称最终稳定性提高。B3-Curriculum 与 B2-Mix 的回报接近，没有明确的总体领先。

| 方法 | 20k 回报 | 40k 回报 | 60k 回报 | 120k 回报 | 20k–120k 归一化节点梯形 AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| B2-Mix | 7.8907 | 8.2386 | 8.3878 | 8.3730 | 8.3038 |
| B2-Curriculum | 7.0991 | 8.3675 | 8.0630 | 8.2692 | 8.0894 |
| B3-Mix | 7.7922 | 8.3241 | 7.9404 | 8.1440 | 8.0634 |
| B3-Curriculum | 6.9280 | 7.7513 | 8.3385 | 8.3541 | 8.0847 |

B3-Curriculum 在 20k/40k 落后；AUC 只增加 0.0213，三个 seed 中仅一个增加，不能声称稳定的样本效率收益。AUC 只是非等距四节点近似，不代表节点间连续测量。中间三次碰撞均发生在场景 207：B3-Curriculum seed2/40k、B2-Curriculum seed2/60k、B3-Mix seed1/60k，全部保留在统计中。

120k 课程对 B2 的回报作用为 -0.1039、对 B3 为 +0.2102，差中之差 G=+0.3140。三个 seed 的 G 约为 -0.0318、+0.2045、+0.7693；正均值同时包含 B2 退化，不能唯一解释成正确利用动作条件信息。B3-Curriculum 相比 B2-Curriculum 均值只高 0.0850，三个 seed 中两个更低。

## 3. 曝光与指标限制

| B3 组别 | H 层实际采集 transition 占比 | H 层累计真实 minibatch 占比 |
| --- | ---: | ---: |
| Mix | 38.18% | 38.43% |
| Curriculum | 38.20% | 32.77% |

名义 reset 配额相同不保证实际任务次数、控制步或 optimizer 更新量完全相同。课程顺序与历史均匀 replay 一起影响更新经历，最终恢复目标采样也不代表 replay 立即恢复目标分布。层标签不进入网络、奖励或动作执行。

同路径前车 gap 在主终点评价中全部缺失，不补零，不能作为安全证据；同路径后车 gap 有完整记录。旧 `minimum_observed_front_bumper_distance_m` 列是车辆参考点欧氏距离，不是真实车身表面 gap，不能据此宣称安全保证。

## 4. 证据定位与不可变哈希

以下为文件字节 SHA256，不能与请求规范化 digest 混用：

| 相对本工作树的文件 | SHA256 |
| --- | --- |
| `artifacts/ic/ic04_train_v1/request.json` | `b2aa5dabe79101ae4b6c4af7e0c01b922a74c69c27d35ae3c2cbd0531fa04352` |
| `artifacts/ic/ic04_train_v1/aggregate.json` | `9f432abf56cd0380c5b378dfcb80c1e7c477ef5680039df4c582fcc5a06d535f` |
| `artifacts/ic/ic04_train_v1/aggregate_receipt.json` | `d08073947b73417649e06f79da56d195ab554026869b0b5df33e5298b23fa045` |
| `artifacts/ic/ic04_train_v1/development_curves.json` | `0f7e87a1f3be9ccda56e8ffe43bba3bea7150d1d832370b0b864e43f72048beb` |

固定任务清单为同目录 `task_manifest.json`。节点策略、episode/replay/minibatch/loss 审计在 `train/`；冻结评价的逐回合 outcome、动作与原生证据在 `evaluate/`。权重仅为评价用途，不构成完整 optimizer/replay/RNG 续训断点。

冻结预测器由只读基线 `E:/Prediction_RL/artifacts/training/predictor_v1/` 提供：ordinary ensemble SHA256 为 `939242f358315bd00ce89978422e4b55b8059f41837463d2e80f661c63817e8d`，conditional ensemble 为 `daddeda8569b88c2a3482a0e8c739995785ee0f8216dda6525c86d2ce120e25f`。完整依赖链、历史失败证据、runtime/config/source 哈希以请求为准，不只保存本次策略。

## 5. 本次实际只读复核与保存边界

- 原 aggregate、development curves 和收据哈希、请求 digest 一致；当前完整 source inventory 与原请求一致。
- 从 960 条 episode 独立重算，冻结节点、配对增量、G 和 AUC 均与原统计一致；验证全部评价 complete binding、metadata、episode/report 哈希。
- 12 个训练记录与 complete metadata 一致、预算合格、同 seed 四组初始权重一致，全部节点权重及冻结 B2/B3 预测器哈希一致。
- 没有重新遍历大型 native/replay/log 的完整文件清单，没有运行 SUMO、模型推理、新训练或新评价。本次是文档归档，不另行宣称新的完整闭环验收。

本提交只增加本文件，不改 scientific source、配置、原结果、main 或差分分支。归档提交不是旧实验的运行提交；原请求严格绑定 `0ca3219b6ae543fc3fce73e8b60c82847144c539` 与干净工作树。归档后不能在新 HEAD 直接调用旧运行入口并通过修改请求绕过校验。后续只读分析应保留旧身份；重新运行须先回到匹配来源或另获新版本请求授权，不自动加预算。

Git 不备份忽略的 artifacts。本次没有复制大产物，也没有证明外部备份已完成；确认数据、原始/失败证据、checkpoint、日志及只读基线依赖独立备份之前，不删除工作树或 baseline 资产。

许可仍按 AGENTS.md 与来源记录标为未解决，现有远端 main 不构成发布授权。本地归档标签不移动，不作为 runnable/release milestone；不推送、不与 main 或差分分支合并。未来若另行研究，另建 `codex/` 分支并重新明确预算和协议。
