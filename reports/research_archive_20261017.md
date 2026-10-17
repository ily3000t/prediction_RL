# 差分监督研究本地归档：暂停，未采用

归档日期：2026-10-17（Asia/Shanghai）。本记录只冻结本研究的代码位置、结果和决策，不是论文复现成功标签、正式发布或独立测试结论。

## 1. 归档身份与决策

| 项目 | 固定记录 |
| --- | --- |
| 状态 | `paused_not_adopted`；不替换原 B3，不合并 main |
| 研究分支 | `codex/p05-response-difference` |
| 工作树 | `E:/Prediction_RL_worktrees/response-difference-v1` |
| 共同基线 main | `75051acc06f6ea2ab6dda9be678f191b63d04569` |
| 请求准备提交 | `3ea8f2eae73ab4f13490c9503e6023d1412c5e3d` |
| 训练执行及聚合来源提交 | `99ce0af0212d21dce60ccf6fbdec4765cf7053d9` |
| 本地归档标签名称 | `archive/response-difference-v1-20261017`，指向包含本文件的文档归档提交 |
| 完整配置 | `configs/development/p07_candidate_difference_120k_v1.json` |
| 请求 | `artifacts/difference_ddpg/delta_120k_v1/request.json` |
| 请求规范化 digest | `e2c86b9f2310b2d9a49a6fe00c5c284a7071102ac153ad484f2b7f9c95e1d372` |
| 发布状态 | `blocked_license_unresolved`；本次不推送、不发布上游源码或权重 |

唯一主要变化是预测训练中的同 root 候选邻车响应差分监督（lambda=0.2）；原连续 jerk、奖励、DDPG、149 维预测摘要和选模规则不变。该增量在本次固定终点未建立足够的闭环收益。保留实现与负结果，不通过挑选早期节点、删除 seed、改变 lambda 或继续增加预算改变本次结论。

## 2. 已完成实验与固定主终点

新增差分 B3 的训练种子为 0/1/2，各从头请求 120k 控制步；实际分别为 120072、120052、120115，按原完整回合边界结束。复用原 B0/B1/B2/B3 的冻结结果，没有重新训练参考方法或使用旧策略初始化新模型。

评价为 Gym、20 秒预热、新进程首回合、无探索、场景 200–219。节点为 20k/40k/60k/120k；120k 是唯一主终点。新增 240 回合、复用 960 回合，共 1200 条记录，仍仅 20 个开发交通场景及三个训练种子；不代表 1200 个独立场景，没有打开独立测试集。

120k 结果按场景先在每个训练 seed 内汇总，再对三个 seed 等权平均：

| 方法 | 平均回报 ↑ | 自然到达 | 碰撞 | 期限失败 | 平均绝对实测 jerk ↓ | 全回合平均仿真时长/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 无预测 | 3.9823 | 52/60 | 8/60 | 0/60 | 0.4155 | 24.7633 |
| B1 零预测通道 | 2.1930 | 46/60 | 6/60 | 8/60 | 0.3575 | 35.7833 |
| B2 普通预测 | 5.2112 | 55/60 | 5/60 | 0/60 | 0.4091 | 23.9833 |
| 原 B3 | 7.0407 | 60/60 | 0/60 | 0/60 | 0.2426 | 26.9267 |
| 差分监督 B3 | 5.9460 | 58/60 | 2/60 | 0/60 | 0.4150 | 25.2633 |

差分 B3 相对原 B3 的平均回报增量为 -1.0947，三个训练 seed 分别约为 -2.7478、-0.1626、-0.3737；平均绝对 jerk 增加 0.1724。全回合时长包含失败结局，不能将更短时长直接解释成成功合流更快。

| 冻结评价节点 | 原 B3 回报 | 差分 B3 回报 |
| --- | ---: | ---: |
| 20k | -4.0603 | 4.2118 |
| 40k | 2.2346 | 4.5427 |
| 60k | 5.1903 | 1.1228 |
| 120k（主终点） | 7.0407 | 5.9460 |

早期改善不能替代固定主终点；本次也不支持“差分学习在任何场景均无效”的泛化判断。归档结论仅为：该实现和固定预算下的增量不足以替换原方法。

## 3. 证据定位与不可变哈希

以下 SHA256 是原文件字节哈希；请求文件的字节哈希不同于上面的规范化 digest，两者不可混用。

| 相对本工作树的文件 | SHA256 |
| --- | --- |
| `artifacts/difference_ddpg/delta_120k_v1/request.json` | `1e2d1ed72f31ed1cb002e546b1c2c976115df4f01312326b6764d1c4c89693b7` |
| `artifacts/difference_ddpg/delta_120k_v1/aggregate.json` | `f38e56919936e9341d099f428b9e04eff508e2db0e1312b485a8f0a18653c350` |
| `artifacts/difference_ddpg/delta_120k_v1/aggregate_receipt.json` | `fa5a8b88185a4b88353659dd7221b825a5039f01c177ae4b2eae03b74753821e` |
| `artifacts/difference_ddpg/delta_120k_v1/development_curves.json` | `fce60ed7abbde8d60ec225b98c66d5e58b271208e2cee04b60e0820ce7ff0fb0` |

差分三成员预测器：`artifacts/prediction_difference/delta_v1/conditional_ensemble.pt`，SHA256 为 `eba3b43dad1425dad76569497cf3ab667761e84a72169c2fbdc96c8471302fbf`。校准证据位于 `artifacts/difference_calibration/delta_cal_v1/`；原参考结果位于只读主目录 `E:/Prediction_RL/artifacts/p7n/p7n_120k_v1/`。完整跨目录依赖与哈希以请求的 `input_hashes`、`reference_files`、`difference_request`、`calibration_request` 和 source/runtime 记录为准，不能只保存新策略文件。

训练权重与日志位于 `artifacts/difference_ddpg/delta_120k_v1/train/`，逐回合评价位于 `evaluate/`。评价权重不是完整 optimizer/replay/RNG 训练断点。

## 4. 本次实际只读复核范围

- 原 aggregate、development/training curves 与 aggregate receipt 的字节哈希及请求 digest 一致。
- 当前 `src/`、`tools/` 的冻结文本哈希与原请求一致。
- 从新增 240 条和复用 960 条 episode 重新计算，全部冻结统计与原 aggregate 一致。
- 检查 1200 条评价的 complete binding、metadata、episode/report 哈希，以及新增三模型的 matched initialization、训练记录和全部节点权重哈希；差分预测器哈希一致。
- 没有重新遍历大型 native/replay/log 的完整文件清单，没有运行 SUMO、模型推理、训练或新评价；不得将本次文档归档称作新的完整闭环验收。

## 5. 保存、复现与发布边界

本归档提交只增加本文件，不改源码、配置、原始产物、main 或其他研究分支。原实验身份仍使用上述准备/执行提交，不能回写为归档提交。复核或复现时须尊重旧 source/config/receipt guards；不得修改旧请求来适配新的 HEAD，也不得从评价权重冒充完整续训。

Git 分支和标签不包含被忽略的 artifacts。本次没有复制大产物，也没有据此证明外部备份已经完成。在确认独立备份包含数据、预测器、策略、日志、原参考证据及依赖哈希前，不删除任何工作树或只读基线资产。

许可状态沿用 AGENTS.md 与 `provenance/upstream_snapshot.json` 的未解决记录；现有远端 main 不视作发布授权。本标签仅作本地研究归档，不是 runnable/release milestone。后续若重新研究，从固定来源另建 `codex/` 分支；不移动归档标签，不自动扩大训练或与课程分支合并。
