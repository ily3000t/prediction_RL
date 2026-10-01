# P7d — 冻结模型与作者完整 RL＋MPC 的开发集对照

## 范围

先完成外部对照，再讨论模型修改；不训练、不选新 checkpoint、不改奖励。
只读复用 `artifacts/p7/p7_explore_v1/train/conditional_s{0,1,2}/final.pt`
及其严格绑定的三成员条件预测 ensemble。旧 P7、P7b 与 sealed test 不变。
P7c 噪声诊断仍是草案，本阶段不执行；没有进入 P8 正式冻结。

| 方法 | 来源 | 本轮回合数 |
| --- | --- | ---: |
| 作者 DDPG | `train_default_1.json` 的环境与作者 default 预训练权重 | 20 |
| 作者 ST/MPC | `st_default.json`，无学习权重 | 20 |
| 作者 RL＋MPC：安全接管 | `combined_default_1.json` | 20 |
| 作者 RL＋MPC：安全＋效率切换 | `combined_default_1b.json` | 20 |
| 当前条件预测＋DDPG | 冻结的三个 P7 训练种子，不接入 ST | 60 |

合计 140 个新回合，交通种子固定为 200–219。两版 hybrid 在查看结果前均纳入。
这不是“普通预测 vs 条件预测”的归因实验；P7 的 B0–B3 内部对照仍然保留。

## 执行和评分契约

全部方法直接调用未修改的 `control.run_episode`；预热 50 秒与作者
`evaluate_control` 默认一致。保留原始 0.2 秒步长、连续 jerk、SUMO 设置和终止检查。
原循环在名义 100 秒上限下允许最多 501 次控制调用，时间限制回合可报告 100.2 秒，
不为了形式对齐而修改成 Gym 的 500 步。原循环检测任意车辆碰撞，不能将该字段
默认为自车碰撞。完整原始观察/速度/加速度/jerk/接管记录保存于每回合 `trace.json`。

所有方法均先计算完成再推进仿真；不超时丢弃、不 stale、不改变 observation。
CPU 推理是显式运行设置（`CUDA=false`），不是奖励/策略/监督器修改。
作者 hybrid 保留虚拟 rollout 中反复查询策略及原 TimeFeature 计数，不重新定义它。
作者 ST 的速度命令直接执行，绝不转成 jerk 后再经过另一个裁剪器。

作者的 `MODEL_NAME` 指向 `runs/...`，但随源码提供的权重实际在
`pretrained_models/ddpg_default1_extended`；加载前严格校验 policy/q SHA256 和目录文件集合，
并记录该路径映射。只使用这份已核验资产，不随机寻找“更好”的作者模型。

本轮每个交通种子从独立进程的首个回合开始，以获得严格配对场景。
这不同于作者 4000 个连续回合的原始批处理，属于公开记录的开发评价适配，
不是论文数值的严谨复现。当前模型只做实际观察历史推理，不预测真实未来、
不输入虚拟 rollout 历史、不执行候选吸附。

奖励评分调用未修改的 `rl.get_rl_custom_episode_stats` 和
`dqn.get_reward_function()`（Slotted Jerk）。输出明确命名为
`author_history_total_reward`：该原始历史函数有 N−1 项、末项应用终止奖励，
不是旧 P7 Gym 的逐 transition return。保留原规则，不把两种 return 混在一张表。
作者后合流最近距离（有位置限制和 100m 截断）也不冒称全回合车身安全距离。

## 工程验收与手动执行

`audit` 固定只用交通种子 200。四个作者控制器各执行一次记录版和一次原控制器版，
逐项精确比较速度命令、实际运动、终止和接管；当前模型执行一次冻结推理 smoke。
总计最多 9 回合、4509 次实际控制调用。它只检验工程，不决定方法效果。

在 `E:\Prediction_RL`、已激活 `pytorch` 环境下：

```powershell
python tools/compare_external_baselines.py audit --run-id p7d_audit_v1
python tools/compare_external_baselines.py prepare --run-id p7d_compare_v1
```

已有验收产物时不要重跑 `audit`。`prepare` 默认验证
`artifacts/p7d/p7d_audit_v1/aggregate.json`；非默认验收 ID 可用
`--engineering-audit artifacts/p7d/<audit-id>/aggregate.json` 指定并记录哈希。
准备输出 request 路径与 `confirm_request_hash`，检查后由用户运行：

```powershell
python tools/compare_external_baselines.py run --request artifacts/p7d/p7d_compare_v1/request.json --confirm-request-hash <prepare打印的哈希>
```

完成后自动生成 `artifacts/p7d/p7d_compare_v1/aggregate.json`。
重复启动必须加 `--resume`；只复用完整、哈希和 lineage 验证通过的回合，
不覆盖或自动重试半成品。source/config/runtime 变更会拒绝旧请求，不能静默继续。
NaN、模型错误、执行错误是无效运行；碰撞、停车和到达失败是有效负结果，正常聚合。

## 分析边界

先联合查看到达、碰撞、时间限制，再看原奖励和舒适性。效率同时报告
“各方法成功回合”的条件均值/数量以及“双方均成功的同场景”的配对时间差。
不把停车导致的低速度/低 jerk 宣称为优越舒适性，不按多数相关指标投票选冠军。
控制器耗时报告 mean/p50/p95/p99/max，是描述性成本，不参与方法效果 gate。

作者只提供一份 default 权重，训练 seed 的可核验来源不足；不伪造三份独立训练。
聚合保留当前模型每个训练种子的 20 场景配对差，再等权汇总，
重复使用同一作者结果不增加作者独立样本量。作者训练预算与本项目 P7 的 20k
训练步不匹配，因此本轮是外部系统探索对照，不支持匹配预算的正式优越性结论。
当前 B3 无 ST，作者 hybrid 有 ST；差异属于完整系统效果，不能归因于预测器一项。

结果完成后先分析优势、退化和代价，再决定是否需要同预算训练、统一 ST 条件，
或定向模型修改。不会因单个失败种子删样本或修改阈值。
