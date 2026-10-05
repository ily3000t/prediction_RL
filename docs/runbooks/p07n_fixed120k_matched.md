# P7n：用户执行的固定 120k 四组匹配实验

工作目录 E:\Prediction_RL，环境 pytorch。完整配置：
configs/development/p07n_matched_120k_v1.json，无 extends。
上游环境/奖励配置仍为 RL-MPC-LaneMerging-master/configs/train_default_1.json。
冻结普通/条件三成员预测器复用；12 个 DDPG 则全部从头训练。

## 1. 准备请求（不训练）

```powershell
python -B tools/train_fixed_budget_ddpg.py prepare --run-id p7n_120k_v1
```

检查打印出的请求和 hash；下面的 HASH_FROM_PREPARE 替换为该值。
若请求已由开发验收生成，不重复 prepare，直接使用其打印 hash。
所有阶段完成前保持源码、配置和环境冻结、Git 干净；不要中途调整超参数。

## 2. 从头训练全部四组

```powershell
python -B tools/train_fixed_budget_ddpg.py run --request artifacts/p7n/p7n_120k_v1/request.json --confirm-request-hash HASH_FROM_PREPARE --stage train --resume
```

B0/B1/B2/B3 × 种子 0/1/2，共 12 实例，串行 CPU 一线程。
每实例 120k，共 144 万请求步；各节点按原完整回合边界允许最多 499 步超出。
保留原 2M scheduler、5000 replay warmup、batch=100、lr=.0002。
新运行内 20k/40k/60k/120k 快照不中断训练状态，120k 为唯一主终点。
不能将新 20k/60k 节点与旧实验当作严格逐步相同的续训结果。
旧 policy/critic 不作为训练初始化；旧数据和结果不覆盖。

--resume 只复用 sealed COMPLETE **整个训练实例**，不从 n60000.pt 或 final.pt
恢复 optimizer/replay。失败或中断目录保留并拒绝自动重试；需审查后另建恢复
请求/version，不能删除失败 seed 或用新 seed 替换。

## 3. 冻结策略开发评价

```powershell
python -B tools/train_fixed_budget_ddpg.py run --request artifacts/p7n/p7n_120k_v1/request.json --confirm-request-hash HASH_FROM_PREPARE --stage evaluate --resume
```

必须先完成全部 12 训练实例。每个 checkpoint/scene 在新进程首回合运行，
Gym + 20s 预热、无探索、无 optimizer 更新、原 500 步限制、blocking exact。
两独立 worker，不和 Author50 混合，也没有 MPC/Shield 接管。
12 实例 × 4 节点 × 20 场景 = 960 回合；仍只有 20 个不同交通场景。
同训练种子内按 simulator seed 配对，再对三个训练种子等权汇总。
这组反复使用的开发场景不是独立确认集。

## 4. 聚合一次完整结果

```powershell
python -B tools/train_fixed_budget_ddpg.py aggregate --request artifacts/p7n/p7n_120k_v1/request.json --confirm-request-hash HASH_FROM_PREPARE
```

需要全部 960 个有效 sealed 回合；负结果正常完成。
aggregate 不覆盖已有结果，不要对完整聚合重复执行。
也不把训练 TensorBoard 的 evaluation/returns 标签当独立验证。

## 5. 输出位置与用途

artifacts/p7n/p7n_120k_v1/ 下：

- request.json / preparation.json：Git、启动命令、原始/解析配置、种子、runtime、
  预测器和 decision evidence 身份。prepared_not_trained 是准备时的不可变状态。
- train/<arm>_s<seed>/n20000.pt、n40000.pt、n60000.pt、n120000.pt 与同名 json：
  评价权重、actual steps、optimizer 更新次数、累计 replay/采样覆盖。
- replay.jsonl.gz、minibatches.jsonl.gz、losses.jsonl.gz、episodes.json、training_logs/：
  真实训练记录，和旧格式保持一致。
- evaluate/<arm>_s<seed>_n<node>_v<scene>/：动作、状态测量、原始终止事件、结果、receipt。
- training_curves.json：**小型曲线索引**，引用原始 episodes.json 和 losses.jsonl.gz
  并带 SHA256/count；不是旧版内嵌全部 loss 的巨大 JSON。读取索引后按行解析 gzip。
- development_curves.json：四节点真正的冻结策略开发结果。
- aggregate.json / aggregate_receipt.json：全部 seed、配对统计、120k 主终点与哈希。

不选择较好的中间节点替代 120k，不以 critic loss 降低宣称收敛。
重点比较 B2−B1 与 B3−B2，并同时审查到达、碰撞、任务期限、原回报、低速、
动作饱和和舒适性；全回合持续时间混合失败结局，不能直接当成功合流效率。
完成后作明确决策，不自动继续增加预算。

## 开发验收命令（不是完整训练）

```powershell
python -B -m pytest tests/test_fixed_budget_extension.py -q -p no:cacheprovider
python -B tools/train_fixed_budget_ddpg.py audit --run-id p7n_smoke_v1
```

audit 复用独立历史工程 smoke 配置（单种子、64 请求步、warmup/batch=8），
四个短训练与十二次冻结评价；不启动 120k。四独立快照边界另外由 synthetic
等价性测试覆盖。历史执行 kernel 的 p7k 标签不会改变新 request 的身份。
