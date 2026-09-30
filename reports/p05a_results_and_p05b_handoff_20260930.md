# P5a 结果复核与 P5b 手动诊断交接

日期：2026-09-30。P5a 已由用户执行完成；本次仅复核其输出，实现 P5b、
运行轻量单元测试和既有开发快照验收。没有执行正式 P5b 验证、SUMO、
数据采集、重新训练、DDPG 或独立测试集评价。旧项目未修改。

## 1. P5a 来源与完整性

- 输入请求：`artifacts/offline/predictor_v1_diag2/request.json`
- 请求规范化哈希：`2c13c83000fae4f54cb6b382bc715d4fbfa32881d0a36399407047ac28d73d08`
- 请求文件 SHA256：`b73b75030feae8040372caa2a292bb7405acc76d795fcfd2990af1225e054a37`
- 输出：`artifacts/offline/predictor_v1_diag2/report.json`
- 报告 SHA256：`69c290a7564ca19b6b2bce0f12e32a1d38d3d788c8e6b376cdb50ea315f44a4a`
- 实际运行 commit：`6bf352ca0641687f5a20599f158e82bc698f1f83`，干净工作树。
- UTC：2026-09-29 07:47:01.036061 至 07:47:10.203209，约 9.17 秒。
- `status=complete`；`test_evaluated=false`；`ddpg_ready=false`。

已校验历史源码、原训练/数据/校准来源、三个方法的逐 root 指标与汇总、
两组配对差异、validation/calibration 的逐 root 区间归因与汇总。
没有通过覆盖旧报告或修改参数重新生成“更好”结果。

## 2. 冻结预测器的验证结果

以下为 root 等权指标。验证集为 64 个原始 episode、177 个已到达 root，
176 个有邻车监督；5 秒 FDE 仅有 135 个有效 root、4500 个完整 actor 分支。
这些不是 177 个独立场景。验证集参与过 checkpoint 选择，不是独立测试证据。

| 指标 | 车道链恒速参考 | 普通预测 ensemble | 动作条件 ensemble |
|---|---:|---:|---:|
| ADE (m) | 0.2500 | 0.1647 | 0.1220 |
| FDE，5 s (m) | 1.0893 | 0.6716 | 0.4278 |
| 最后可观察时刻位置误差 (m) | 0.7231 | 0.4422 | 0.2746 |
| 速度 MAE (m/s) | 0.1733 | 0.1247 | 0.0888 |
| 加速度 MAE (m/s²) | 0.1705 | 0.1616 | 0.1350 |
| 同 root 候选响应差异 scaled MSE | 0.05597 | 0.05597 | 0.03367 |

条件预测相对普通预测：ADE 降低 25.96%，5 s FDE 降低 36.30%，
候选响应差异误差降低 39.84%。episode 等权 ADE 分别为 0.2445、
0.1642、0.1234 m，改善方向一致。恒速与普通预测均不接收候选计划，
故其邻车预测差为零，响应差异误差相同符合设计。

这支持“条件信息有预测价值”，不能直接推出候选排序或闭环收益。
也不能声称所有通道均最好：恒速参考 y MAE 为 0，普通与条件预测分别
约 0.00942、0.00895 m。此项原样保留，不为消除差异修改参考或数据。

## 3. 宽区间的原因与限制

validation 与 calibration 两个 split、两种模型的每组 64 个 episode，
其最大标准化残差全部由 **acceleration** 通道产生。
实例：校准 root `0790f8fe130258233a1024279a2a0a6c6f7213afdcf97e13bfa97284c6dd4ba1`，
`traffic_9600`、候选 0、3.8 s，真实加速度 −9 m/s²，条件模型均值
约 +1.7802 m/s²，成员标准差约 0.1925，小于 0.2 的尺度下限，
得到约 53.90 的标准化残差。可见 ensemble 分歧未覆盖这类共同预测偏差。

既定的单一全通道 episode-max q（普通约 46.18、条件约 48.05）
随加速度残差放大，再同时乘到位置通道上：

| 验证指标 | 普通预测 | 条件预测 |
|---|---:|---:|
| episode 全观察坐标覆盖率 | 53/64 = 82.81% | 62/64 = 96.88% |
| x 平均区间全宽 (m) | 92.44 | 96.32 |
| y 平均区间全宽 (m) | 27.71 | 28.83 |

校准样本本身两者均为 59/64 覆盖，不能将校准集覆盖率作为独立泛化验证。
这些区间虽有可审计定义，但几何决策分辨力不足。暂不更改 q、下限、
目标通道或 checkpoint，也不将其投入安全动作判定。若后续改校准，应另建
明确版本、重新讨论评价与独立确认，不能静默覆盖本次负结果。

## 4. 本次新增 P5b

- 配置：`configs/development/p05_candidate_ranking_v1.json`
- 实现：`src/prediction_rl/prediction/candidate_ranking.py`
- 手动入口：`tools/diagnose_candidate_ranking.py`
- 规范：`docs/tasks/p05_candidate_ranking.md`
- 运行手册：`docs/runbooks/p05_candidate_ranking.md`

仅根据 root 与候选 jerk 重建共享自车轨迹；普通、条件、恒速预测使用
相同自车轨迹，避免把真实未来自车位置泄漏进预测输入。
比较已选择邻车的最小前保险杠欧氏距离排序，统计误差、两两排序、
选择后 proxy regret；所有方法在五候选相同 actor/time 交集上比较。
同时报告完整 5 s 与截短共同支持两种视图、空集合、遗漏/新增车辆。

**这只是 proximity proxy，不是车身净间距、碰撞概率、任务可行性、
最优动作或安全保证。** 它可能偏好停车，不能直接作为新 reward 或策略。
未来 mask 仅用于离线评价，不是可部署 observation；三成员均值用于比较，
没有将本次宽置信带重新解释为有效安全区间。

## 5. 实际测试和开发验收

- 全套测试：**375 passed in 14.76 s**。
- `git diff --cached --check`：通过。
- 实现 commit：`d8682838064635c1bad07209f026749def8b1cc7`。
- 验收：`artifacts/p5/p5b_ranking_v1_01/report.json`。
- 验收 SHA256：`5e8dce8ba0baea3ec53b3a82271d7b3802cf1c46e3ded199fb2a11c6c8c4b6ad`。
- `status=complete`，`interface_gate=true`，干净实现版本运行。
- 既有 9 开发 root、749 个有效自车时刻：最大位置误差 0.00004960 m，
  最大速度误差 0.00001907 m/s，最大加速度误差 0.00008774 m/s²。
- 只有 2 个 root 的全部选中邻车在五候选下都有完整 5 s；9 个 root 均有
  未单独选中车辆和新进入车辆。因此验收不代表全场景安全覆盖。
- 验收仅说明接口、执行近似与度量可运行，未据此选择模型或调整参数。

## 6. 用户下一步

准备已经完成，尚未运行完整验证。
请求：`artifacts/ranking/predictor_v1_rank1/request.json`。
文件 SHA256：`401e7ccc8b8aa72ba5771baa486d085d7b440cabec21e2937c5faa237f522fc9`。

```powershell
cd E:\Prediction_RL
python tools/diagnose_candidate_ranking.py run --request artifacts/ranking/predictor_v1_rank1/request.json --confirm-request-hash 9322bc20652539c076c79e111f474167915e452ef61936ad41ae08c12a3b6da3
```

完成后查看 `artifacts/ranking/predictor_v1_rank1/report.json` 与 `roots.json`。
先看自车 parity 与分母，再比较两种支持视图的条件预测增量和近似并列程度。
负结果正常聚合；`ddpg_ready=false` 保留，不自动进入 P6/昂贵 DDPG。
正式独立测试集仍封闭。不重新训练，不推送；上游许可发布问题仍未解除。
