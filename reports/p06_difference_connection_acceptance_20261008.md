# 差分预测器接入与匹配闭环请求验收

日期：2026-10-08。工作树：`E:\Prediction_RL_worktrees\response-difference-v1`，
分支：`codex/p05-response-difference`。旧 `E:\Prediction_RL` 仅作只读来源；
没有修改旧源码、旧配置、训练报告、模型或评价结果，没有推送或合并。

## 已完成实现

- `1f13f26`：严格 difference ensemble loader、独立校准和原预测特征接口。
- `09ec81f`：仅新增差分 B3 的从头训练、冻结评价与五方法配对聚合入口；
  原训练/评价内核仅扩展显式可选 execution context，旧默认路径保持不变。
- `3ea8f2eae73ab4f13490c9503e6023d1412c5e3d`：上游配置仅 LF/CRLF 迁移的
  历史 Git 验证。旧字节哈希仍绑定旧 Git blob，新请求保存新字节哈希，
  不接受参数、奖励或额外空白变化，不重写历史证据。

沿用原 149 个预测通道、候选顺序、归一化、actor/history masks、连续 jerk
和上游奖励/终止/回放语义。没有候选吸附、动作屏蔽、MPC/Shield 接管或新特征。
旧 B3 策略不用于新方法初始化；相同训练种子的扩展策略初始参数须匹配 P7n。
新旧预测器的 policy checkpoint 由 predictor SHA 区分，禁止静默部分加载。

新 ensemble SHA256：
`eba3b43dad1425dad76569497cf3ab667761e84a72169c2fbdc96c8471302fbf`。
Git 忽略的 st_cy 二进制从旧工作树复制，未重新编译或覆盖旧文件；SHA256：
`498a57f2096e905bb1151eb37dfde41cfb1b76668c343c16e55ae63fcebc76b3`。
上游许可仍未解决，不发布其源码或权重。

## 实际验证

修复前的 smoke 在 SUMO/训练启动之前，被上游配置字节校验拒绝。逐项核查：
环境、依赖、SUMO、上游规范化源码、原请求、aggregate 和 receipt 均一致，
唯一不同为 Git core.autocrlf 检出的 LF/CRLF。该次没有创建训练目录。

原 LF 字节哈希：`537423fe0bb27ea19bff87caaa4ab2388789bd2cac95c1cce3d9832beb142d8a`；
新 CRLF 字节哈希：`29ee8dcdfa8165bee21d31f7abad87937c1707e16f4055737132f70c568142f6`。
两份配置与原 Git 内容逐字一致，仅换行不同。

最终回归子集 **190 passed in 22.32s**，命令：

```powershell
python -B -m pytest tests/test_difference_connection.py tests/test_difference_closed_loop.py tests/test_historical_checkout.py tests/test_candidate_difference.py tests/test_candidate_difference_cli.py tests/test_fixed_budget_extension.py tests/test_matched_learning_curve.py tests/test_training_contract.py tests/test_all_ddpg_features.py tests/test_probe_features.py -q -p no:cacheprovider --basetemp artifacts/tests/delta_closed_loop_20261008_03
```

涵盖严格模型/数据/校准绑定、候选批量等价、旧加载器拒绝新目标、匹配预算、
负结果正常聚合、初始场景/终止事件检查，以及换行迁移与非法语义改动拒绝。
`git diff --check` 通过。这是回归子集，不是声明执行了项目全部测试。

实际 SUMO audit：`artifacts/difference_ddpg/delta_smoke_v1/audit.json`。
请求 hash：`c4a54d689181f698007dd3de9d1301e7bc876bc4049e55eafa7285bd46f83588`。
一个请求 64 步训练 job，在原完整回合边界结束，实际 **100 步**、**93 次更新**。
16/32/64 三个工程节点均指向该 100 步边界，不能解释成三个学习阶段。
三个新进程场景 200 评价均为 55 步自然到达、无碰撞、原回报
0.434380761006729，初始交通哈希一致，策略/预测器无更新。
这实际只验证一个场景和一个短训练得到的策略，不构成方法效果证据。
`engineering_gate=true`，`formal_training_started=false`。

## 独立校准的实际结果与限制

产物：`artifacts/difference_calibration/delta_cal_v1/{request,calibration,report}.json`。
请求 hash：`2ed664cfe5a0000d640d83a07ac165197c289b9e5ce70c390e29af6b7b4a4aaf`。
64 个 calibration episodes 全部有效；原 episode-max standardized residual，
nominal 90%，rank 59，q=44.44401353597641，拟合集 episode coverage=92.1875%。
179400 个 actor-time cells；平均 band 全宽 x=89.10m、y=26.67m、speed=44.62m/s、
acceleration=19.01m/s²。区间很宽，不能宣称精确风险保证或独立测试覆盖。
在线 149 通道只消费 ensemble mean，**不消费 bands**；没有重新选模或打开测试集。

## 已准备但未运行的匹配实验

独立配置：`configs/development/p07_candidate_difference_120k_v1.json`。
请求：`artifacts/difference_ddpg/delta_120k_v1/request.json`。
hash：`e2c86b9f2310b2d9a49a6fe00c5c284a7071102ac153ad484f2b7f9c95e1d372`。
准备状态：`prepared_not_trained`，没有正式训练或正式评价 job。

- 新差分 B3：种子 0/1/2，各从头 120k，总请求 360k；CPU 一线程，训练串行。
- 20k/40k/60k 为诊断节点，120k 唯一主终点；不按结果选较佳节点或删种子。
- 原 lr=.0002、warmup=5000、batch=100、2M scheduler 不变。
- Gym20、新进程首回合、场景 200–219；新增 3×4×20=240 评价回合。
- 旧 B0/B1/B2/B3 的 960 个封存回合、48 个节点权重及 receipt 已验证，只读复用。
- 1200 条评价记录仍仅 20 个交通场景；不是新测试集或 1200 个独立样本。
- 主要增量：新差分 B3 − 原 B3，同时报告相对 ordinary/B0 的差异。
- 不混入 Author50、泛化、新奖励、Shield/MPC 或自动扩大预算。

完整启动命令见 `docs/runbooks/p07_candidate_difference_ddpg_v1.md`。
用户确认配置后手动依次 train、evaluate、aggregate。resume 只复用封存 COMPLETE
整 job，不从 evaluation-only 权重假装恢复 optimizer/replay。失败目录保留。
负结果也应正常完成聚合；离线差分改善是否转化成闭环收益尚待本次实验。
