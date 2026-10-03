# P7i：固定策略噪声与 DDPG 训练稳定性诊断

不重新训练；复用全部 12 个 P7 final.pt 和已有普通/条件预测器。
保持原奖励、连续 jerk、SUMO 设置和阻塞式同步计算。噪声仅用于诊断，
不能代替正式确定性评价，也不能与 author50 的外部基线表直接合并。
旧 P7c 文件仍是历史草案；本版本是独立、可执行的 P7i 实现。

## 1. 准备：只读历史结果，生成新请求

使用干净、已提交的工作树和原 pytorch 环境：

```powershell
cd E:\Prediction_RL
python -B tools/diagnose_ddpg_stability.py prepare --config configs/development/p07i_ddpg_stability_v1.json --run-id p7i_diag_v1
```

prepare 不启动 SUMO，不执行模型推理或训练。它验证 P7/P7b 原始来源与
完整收据，解析全部 12 个策略的 actor/critic loss 和训练回报，保存：

- training_audit.json：固定训练步数窗口、优化次数、最终权重范数等。
- training_curves.json：原始 scalar 序列，可供后续绘图。
- request.json / preparation.json：完整配置、代码/环境/依赖指纹及请求哈希。

返回 confirm_request_hash。已有目录不可覆盖；不要删除失败目录“重试”。

## 2. 用户运行有界诊断

把 HASH 替换为 prepare 输出的完整哈希：

```powershell
python -B tools/diagnose_ddpg_stability.py run --request artifacts/p7i/p7i_diag_v1/request.json --confirm-request-hash HASH
```

四组 B0/B1/B2/B3，训练种子 0/1/2，既有场景 200/210/219。
36 个确定性回放、每单元 3 个噪声重复共 108 个噪声回合，合计 144 回合，
最多 72,000 次控制，两进程并发。三个噪声重复不是三个新交通场景。
所有 checkpoint 固定；不选择较好种子，不训练、不更新 replay。

确定性动作/奖励/原结果必须与 P7 精确一致。两种条件都采集真实 SUMO
碰撞参与者、到达位置和移除事件，避免依赖原循环的 arrived 标志。
原奖励和终止逻辑不修改。异常保存为 failure 并停止，不当作普通失败回合。

run 自动完成分层配对聚合：

- aggregate.json / aggregate_receipt.json。
- episodes/<cell>/trace.json：原 actor 动作、噪声、裁剪、请求 jerk、实际
  执行审计、带 TimeFeature 的策略输入、速度以及冻结 critic 的诊断值。
- raw_events.json / decisions.json / episode.json / complete.json。
- invocations/<id>：子进程日志及完整/失败状态。

仅需继续未运行单元时使用 `--resume`；只复用收据和内容均验证通过的
COMPLETE 单元。失败/不完整单元不自动重试，现有 aggregate 不覆盖。

## 3. 如何判断下一步

1. 看逐训练种子的到达、真实自车碰撞、时间上限和其他终止事件，保留分母。
2. 看低速持续时间、原 actor 与加噪声后负 jerk 比例、裁剪和动作变化。
3. 看固定场景/checkpoint 下噪声是否解除停车，是否同时增加碰撞。
4. 联合检查训练回报、TD MSE 和 actor 的负 Q 值；loss 下降不是收敛证明。
5. Q 与执行轨迹折扣回报的差异是回顾性诊断；噪声改变后续策略分布，
   不能把它直接解释为无偏价值误差或真实因果效应。

若噪声能解除停车，支持探索/确定性执行敏感性，不证明预测器整体有效，
更不能把正式策略改为噪声策略以提高结果。若无帮助或变差，保留负结果。
下一轮统一增加四组训练预算或改变预测数据覆盖，需另行确认；不同时
修改模型、奖励和训练预算。现有 final.pt 不是完整 optimizer/replay 恢复点。

只有三个开发场景，不作显著性、泛化或作者方法排名结论。P8 未批准。

## 工程 smoke（开发验证，不代替完整诊断）

```powershell
python -B tools/diagnose_ddpg_stability.py audit --run-id p7i_audit_01
```

四组、训练种子 2、场景 200、确定性与一个噪声重复，最多 8 回合/4,000 步。
这是预先固定的失败路径覆盖，不作为方法效果依据。

## 本机已准备的完整请求（2026-10-03）

工程验收：738 项完整回归、8 回合真实 smoke、complete-only resume 均通过。
完整请求已经准备，尚未启动；无需再次 prepare 或重新训练，可直接执行：

```powershell
cd E:\Prediction_RL
python -B tools/diagnose_ddpg_stability.py run --request artifacts/p7i/p7i_diag_v1/request.json --confirm-request-hash eb52a72518773039aca4b6de215d2667fdec40ba02255886293b76cdd0c7299a
```

正常中断后需要继续时，在同一命令后加 `--resume`；失败/不完整单元依然
不会被重试或覆盖。完整结束后的报告：artifacts/p7i/p7i_diag_v1/aggregate.json。
验收记录：[P7i acceptance](../../reports/p07i_ddpg_stability_acceptance_20261003.md)。
