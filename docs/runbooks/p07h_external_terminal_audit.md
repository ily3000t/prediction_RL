# P7h：原作者DDPG/ST/RL+MPC统一事件审计

在E:/Prediction_RL使用pytorch。无需重新训练；旧结果保留不覆盖。
固定作者原50秒预热、100秒任务上限、原奖励、连续控制及ST设置。
这是开发评价，不是正式方法优势或碰撞责任归因结论。

## 范围

开发场景200-219，共20个独立simulator seeds。

| 冻结模型 | 评价回合 |
| --- | ---: |
| 作者预训练DDPG | 20 |
| 作者ST | 20 |
| 作者RL+MPC safety | 20 |
| 作者RL+MPC switching | 20 |
| 项目B0无预测DDPG，训练种子0/1/2 | 60 |
| 项目B3动作条件预测DDPG，训练种子0/1/2 | 60 |

总200格，不能称200个独立交通场景。
先复用21条P7g author50原始事件记录；有界接口审计另产3条ST/MPC记录。
完整请求再复用这3条，因此共24条复用、176条新仿真。
模型全部复用已有冻结checkpoint，不改变训练种子或选择最佳checkpoint。

## 有界开发验收

```powershell
python -B tools/audit_external_terminal_events.py audit --run-id p7h_audit_v1
```

仅场景200：7格中4格旧raw复用，3格新仿真。此命令需干净源码。
不重复运行已完成审计；如有失败，保留现场，不删除或自动重试。

## 准备完整请求（不运行SUMO）

```powershell
python -B tools/audit_external_terminal_events.py prepare --run-id p7h_external_v1
```

验证原20场景参考、模型、历史收据、完整P7g、接口审计以及原始事件后，
冻结不可变request/source_manifest并打印confirm_request_hash。
已准备的请求不要重复prepare。完成验收后实际hash会记录在交接信息中。

## 用户运行完整评价

```powershell
python -B tools/audit_external_terminal_events.py run --request artifacts/p7h/p7h_external_v1/request.json --confirm-request-hash <prepare打印的实际hash>
```

命令中的占位值必须替换。2个独立进程worker，每个Torch一CPU线程。
不并行训练/修改源码/配置/原结果。运行结束自动聚合，不再另行aggregate。
断点续跑仅对已验证完整回合允许加--resume；不完整或失败目录直接暂停，
不会删除或覆盖，也不会静默换seed。

最终artifacts/p7h/p7h_external_v1/aggregate.json，包括：

- 原生奖励及标签，独立事件标签，逐回合来源和严格参考parity。
- 各方法/训练种子的自然到达、ego碰撞、任务超时及异常类别。
- B3对作者四组及同训练seed B0逐场景配对，不报告伪独立显著性。
- 双方都自然到达时的完成时间/jerk差异，不把碰撞结束当快速完成。
- 模型负结果仍status=complete、method_effect_gate=null。

自然到达只代表有记录的终点一致、无已记录ego碰撞，不是完整安全保证。
任务超时是SUMO仿真任务时限，不是CPU wall-clock超时。
原作者arrival/collision分支没有修改，奖励也未重新计算。

## 后续边界

完成此报告后再诊断DDPG训练稳定性、B1/B2/B3特征利用。
本轮没有证明20k收敛，也没有开启P8、重新训练或加大预测网络。
