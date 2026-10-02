# P7g 任务时限审计修复与只读恢复验收（2026-10-03）

## 范围与结论

用户批准修复v1任务时限旁路审计错误，复用现有原始证据，只补跑原缺失回合。
实现提交cdbbf2148822c8a7c0471c8afce6946c01cbfb47，分支codex/p07g-time-limit-recovery。
仅新增独立v2分析器、恢复入口、配置、测试和文档；v1执行源码、上游、奖励、动作、
终止分支、训练种子、模型、旧请求及全部原始产物未改。E:/WCDT_ACCVP只读。
本轮没有启动SUMO或训练；10个剩余回合由用户执行。没有推送或P8批准。

## 确认的缺陷与修复

原作者循环达到任务时限时共501次控制调用：第501次命令后立即remove_ego_car，
再执行一次cleanup step并break，不执行该次命令对应的ego动力学步。
20秒预热包含100步，插入1步、ego演进500步、清理1步，总602步，非v1要求的603步。
50秒条件对应752步。原生simulation_duration仍按501×0.2=100.2s记录，不重新计算原评分。
最后一次实际决策状态绑定记录到的移除前快照，不能用移除后无ego的step快照核对。

独立terminal_events_v2.py严格检查501次控制、单一移除、原始连续时间、真实最终状态、
移除/cleanup边界及无增删步；正常分支输出与v1一致，不制造虚拟step或放宽轨迹parity。
旧v1源码保留，历史来源验证只允许新增Python文件，每个旧执行文件仍匹配Git指纹。

## 已有证据重验

源请求：artifacts/p7g/p7g_diag_v1/request.json。
canonical hash：1d4e443824312f75d3b46831c41d9dc938892892ed005231bf6e3d408b67fcf2。
原执行提交：e327fb38ddf1ca7e4f329256a3b5e585087d2965。
原failed invocation：artifacts/p7g/p7g_diag_v1/invocations/15ba5bc958d9/report.json。

| 原状态 | 数量 | v2验收 |
| --- | ---: | --- |
| complete | 72 | 收据/文件哈希/权重来源/原结果/全部可得轨迹/事件逐项匹配，分类不变 |
| 只在classify计数校验失败 | 2 | 完整raw与trace重验通过，仍为真实time_limit负结果 |
| 未启动 | 10 | 原任务身份保持，纳入独立新请求，不复用不存在的raw |

两份恢复记录为B3训练种子2、author20、simulator seeds200/210，各501次控制、602步。
全部动作、原评分、TimeFeature和已有observation与历史参考严格一致，没有记录到ego碰撞。
原failure.json继续保留，未向旧目录补写analysis.json或complete.json。
只有指定两份完整post-simulation计数失败允许恢复；缺文件、其他错误、损坏收据、
不同来源/环境/权重、活动writer或异常新增文件全部拒绝，不自动重试。

新source_manifest保存74个目录的全部原文件哈希（含旧complete或failure）、v2分析、
原请求/参考/执行报告/失败日志哈希，以及恰好10个剩余任务。正文与SHA双重冻结，
启动、每个worker及聚合均重新校验；不将多回合或训练seed冒充独立场景。
完整结果仍仅三个开发场景200/210/219，不能当作正式方法优越性确认。

## 实际测试

```powershell
python -B -m pytest tests/test_terminal_event_recovery.py tests/test_terminal_events.py -q -p no:cacheprovider --basetemp E:/Prediction_RL/artifacts/test_tmp/p7g_recovery_unit_20261003b
python -B -m pytest tests -q -p no:cacheprovider --basetemp E:/Prediction_RL/artifacts/test_tmp/p7g_recovery_final_20261003a
```

针对性版本89 passed；随后补充84格聚合来源测试，最终完整回归692 passed（52项新增），
2 warnings为原作者可信完整policy/q的torch.load警告。AST语法检查与git diff --check通过。
最初新单元测试发现损坏元数据错误类型和系统版本查询mock隔离问题，均修复后重验，
未调整控制语义、种子或原数据。74份真实记录的完整离线重验证及干净提交上的prepare均通过。
测试文件、原始数据及新清单均在Git忽略的artifacts内，不提交大文件。

## 已准备交接请求（未仿真）

在干净提交cdbbf21上执行：

```powershell
python -B tools/recover_terminal_events.py prepare --run-id p7g_recovery_v1
```

新请求：artifacts/p7g/p7g_recovery_v1/request.json。
canonical hash：a8fcf644880e745f5b6cb57f25b926d13b37cf474c08fcc280da18470fcd8988。
source_manifest byte SHA256：442f287801c556dcb91cab6b05d67b8457d3f0a20a3453eca4f9b6272aaa4355。
preparation状态prepared_not_simulated，72完整复用、2原始记录重验、10新任务，
source_artifacts_modified=false、training_started=false，无evaluate/aggregate产物。

剩余任务：author20/B3种子2/场景219一回合，以及author50/B3三个训练种子×三个场景九回合。
复用原三成员预测器、DDPG权重、连续动作、原奖励和原执行函数字节码，2worker/一CPU线程。
用户命令见docs/runbooks/p07g_time_limit_recovery.md；不必重复prepare、旧audit或训练。
run完成自动聚合新aggregate.json，完整84格明确区分72复用、2恢复、10新运行的证据来源。
负结果正常complete，method_effect_gate=null，不用新事件标签重算原奖励。
不完整新回合仍停止并保留；--resume仅复用完整且验证一致的新收据，不重试失败。

此次是工程错误修复，不是模型效果提升。完整双口径统计完成后再判断终止逻辑修订，
不直接删除失败训练种子、增加模型或扩大训练。原子本地提交、普通merge、不推送，许可限制不变。
