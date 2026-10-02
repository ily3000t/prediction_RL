# P7g v2 任务时限审计恢复运行说明

在E:/Prediction_RL，激活pytorch。无需训练，不再运行旧audit_terminal_events.py请求。
修复只涉及旁路分析；原作者连续控制、奖励和终止分支均保留。

## 来源与恢复范围

配置configs/development/p07g_terminal_events_recovery_v1.json，不支持extends或未知字段。
旧artifacts/p7g/p7g_diag_v1原样保留：72份已完成收据、2份计数分析失败原始记录。
两份失败重验后仍是time_limit负结果；不会把failure.json改为complete。
新source_manifest冻结全部原文件哈希、逐回合分析及原10个缺失任务。
正常分支分类不变；作者时限末次命令从真实移除前快照核验，无虚构simulationStep。

## 开发者准备（不运行SUMO）

```powershell
python -B tools/recover_terminal_events.py prepare --run-id p7g_recovery_v1
```

必须干净工作树。验证完整来源/权重/环境/旧轨迹与raw事件后准备不可变请求。
此命令已在干净源码提交cdbbf21上执行；用户不必重复prepare。

## 用户运行剩余10回合

```powershell
python -B tools/recover_terminal_events.py run --request artifacts/p7g/p7g_recovery_v1/request.json --confirm-request-hash a8fcf644880e745f5b6cb57f25b926d13b37cf474c08fcc280da18470fcd8988
```

只运行原10个未启动回合，2worker、一CPU Torch线程；74个旧回合不再仿真。
不要编辑配置/代码、安装依赖或并行训练。完整且收据一致的新回合可加--resume；
不完整/其他失败目录保留并停止，不删除、不自动重试。
run完成自动输出artifacts/p7g/p7g_recovery_v1/aggregate.json，含完整84格双口径统计。
负方法结果正常complete，method_effect_gate=null；不以新标签重算历史奖励。

## 产物

- source_manifest.json：原74个目录的逐文件哈希、v2分析、原缺失10个任务。
- evaluate/：仅10个新仿真回合，各有原始记录/analysis/不可变complete收据。
- aggregate.json：84格、72完整复用/2原始失败重验/10新运行，逐回合来源可追溯。
- invocations/：本次执行日志及实际命令/Git/状态；worker失败显示具体日志路径。
- 旧请求/原文件不修改；原失败状态仍保留，不用新收据伪装v1已成功。

## 已完成验收（2026-10-03）

完整回归692 passed（52项新增），原作者可信pickle加载仍有两条既有警告。
74份真实原始记录的来源/收据/轨迹/事件均已离线验证：72份分类与v1完全一致，
两份author20种子2场景200/210仍是time_limit，各501次控制、602次原始仿真步。
此次没有启动SUMO、没有训练；剩余10回合尚未运行。

request canonical hash：a8fcf644880e745f5b6cb57f25b926d13b37cf474c08fcc280da18470fcd8988。
source_manifest byte SHA256：442f287801c556dcb91cab6b05d67b8457d3f0a20a3453eca4f9b6272aaa4355。
验收报告：reports/p07g_time_limit_recovery_acceptance_20261003.md。
直接运行上方run命令；不要重复prepare，也不要删除或重跑旧p7g_diag_v1。
仅文档提交/普通merge不改变冻结源码指纹；运行时源码、环境、权重与原始证据仍需匹配。
