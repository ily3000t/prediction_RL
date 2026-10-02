# P7g 任务时限审计修复与恢复

## 目标、前提与范围

用户批准修复v1审计误判，保留原请求与全部产物。源请求1d4e4438…有72份complete、
2份完整原始数据但分析失败、10个未启动回合。E:/WCDT_ACCVP只读，无训练/推送/P8。
只新增独立v2分析器、恢复入口、配置、测试及文档；不改v1执行源码/配置/报告。

## 实现与验收

- 作者时限分支501次控制，最后命令后立即remove，cleanup替代最后的ego动力学步。
- 20秒预热总602步，50秒预热总752步；严格验证单一移除及最后真实控制状态。
- 其他分支与v1分类输出一致；不制造补步、不放宽parity、不改原评分。
- 原72份收据逐项验证；仅两份指定的post-simulation计数失败允许只读重新分析。
- 缺数据、其他失败、来源变化、额外文件、活动writer均拒绝恢复，不自动重试。
- 新请求冻结原文件哈希和74份新分析，旧目录不写complete/analysis，不伪装旧成功。
- 新仿真只安排原10个缺失任务，模型/seed/条件/奖励/执行字节码保持原样。
- 单元测试、完整回归、74份真实离线重验证通过后提交并prepare，不启动剩余SUMO。

## Git与交接

codex/p07g-time-limit-recovery；原子Conventional Commits、审查暂存、普通merge。
配置configs/development/p07g_terminal_events_recovery_v1.json；命令与产物见新runbook。
原始负结果正常聚合，method_effect_gate=null。工程修复不是策略性能改善。
