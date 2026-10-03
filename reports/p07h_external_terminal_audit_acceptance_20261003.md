# P7h 外部基线统一事件审计验收（2026-10-03）

## 结论

实现、单元/回归测试、历史来源核验和有界SUMO审计通过。
完整开发评价请求已准备，未启动176个剩余回合，也未训练或开放sealed test。
工程通过不是方法胜出：原碰撞与任务停驶仍按独立原始事件报告。

## 修改范围及Git

仅新增external_terminal_audit模块、audit_external_terminal_events工具、
独立P7h配置/测试/任务/运行文档；没有修改P7-P7g执行源码、旧配置、原始结果。
E:/WCDT_ACCVP保持只读。没有修改上游或安装包、奖励、连续jerk、TimeFeature、
SUMO步长、终止分支、MPC接管逻辑、训练种子或checkpoint。

- 功能分支：codex/p07h-external-terminal-audit。
- 实现提交：68befcabedc76629d3060b5168b90e325dc62ac5。
- 任务/运行协议提交：721295d00ffc2feadcfe45f10d088a40b3d93c97。
- 本地原子提交，不推送；上游许可仍未解决。

## 实际测试

完整tests回归727 passed，2条原有可信作者模型pickle加载FutureWarning。
命令：python -B -m pytest -q -p no:cacheprovider --basetemp <独占临时目录> tests。
新增35项测试覆盖冻结设计、全部种子/外部控制器、失败seed完整性、
native奖励保留、模型绑定、记录器parity、MPC多查询、ST无checkpoint、
缺失历史显式标记、自然到达条件效率、原raw复用与不完整目录禁止重试。
默认pytest临时目录访问受限，改用新项目artifacts下独占临时目录，
没有修改系统权限；完成后清理本次两个测试目录，实验文件未删除。
git diff --check及staged diff检查通过；未暂存模型/raw/log。

## 历史来源验收

P7、P7d、P7e、P7g来源链、历史Git文本指纹、当前环境、SUMO/runtime、
模型完整文件哈希、逐回合complete收据与aggregate receipt验证通过。
完整P7g recovery84格可重算，200条P7h历史参考均可绑定旧P7d/P7e结果。
21条P7g author50场景200/210/219原始事件记录与相应旧native参考等价。
这些记录包括碰撞和停驶，不因失败而重跑或删seed。

## 实际有界SUMO验收

请求：artifacts/p7h/p7h_audit_v1/request.json。
请求hash：2a1a292fcfaed97e1dae550df3b130beca64d56744a66bb33c0f26ea6b67fba7。
aggregate SHA256：3ab513c2705d939c6555ab6bf82a0f4012141f4d2e9a3c24e32d7d2753ac830c。
执行：python -B tools/audit_external_terminal_events.py audit --run-id p7h_audit_v1。
实际执行源码提交721295d，干净工作树；只新运行3回合，其余4格复用。

| 场景200诊断对象 | 来源 | 独立事件 | 仿真任务持续时间 s |
| --- | --- | --- | ---: |
| 作者DDPG | P7g原始事件 | 自然到达 | 28.8 |
| 作者ST | 新有界仿真 | 自然到达 | 31.4 |
| 作者RL+MPC safety | 新有界仿真 | 自然到达 | 37.4 |
| 作者RL+MPC switching | 新有界仿真 | 自然到达 | 37.4 |
| 项目B0训练种子2 | P7g原始事件 | ego碰撞 | 23.8 |
| 项目B3训练种子0 | P7g原始事件 | ego碰撞 | 70.8 |
| 项目B3训练种子2 | P7g原始事件 | 仿真任务超时 | 100.2 |

这是有意覆盖已知异常分支的接口验收表，不是代表性方法效果排名。
7格均与相应旧native结果和可用轨迹严格等价。
ST157次控制；两种MPC各187次，每步5次DDPG查询，接管序列保持一致。
MPC接管率各0.0106951871657754；不会把5个虚拟查询当成5个实际仿真步。
原奖励未按独立事件重算。到达/碰撞重叠按ego碰撞报告，但不改变作者分支。
100.2秒保持作者501次控制调用的原时限语义，不是CPU推理超时。

## 完整用户请求

artifacts/p7h/p7h_external_v1/request.json已准备。
hash：1d0a82748693e00aa3105bf2325248a981a25c39832c6f91b6e17a811f99f855。
manifest SHA256：91aff7c22e597cd85ff0ffb8b9855d478b8e9c226b59a9df87ddfd7f11d6d4c2。
原50秒预热、原奖励、阻塞闭环；场景200-219。
作者DDPG/ST/两个MPC各20回合；项目B0/B3各3训练种子×20回合，共200格。
21条旧P7g raw＋3条接口审计raw复用，共24；仅176条需要新仿真。
source_manifest逐参考文件哈希、每条复用收据/分析来源和待运行名单冻结。
状态prepared_not_simulated；没有evaluate/invocations，不需要重复prepare。

## 限制和下一步

只有20个独立交通场景，不把200格或重复作者快照当独立样本。
作者训练预算与项目20k未匹配；没有正式显著性/胜出gate。
自然到达并非完整安全保证，ego参与碰撞并不等于事故责任判定。
负方法结果正常聚合，配对效率只看双方独立自然到达的回合。
当前仅审计外部系统与B0/B3，不替代B1/B2/B3匹配维度信息增量比较。
先由用户完成此次统一20场景评价，再确定训练稳定性/特征利用诊断。
不自动扩大预测网络、改奖励、换seed、挑checkpoint、重训或启动P8。

用户运行命令见docs/runbooks/p07h_external_terminal_audit.md。
