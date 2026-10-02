# P7g 原始终止事件审计接入验收（2026-10-02）

## 范围与结论

用户批准在P7f诊断后记录原始终止事件。独立实现、完整回归及12回合有界接入通过。
源码提交ba0c7daeb125d2078019509417a5364d902a6407，分支codex/p07g-terminal-events。
本次只新增审计模块、入口、独立配置、测试和文档；原执行源码、奖励、动作、终止分支、
模型、训练种子和旧结果未修改。E:/WCDT_ACCVP保持只读，没有训练或正式测试。
完整84回合诊断由用户启动；P8、修订作者终止逻辑及任何重训均未批准。

## 实现与来源

- src/prediction_rl/evaluation/terminal_events.py：原始事件记录、严格轨迹等价及旁路事件分类。
- tools/audit_terminal_events.py：历史链验证、不可变请求/收据、单writer及用户手动执行入口。
- configs/development/p07g_terminal_events_v1.json：原三场景、全部冻结模型的84格协议。
- tests/test_terminal_events.py：38项新增回归，包含异常恢复、不增加step、到达/碰撞重叠等。
- docs/tasks/p07g_terminal_events.md、docs/runbooks/p07g_terminal_events.md：阶段边界和真实命令。

局部包装原control.step，原调用恰好执行一次，返回后、清理或额外simulationStep前读事件。
只在项目session中代理control.traci的vehicle.add以捕获实际到达位置参数；不修改安装包。
保存前后ego/live IDs、arrived IDs、colliding IDs、碰撞对象、实际路线和显式remove边界。
复用原P7f执行函数字节码并注入局部session，不改旧模块的全局函数。

依赖链含完整P7/P7b/P7d/P7e/P7f；旧执行源码、运行环境、权重、收据及轨迹逐项验证。
父P7f canonical request hash为a87464d61fcf154bbc81444267ff8bf10e13ab80c09a077d72ccd2e45d76b917，
aggregate byte SHA256为9a1a37d30ec0c2181999570f9766655bc5a28179f968215ec56a57cf2f302d9e。
原observation未记录的历史片段仍为空，不用重建数据冒充旧观测。

## 实际验收

```powershell
python -B -m pytest tests/test_terminal_events.py -q -p no:cacheprovider --basetemp E:/Prediction_RL/artifacts/test_tmp/p7g_unit_20261002a
python -B -m pytest tests -q -p no:cacheprovider --basetemp E:/Prediction_RL/artifacts/test_tmp/p7g_all_20261002a
python -B tools/audit_terminal_events.py audit --run-id p7g_audit_v1
```

单模块38 passed；完整回归640 passed、2 warnings（原作者可信完整policy/q的torch.load警告）。
AST语法检查和git diff --check通过。测试及SUMO产物均在Git忽略的artifacts内。
audit在干净源码提交ba0c7da上执行，12个回合全部新运行，未复用旧raw事件。
控制调用2078/上限6006；包含预热、插入及原清理的仿真步4196/上限8136。
执行记录北京时间21:43:37至22:03:08，约19.53分钟；含来源检查、进程启动和推理，
不是纯SUMO运行时间。执行结束无writer.lock，聚合和收据一致。

| 冻结模型（仅场景200） | Gym20 | Gym50 | 作者20 | 作者50 |
| --- | --- | --- | --- | --- |
| 作者预训练DDPG | 到达/自然到达 | 到达/自然到达 | 到达/自然到达 | 到达/自然到达 |
| 本项目B0，训练种子2 | 碰撞/ego碰撞 | 碰撞/ego碰撞 | 到达/ego碰撞 | 到达/ego碰撞 |
| 本项目B3，训练种子0 | 到达/自然到达 | 碰撞/ego碰撞 | 到达/自然到达 | 到达/ego碰撞 |

各格为原生标签/独立事件。12/12原生结果、原评分、步数及可得轨迹严格等价。
共6回合ego碰撞与arrived flag同时出现，3回合被作者循环原生标签记录为到达。
工程状态complete、engineering_complete=true，method_effect_gate=null。

直接证据例：B0种子2、Gym20的raw step 223同时给出arrived IDs与colliding IDs
为[traffic_11800, ego]；碰撞对象collider=ego、victim=traffic_11800，type=junction，
lane=:mergenode_1_0，position=30.77486808776861m，之后ego消失。
对应作者20回合仍沿原逻辑记录arrival=1、collision=0，但旁路事件为ego_collision。
两回合分别保持各自旧轨迹/原评分完全不变，不能把错误归因于新审计改变了控制。

## 解释边界

这证实本机SUMO碰撞移除的arrived事件与作者循环到达优先判断会掩盖ego碰撞。
但是B3在同一场景20秒预热自然到达、50秒预热真实碰撞，也说明预热敏感性确实存在；
不能把所有协议差异仅解释为标签错误。这里只有一个独立交通场景，不能做模型优劣统计，
也不能据此直接改写此前20场景报告的数值。

ego_collision表示原始事件记录的参与者包含ego，不认定事故责任。
自然到达要求原始arrived flag、未记录到ego碰撞且实际路线终点一致（1m一致性余量）；
不是现实安全保证。背景碰撞、异常移除和任务期限独立分类。
不以独立分类重算奖励，不合并两循环评分，不选有利协议或删除失败种子。

## 产物与交接

验收aggregate：artifacts/p7g/p7g_audit_v1/aggregate.json。
canonical request hash：23e6ba60e593ca295d1cff0bbde85b8322d9052220c9c4937f8616f12ea856f1。
aggregate byte SHA256：a370762a05759ae535d0a6f4ee554d4910e3288f4fc131d3413eba1827eb7134。
真实执行记录：artifacts/p7g/p7g_audit_v1/invocations/124f920357e7/report.json。
逐回合raw_events.json、episode.json、trace.json、analysis.json及complete.json保留原始证据。

下一个prepare仅冻结请求，不运行SUMO：原三个场景200/210/219、作者一份权重、
B0/B3各三个训练种子、四条件，共84个新raw审计回合；历史轨迹仅作等价参考。
固定两worker、每worker一CPU Torch线程，阻塞同步计算；用户按runbook手动启动。
完成后统一检查双口径事件，再决定是否批准后续终止语义修订，而非立即扩大模型。
本地原子提交、不推送、不创建论文复现里程碑标签；上游许可限制仍保留。
