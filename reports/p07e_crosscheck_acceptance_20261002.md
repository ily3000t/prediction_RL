# P7e 交叉检查工程验收（2026-10-02）

## 范围及结论

模型来源×原评价协议的独立入口已实现，场景200四个新增组合工程验收通过。
完整80回合仅prepare，未运行；旧P7/P7b/P7d结果和配置未改，未训练任何模型。
P7c仍未实施，P8未批准，sealed test未开放，旧E:/WCDT_ACCVP只读，无Git推送。

实现提交：`ac8ee844343c4723807a6c37bb8627d04550d891`，
功能分支`codex/p07e-model-protocol-crosscheck`，从main `f78d3c2`开始。
后续验收文档独立原子提交；不发布作者源码/权重。

## 新增文件和接口

- `configs/development/p07e_model_protocol_crosscheck_v1.json`：完整独立配置，未知字段或语义变化拒绝。
- `src/prediction_rl/evaluation/model_protocol_crosscheck.py`：六格矩阵、协议内配对、原事件口径的协议敏感性。
- `tools/crosscheck_model_protocols.py`：audit/prepare/run/aggregate，完整哈希、收据和只读历史来源校验。
- 新测试、阶段任务文档、运行文档。原src/tools执行文件全部未改。

原P7全部240回合与12份最终checkpoint，原P7d全部140回合和旧9回合工程
验收链重新核验。跨表只引用200回合，没有把作者权重重复使用当作独立训练。
允许增加新审计文件；原执行文件、上游配置、依赖、SUMO和资产指纹必须保持不变。

作者Gym接入使用原GreedyAgent＋TimeFeature及完整固定policy/q模块；
原模块反序列化前校验完整目录与已知SHA256，未使用部分权重加载。
本项目B0作者循环完整加载各原final.pt契约，仅使用观测codec，不调用Gym reset/step。
原奖励、连续jerk边界、执行器、控制步长与两种原终止定义保持不变。

每个实际动作都与同输入/同时间的原网络forward逐位一致；旁路forward不推进时间，
实际eval恰好一次，终止mask把时间清零，权重/replay/optimizer不更新。
这不是双实现整回合独立重跑证明，而是实际轨迹上的接口/时间/冻结检查。

## 实际测试

最终全套569 passed、2个原作者torch.load提示，18.94秒。
命令：pytorch Python `-m pytest tests -q -p no:cacheprovider --basetemp artifacts/pytest_p7e_final_<uuid>`。
新增34项覆盖固定80/200/280和4/10/14工作量、完整作者模块、原观测布局、
TimeFeature/terminal、源文件新增与旧文件不变、历史/新结果身份、模型/predictor绑定、
评分隔离、半成品保留不重试以及负结果正常聚合。
CLI --help、py_compile和git diff --check通过。未安装或修改任何依赖。

## 真实SUMO有界验收

执行：`python tools/crosscheck_model_protocols.py audit --run-id p7e_audit_v1`。
干净实现提交ac8ee84。请求：`artifacts/p7e/p7e_audit_v1/request.json`。
request canonical hash：`50e460cd01413055b73716d3d3b6cb91b858af2142337e129f5e2c6c200d1277`。
aggregate：`artifacts/p7e/p7e_audit_v1/aggregate.json`。
aggregate字节SHA256：`92d60b0fe1183bf3fcac5f3dc97e94330282721d8f5b7103efc0bcfdb58ee12b`。
完成时间：2026-10-02 11:01:52（北京时间）。

| 新增组合（仅场景200） | 实际控制次数 | 到达 | 碰撞终止 | 时间上限 |
| --- | ---: | ---: | ---: | ---: |
| 作者预训练DDPG，P7 Gym20 | 149 | 1 | 0 | 0 |
| 本项目B0训练种子0，P7d author50 | 327 | 1 | 0 | 0 |
| 本项目B0训练种子1，P7d author50 | 345 | 1 | 0 | 0 |
| 本项目B0训练种子2，P7d author50 | 119 | 1 | 0 | 0 |

合计940次，低于事前上限2003次。四个完整收据、输入/动作/时间检查、终止重置、
冻结检查及协议内与历史初始交通哈希一致性通过。另10条历史记录只引用。
工程通过不以到达为门槛；四个小样本恰好到达不能代表20场景效果或因果解释。

## 用户完整运行交接

prepare已执行：`python tools/crosscheck_model_protocols.py prepare --run-id p7e_cross_v1`。
请求：`artifacts/p7e/p7e_cross_v1/request.json`。
确认哈希：`f50f32e772fa47c8195e39ecf93fb8153cf989720f2e05be5260840e82bab5de`。
新增80回合（作者Gym20＋本项目B0作者循环60），历史引用200回合，20个共同seed编号200–219。
完整请求没有evaluate产物或活动运行；不用训练、不重复prepare。
用户启动的精确命令见runbook。本轮没有创建80回合结果或称已完成效果比较。

## 限制

两套初始交通、终止/碰撞条件和奖励汇总不同，跨协议只解释整体敏感性，
不将其单独归为预热变化。跨协议交通哈希编码也不同，不比较这些哈希的物理等价。
同协议内要求全部方法初始交通一致；同checkpoint在两协议间严格绑定。
奖励差只在各协议内部计算，跨协议不计算reward delta。
作者一份预训练模型与本项目20k训练预算不匹配，不是正式优越性/收敛检验。
B0/B3输入维度不同，预测信息增量仍需原B1/B2/B3同预算对照。
不用失败seed筛选、噪声替换、旧模型覆盖或不同协议混合奖励来改变结论。
