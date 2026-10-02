# P7f 有界评价循环 × 预热诊断接入验收（2026-10-02）

## 范围与结论

用户已批准在P7e之后分离Gym/作者循环和20/50秒预热。
独立实现、测试、文档及有界SUMO接入通过；未运行完整42个新回合、未训练。
保留原环境/奖励/连续DDPG与全部冻结权重；不选择最好协议或训练种子，不进入P8。
原E:/WCDT_ACCVP只读，原执行源码/配置和结果均未修改，sealed test仍关闭。

## 变更与来源

- `src/prediction_rl/evaluation/driver_warmup.py`：冻结2×2协议、共同字段轨迹、配对聚合。
- `tools/diagnose_driver_warmup.py`：历史链验证、完整模型加载、新进程回合、请求与不可变收据。
- `configs/development/p07f_driver_warmup_v1.json`：独立可读配置，无extends、未知字段拒绝。
- `tests/test_driver_warmup.py`：33项新增回归。
- `docs/tasks/p07f_driver_warmup.md`及`docs/runbooks/p07f_driver_warmup.md`：阶段边界及手动命令。
- 源码原子提交：62d1798，codex/p07f-driver-warmup；仅新增文件，旧执行文件未改。

作者checkpoint全文件集/哈希验证后加载完整原模块，不部分移植。
B0/B3复用P7 final.pt，严格policy/q布局及预测器绑定，evaluation-only而非训练恢复。
历史链包含P7、P7b完整记录、P7d外部基线及P7e完整交叉结果和各自工程验收。
历史源码允许新增独立文件，但原执行文件必须逐一匹配Git来源与当前指纹。

## 验收实际运行

最终已提交源码上的命令：

```powershell
python -B -m pytest tests -q -p no:cacheprovider --basetemp E:/Prediction_RL/artifacts/test_tmp/p7f_final_20261002a
python -B tools/diagnose_driver_warmup.py audit --run-id p7f_audit_v1
```

完整回归602 passed；原作者可信pickle加载保留两条torch.load FutureWarning。
最初系统pytest临时目录权限失败，改用新项目专用目录后测试通过；没有修改环境依赖。
Python语法检查和git diff --check通过。测试/工程产物位于Git忽略的artifacts内。

工程运行只用场景200、作者模型和B0/B3种子0：

| 条件 | 新回合 | 引用旧回合 |
| --- | ---: | ---: |
| Gym20 | 0 | 3 |
| Gym50 | 3 | 0 |
| 作者20 | 3 | 0 |
| 作者50 | 0 | 3 |

6个新回合共1473次控制调用，低于预先定义的3003次上限。
请求准备时刻19:10:32，聚合完成19:17:07（北京时间）；不等于纯仿真计算耗时。
`artifacts/p7f/p7f_audit_v1/aggregate.json`：status=complete、engineering_complete=true，
method_effect_gate=null。模型、输入、TimeFeature、终止重置、replay/optimizer冻结检查通过。
三模型相同预热的初始共同交通全匹配；不要求两循环轨迹逐位完全相同才工程通过。

- audit request canonical hash：`29c6e0adae357b555803cfd2387bea66ede06e4da1f3bfdb3e80163d737de09b`。
- aggregate byte SHA256：`2496c9c8c50febbbc9479df84eb4bcf98129ac1f893c183522588ad332d6ee6b`。
- 源码指纹验收前后相同；没有活动实验期间编辑文件。
- 完整协议的42份历史轨迹已单独验证可复用，共10,558个实际决策记录。

## 语义与分析局限

Gym仅在首次reset前改实例wait_before_start；作者循环传原wait_before_start参数。
不修改原控制周期、jerk边界、速度/加速度修正、原奖励或碰撞/终止检查。
场景由独立新进程首回合执行，一CPU Torch线程；预测完成后才推进仿真。
实际query与同输入/同TimeFeature原网络forward逐位一致；终止查询不执行动作。

新回合记录真实policy observation与TimeFeature，禁止用历史缺失规则掩盖新记录缺失。
P7d作者/B3旧记录没有observation，保留null；时间特征按原单次实际查询序列重建。
Gym命令速度是原helper重建，作者循环是原controller返回，不冒充线级命令测量。
不计算跨driver奖励差；只在同driver下报告原评分的预热变化。
原碰撞定义及500/501调用不同，不能把native事件差称统一ego责任碰撞差。

工程场景中的作者模型初始jerk相同，第一步后速度差约9.67e-8/2.55e-7m/s；
全共同索引速度最大差约4.91e-5/2.04e-5m/s（50/20秒预热分别列示）。
因此exact字段会报告后续分歧，但不能仅凭不逐位相等宣称循环严重错误；
完整结果仍需检查差异量级、控制执行和实际任务结果。本例不是方法优越性证据。
历史未观测字段不重建、不插值；状态分歧后的动作差不独立证明某一因素的因果作用。

## 交接

完整协议为作者1份、B0/B3各3份模型，场景200/210/219，共84格；
Gym20/作者50复用42格，Gym50/作者20新增42回合。场景选择沿用结果前的P7b规则。
完整请求准备后由用户运行，无需训练、无需重复audit或旧结果。
只复用完整且哈希一致的回合；不删除失败目录、不自动重试、不修改标准。
P7c噪声诊断保持草案；模型扩大、训练预算调整、正式协议选择及P8仍需另行确认。
未创建可运行论文里程碑tag，未推送；上游许可限制仍保留。
