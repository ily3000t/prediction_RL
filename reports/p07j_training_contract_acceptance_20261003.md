# P7j 前置审计结果与验收（2026-10-03）

## 结论与边界

有界训练语义审计通过；本次没有发现需要先修复的 reward、done、
bootstrap、replay 动作或 transition 对齐偏差。但这不证明旧 20k 模型
已经收敛，也不证明旧训练与独立评价的完整状态分布一致。

已经确认连续交通训练与新进程首回合评价存在设计层面的初始条件差异。
冻结模型在后续回合能部分恢复到达，说明协议/状态敏感性值得重视。
没有按有利结果切换协议，未更换 reward、网络、种子或 checkpoint。
旧 20k 未保存完整 replay、状态采样日志和原始训练终止事件，无法追溯
其低速、恢复加速及早期匝道的真实 minibatch 覆盖。此项保持未知。

## 执行与证据

- 新诊断实现 commit：062403e753a20c30d5c035645f1825468e7263e8。
- 追加回归测试 commit：c76c7386886acf5c124bc534df51fc44a5a80025。
- 命令：`python -B tools/audit_training_contract.py audit --run-id p7j_audit_v1`。
- 输出：`artifacts/p7j/p7j_audit_v1/aggregate.json`。
- request hash：63a5e0d39dd98511c0fff8864977866512648a2f6aba2c8eed603ce72d988336。
- aggregate SHA256：8ec67ea67d9be2f29238accb6563488636322ae4426c89292a9cbc12b38de7a9。
- 7 个子任务收据、聚合文件哈希和不可变 P7 父证据均独立复核通过。
- 运行使用干净 Git 源码，原 P7/P7b executable 与旧 artifacts 未修改。
- Python 3.10.16、Torch 2.5.1、ALL 0.5.3、CPU float32、每 worker 1 Torch 线程。
  完整 SUMO/package/编译模块身份在 request.runtime 中，并验证原 P7
  environment 和 runtime_identity 未变。
- 审计开始 22:13:57、结束 22:17:39（Asia/Shanghai）；约 3 分 42 秒。

## 1. 训练语义

四个极短 optimizer smoke 的实际步数为 B0=173、B1=144、B2=87、B3=173：
共 577 个 transition、550 次 Critic 更新、5 个真实训练终止事件。
采用原完整回合预算边界，因此实际步数不是请求的 64。

这是专门的工程 smoke，显式使用 replay_start_size=8、minibatch_size=8。
不能替代原 5000/100 正式训练设置，不比较这四个短任务的学习效果，
也不把它们的到达/碰撞比例与旧 20k 结果直接比较。

检查结果：

- `s_t, a_t, r_(t+1), s_(t+1)` 与原环境和 TimeFeature 逐 transition 对齐。
- 最后终止 transition 确实进入 replay；没有跨 episode reset bridge。
- replay 动作逐项等于实际提交给环境的 jerk 命令。
- 550 次实际 Critic TD MSE 与原 masked TD target 完全相同。
- 实际 minibatch 共采到 8 个 terminal 样本，bootstrap 均为零。
- 577 个训练 reward 按原 Slotted Jerk 实现复算，最大绝对误差为零。
- 本次 4 个碰撞、1 个自然到达与独立原始 SUMO 事件一致。
- 内部期限在冻结回合中另有真实执行；可控测试覆盖普通步、碰撞、
  到达、内部期限及外层 TimeLimit。内部期限保持原额外 cleanup step，
  不因两次 simulation step 就错误重算成双倍单步奖励。
- 合成原 ALL 训练的插桩/不插桩版本在动作、replay、参数、Adam、
  target network、scheduler、Torch 和 NumPy RNG 上完全一致。

572 个可测 jerk 样本中，10 个命令/实测差值超过描述阈值 0.001；
最大差值 7.2272 m/s^3。核查这些样本的速度命令均被原速度下限限制为
零，SUMO 回报的加速度变化不再等于请求 jerk。不是 replay 保存错动作，
不能用实测 jerk 替换原 MDP 的命令动作。

实际 DDPG discount_factor 为 ALL preset 的 0.98，不是 Settings 中的
DQN DISCOUNT_FACTOR=0.999。上游 DDPG 入口同样没有传入该 DQN 字段，
所以本轮保留 0.98，不把配置同名字段误判为执行值。

原 reward 无额外超时失败罚分。理想静止、jerk=0 时每步 -0.02，
500 步 undiscounted return 约 -10，gamma=0.98 的 discounted return
约 -1。这提示原目标可能存在停车局部策略，但不是代码错误证明，
本轮没有改 reward 或 discount 来挽救结果。

## 2. 连续回合与新进程首回合

固定旧 B1/B3 的训练种子 2，simulator seed=200，每个 checkpoint 连续
执行三个 Gym20 回合；无探索、无更新。两者首回合的动作、reward 和
done 都与原 P7 完全一致。六回合实际共 2402 个控制步。

| 模型 | 回合 | 开始仿真时刻/s | 初速/(m/s) | 背景车数 | 实际终止 | 低速状态比例 |
|---|---:|---:|---:|---:|---|---:|
| B1 零通道 | 1 | 20.2 | 7.745 | 13 | 任务期限 | 33.20% |
| B1 零通道 | 2 | 140.6 | 24.555 | 28 | 自然到达 | 0% |
| B1 零通道 | 3 | 209.8 | 18.559 | 32 | 自然到达 | 0% |
| B3 条件预测 | 1 | 20.2 | 7.745 | 13 | 任务期限 | 78.60% |
| B3 条件预测 | 2 | 140.6 | 24.555 | 28 | 自然到达 | 27.32% |
| B3 条件预测 | 3 | 197.4 | 18.559 | 37 | 任务期限 | 68.80% |

低速描述阈值为 0.1 m/s。该表是单流 reset 敏感性诊断，不是三个
独立场景，更不是 B1/B3 的正式配对方法比较。第三回合因前一回合
完成时间不同，两个模型已不面对相同完整交通状态。

初始位置和初始加速度没有额外改变：位置均为
(-211.4084, 19.9870)，加速度 0。不同的是随机初速、背景交通累计、
具体邻车状态和全局仿真时刻。每次都预热原 20 秒，B3 每次都重新
从 1 帧预测历史开始，随后积累到 11 帧，不存在“后续回合保留了
上一个回合完整预测历史”的接入错误。

## 3. 二十个冻结开发场景的初态

仅执行 reset，不推进策略 rollout：200--219 全部 20 个初始交通
hash 与原 P7 完全相同，确认没有静默改变 Gym20 评价初态。

- 背景车 11--13，均值 11.75；自车初速 5.00--18.50 m/s，均值 13.25。
- 初速分层：<=10 为 3 个，(10,15] 为 10 个，(15,20] 为 7 个。
  无 >20 的初态；不因此更换已冻结的场景种子。
- 初始自车位置固定、加速度 0、历史 1 帧；ego/front/rear 历史有效
  比例均 1/11。front/rear presence 均为 1。
- 2/20 初态存在容量外 actor，原 omitted summary 正常启用；不据此
  声称已验证全部潜在关键车辆、或认定 overflow 导致停车。

另核查到上游行为：reset 初速使用全局 NumPy normal，replay.sample
使用同一个全局 NumPy choice。训练中的采样会影响后续初速 RNG 状态；
冻结连续评价不进行 replay.sample，因此不是完整的训练交通序列。
实际安装版本的合成回归已验证此耦合：seed200 不采样时前三个初速
为 7.745/24.555/18.559；一次真实 replay.sample 后为
7.745/24.555/13.893。第二次 normal 的缓存值仍相同，之后才受影响。
这是继承的原行为，本轮没有改 RNG、seed 或 reset。

## 验收、限制与下一步

完整回归：758 passed，2 条既有作者 torch.load pickle warning。
新增源码 AST 检查及 Git whitespace 检查通过。上游源码、原训练源码、
旧实验产物、旧 WCDT 项目未修改；没有推送或生成新的模型 checkpoint。

工程结论：有界 transition 语义及 Gym20 初态一致性通过。分布结论：
已确认设计差异和冻结策略敏感性，但没有唯一归因，也没有旧 replay
覆盖的完整证据。aggregate 保持 historical_20k_replay_coverage_gate=null，
training_60k_authorized=false、training_60k_started=false。

建议用户审阅后批准一次匹配 60k 开发实验，同时从新运行开始保存
真实 replay 状态/采样覆盖和独立冻结策略曲线。保留四方法、种子
0/1/2、原奖励及 Gym20 二十场景；从头训练，20/40/60k 节点按完整
回合边界记录。最终 60k 是终点，不能自动延长，也不能根据此次连续
回合诊断改成更有利的评价条件。此报告不提供尚未实现的训练 CLI。
