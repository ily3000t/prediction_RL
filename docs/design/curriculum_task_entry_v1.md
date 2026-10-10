# IC02：课程任务入口设计（IC02b 有限原型已批准）

## 1. 决策摘要

推荐候选协议 **`seeded_fresh_gym20_task_v1`**：每个 task 用显式的训练侧 seed 启动全新 SUMO，调用原 Gym reset，等待原 20 秒，在原位置插入 ego，DDPG 从第一个原始 observation 开始控制。环境随机数只在独立 worker 中播种；父进程中的策略、优化器、replay、探索噪声和课程 RNG 持续存在。

IC02a 已完成静态核查，其不可覆盖报告与旧 proposal 配置保持原样。随后用户在明确的 IC02b 入口和 12 回合/144 控制步问题后回复“继续”，批准原型与有限 smoke；来源记录在 `provenance/ic02b_smoke_authorization.json`。新入口实现位于 `envs/fresh_task.py`，运行规范见 `docs/runbooks/ic02b_entry_smoke.md`。该预算现已执行；12 项封存轨迹 parity 精确通过，但原进程在 JSON 键表示聚合错误后退出，未采集最终内存 tensor/RNG postflight。原失败保留，独立离线报告不冒充完整成功；详见 `reports/ic02b_entry_parity_acceptance.md`。**这不批准课程训练或新任务采集。**

这一入口不自动解决 C0=0。IC01 证实旧近合流 root 分布不适合原三层交集规则；入口一致性与任务层覆盖是两个不同的 gate。没有覆盖与任务分层证据前，不提供课程训练请求。

## 2. 原代码的实际入口能力

| 事实 | 源码依据 | 影响 |
| --- | --- | --- |
| SUMO 在构造器启动，reset 不重启 | `merge_gym.py` 的 `JerkEnv.__init__/reset` | 普通 reset 延续交通状态，不能按名称选择冻结任务 |
| reset 无 task/seed 参数 | 同上 | 给 reset 传 task 或只改 seed 均不是当前行为 |
| 等待 20 秒、插入一步 | 同上，`control.add_ego_car` | 0.2 s/tick 下是 100 个等待 tick 加 1 个插入 tick |
| 原 route/位置/执行模式 | `rampRoute`、departPos=40、arrivalPos=50、speedMode=22 | 不移动自车到中段 root，不换路线或底层模式 |
| 交通生成是持续状态 | `control.step` 的 delay 与 Python random | 当前交通不是下一次 seed 可以重建的样本 |
| 会话播种 Python/NumPy/Torch | `envs/upstream.py` 的 `upstream_session` | 不得在 learner 进程每回合调用它来切任务，否则会重播种学习随机数 |
| 学习型输入 reset 时清历史 | `envs/prediction_features.py` | 第一个预测 observation 仅有 1 帧，不免费补齐 11 帧 |
| ALL 使用独立的 TimeFeature | `training/all_ddpg.py` | 保留尺度 0.001、首个决策时间零及既有终止处理 |

实际配置是 `RL-MPC-LaneMerging-master/configs/train_default_1.json`，不是仅用源码默认值：Slotted Jerk、碰撞/成功 -10/+10、时间权重 -0.1、jerk 权重 0.1。简单车流 interval=1.2、随机 jitter、邻车速度 7 m/s；起步速度均值 15、标准差 5、裁剪到 5–25 m/s。虽然变量名为 `START_SPEED_VARIANCE`，它实际作为 `np.random.normal` 的标准差传入。这里不更正变量或参数，以保留原语义。

## 3. 备选入口及取舍

| 路线 | 能保持什么 | 缺口或混淆 | 第一版建议 |
| --- | --- | --- | --- |
| 原连续交通 reset | 旧训练会话语义 | 无冻结 task 选择接口；状态受前面回合耗时及行为影响；不能直接实现固定课程层配额 | 保留历史基线，不伪装成可调度任务入口 |
| fresh seed + 原始 Gym20 reset | 原动作/reward/位置/首个控制时刻；同 task 可检查复现 | 将持续交通改为每 task 首回合，改变训练任务分布 | 推荐，但作为新命名协议、四组共同使用 |
| fresh + 参考前缀恢复近合流 root | 已有 root 历史与反事实证据 | 跳过启动控制、改变剩余期限/TimeFeature/历史及控制责任；前缀不能免费 | 不选择为默认方案 |
| 每层保留一个持续 SUMO 上下文 | 上下文内部交通持续 | 切换上下文是新协议；task 分层可能随时间和策略失效；旧 seed 不能标记全部后续回合 | 不选择为第一版 |
| 新车流密度/车辆行为/道路 | 可增加覆盖 | 混入场景变化和冻结预测器 OOD，尚未验收 | 暂不引入 |

首次实现只验证原配置、原训练侧 seed 下的 fresh 入口。不得在看到某个场景难或不满足 C0 后调流量、丢弃 seed 或将 C1/C2 人工改为 C0。

## 4. 新协议边界

### 4.1 环境与控制

1. 同一 task 的定义含 seed、冻结配置/网络哈希、first-episode 标识和入口版本。启动环境 worker 的新会话；不在运行中的交通流写新 seed。
2. 只调用原 reset；不改 20 秒预热、插入位置、起步采样、控制周期、动作限制、奖励、自然终止及原 500-step 时间限制。
3. DDPG 从原 first observation 接管；参考策略只可用于离线任务刻画，不给训练 episode 提供动作前缀。
4. 历史从原 1 帧开始并按原 mask 增长；普通/条件预测均复用原三成员冻结 ensemble。无新 token、特征或课程标签输入。
5. 环境异常、进程异常或非法输入使运行无效并保存原因；不通过 synthetic done、stale、自动换任务或 reward 修改掩盖问题。

### 4.2 进程与 RNG

- 父进程是持久 learner：DDPG 只初始化一次，uniform replay、target、优化器及探索状态不随 reset 重建。
- 单独课程 RNG 只选择 task，不使用 learner 或原交通 RNG。
- 环境 worker 拥有上游模块、CWD、TraCI 与 Python/NumPy/Torch 会话 seed；上游 globals 不进入父进程。一次只打开一个会话，episode 结束后关闭，下一 task 新建 SUMO。
- 普通/条件模型及原 feature adapter 可在父进程保持冻结，用 worker 返回的真实当前 extended frame 增长 history。传输是观测通道，不运行真实未来分叉。
- StepObserver 的原始事件在 worker 捕获并随审计信息返回；父进程不能再假定自己拥有 SUMO。需要核验 deadline 的清理额外 tick、移除原因及 reset/store 边界。
- 未确认接口前不改既有训练脚本或 `upstream_session`，不在单进程反复会话播种来绕过隔离要求。

### 4.3 步数、终止及输出

120k 指原计数语义下的真实控制 transition；20 秒等待、插入 tick、启动 wall-clock、任务采样调用、参考刻画和 reset 都不是学习 transition，另行记录成本。阶段切换和 checkpoint 仍只在真实 episode 结束后执行，不截断、伪造 terminal、清 replay 或重置优化器。

短 smoke 的 12-step 上限只是工程采样停止，不能写成 MDP 的失败、期限终止或训练 truncation。完整训练后仍需区分请求步数、实际 transition、ALL 调用数、等待/插入/cleanup tick 与每层曝光。

所有输出只在本工作树的短 run ID 目录下，保留失败与 hash。不写原基线或调用旧 120k 请求代替课程请求。

## 5. Task 与 root 的映射：尚未冻结

IC02a 只把 713 个旧 root 关联回 256 个 source episode，并记录每个 episode 的参考 root 层；不将“出现过 C1 root”自动变成 C1 task。C1 root 来自 181 个 episode，这仍不证明它们是单关键交互任务。

待核查的新 task manifest 至少包含：

```text
task_id, scene_seed, episode_index=0, entry_contract
upstream_config_hash, network_hash, generation_parameters
initial_traffic_hash, initial_observation_hash
reference_controller_identity, reference_checkpoints
root_interaction_evidence, support_and_censoring
task_level, level_rule_version, predictor_input_coverage
engineering_validity, unavailable_checkpoints
```

分层依据是冻结的任务定义与训练侧参考刻画，不是当前 B2/B3 的评价胜负。只有自车几何变化仍不足以标记明确交互。按当前 root 代理比较的跨路线 gap 也不能当作实体净距；任务级 D 与 I 的道路位置、支持、参考检查点和聚合优先级需另行冻结。

入口 smoke 不解决分层。后续如需新训练侧分支采集，先提出固定 seed/检查点/参考策略/重复/步数预算的独立版本，再获批准；不直接使用现有分位数填出 C0，不把不可达 reference checkpoint 当作低响应，不依据训练失败删除合法任务。

## 6. 目标分布 pi 的命名

新入口的 base 分布应叫 **原生成器参数下的 fresh-episode task target**，不能等同旧 continuing-traffic 的状态分布。旧连续分布还依赖回合顺序、时长和学习策略。

`pi` 当前保持 null。任务池、task 内采样和目标权重确认前，不能用 root 的 0/211/502 比例、每层均匀比例或 20 个开发场景计算它。最后 20k 返回新任务池的冻结 target 需用户确认其含义；不能宣称恢复了旧持续交通分布。Mix 的 pbar 公式保持原设计，但须以最终批准的 target 权重计算。

旧 B3 只作历史参照；B2-Mix/B2-Curriculum/B3-Mix/B3-Curriculum 全部使用同一批准入口、任务池和预算，从头训练。新的共同入口改变了与旧 B3 的可比范围，不改变四组内部的共同控制条件。

## 7. IC02b 有限 smoke（已执行，封存 parity 通过，完整 postflight 缺失）

旧静态提案 `configs/development/ic02a_entry_design_v1.json` 保持不可执行。新批准范围单独冻结为 `configs/development/ic02b_entry_smoke_v1.json`，不会把旧提案的 authorized 字段直接改为 true：

- train seeds 11000/11001/11002，原配置，非开发评价 seeds。
- 原 direct 与拟新增 worker-entry 两条路径，各重复 2 次：最多 12 个工程 episode。
- 每条路径固定 12 个原始连续 jerk；最多 144 个控制步，另有最多 1,212 个等待/插入 tick。遇真实终止则保留并提前结束，不补步或换 seed。
- 每 predictor mode 最多 156 次当前状态特征调用（普通与条件均使用冻结 ensemble）；没有 predictor 或 DDPG 训练，没有新候选未来分叉。
- 对齐首个/后续 observation、实际交通、requested action、reward、done、事件、有效 history 和相同候选摘要；检查 task RNG/worker reset 不重播种父进程 learner RNG。
- 终止传输另用已有事件 fixture 核查碰撞、自然到达、deadline cleanup；12 个控制步的 live smoke 不宣称覆盖了所有实际终止情形。

验收要求是两条路径在相同 seed/动作下语义一致，重复可复核；不是任务完成率或 B3 必须胜出的 gate。此阶段不运行任何 120k 训练。

## 8. 前置条件、批准和当前状态

本工作树没有自动继承忽略的 `st_cy.cp310-win_amd64.pyd`；原基线有该扩展。原 `st.py` 在 `USE_CYTHON` 时导入它，不能悄悄关闭该设置或从旧目录导入模块来“跑通”。批准 IC02b 后，需要在新工作树按已固定来源构建兼容扩展或明确验收同来源本地构建资产，记录依赖与哈希，原基线不动。

当前：IC02a 离线核查完成；IC02b 实际 12 回合/144 控制步的封存结果经过独立零仿真复核，12 项精确比较通过。原 `failure.json` 和缺失的最终 postflight 如实保留，本次预算不能重跑或更名续跑。C0/C1/C2 task 定义、target pi、课程调度与训练仍未通过。入口 prototype 是 single-task，不宣称已集成或验证持久 learner/replay。IC02c 已完成 task 参考刻画的纯离线设计与实际审计，见 `reports/ic02c_task_design_acceptance.md`；下一步需要批准三任务 collector 工程范围及完整预算，不自动开始新采集。
