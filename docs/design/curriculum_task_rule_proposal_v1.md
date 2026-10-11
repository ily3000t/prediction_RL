# IC02g：任务级交互课程规则初稿（未批准，不可训练）

## 研究边界与来源

仅依据 IC02f 封存的 32 项训练侧任务/96 roots 提出一次规则并检查覆盖；不读取 B2/B3 控制结果或 predictor test，不补采、推理、训练或接入在线调度。所有源报告和现有模型保持不变。参数定义在运行覆盖检查前提交；覆盖不佳也完整聚合，不事后改规则凑层。

有限池为原生成参数下 seeds 11000–11248、步长 8 的 32 个 fresh Gym20 first-episode tasks，参考作者策略仅离线刻画。策略训练若后续获准，应从原 reset 的第一 observation 接管，不重放参考前缀。四组 B2/B3 × Mix/Curriculum 共用同一入口，从头训练；旧 B3 只作历史参照。

## 1. 每个 task 的统一刻画

- 保留 reset、early_ramp、approach 三个 root 的映射，不把三者算成三个任务。
- 主刻画 anchor 固定为 approach：原参考策略首次到 ramp 末端前 20m 的非终止检查点。它不是训练初始化状态，也不随正在学习的策略改变。
- I 和 D 均用该 anchor，不取最容易/最有利的 root。早期零响应用于描述启动阶段，不自动变成 C0。
- I 时间窗拟采用 14 ticks/2.8 秒：这是 IC02f 所有任务完整支持的观测上限。它是训练侧审计后形成的开发提案，不是先验最佳窗，不改原 predictor 的 25-tick 预测时域。
- 五候选、所有固定关键 actor、每个 tick 必须完整。合法不可计算响应单独记录，候选规则归 C2；源工程错误/数据被改则审计失败，不能冒充 C2。

## 2. I：保留关键角色，避免平均稀释

对每个 root 选定的关键邻车 r，先计算十个候选对、14 ticks 的平均响应：

```text
I_r = Mean_(candidate-pair,time) [ |dv_r| / old_c_v + |ds_r| / old_c_s ]
I_task = max_r I_r
```

道路坐标、有效性及 dv/ds 尺度都保持 IC01/IC02f 定义。它是“先平均，再在关键 actor 中取最大”，不是在所有 noisy cells 上取瞬时峰值。目标前车、后车及额外关键车都保留，稳定 ID 去重，role 重叠不重复加权。并列最大 actor 全部记录，不按 vehicle ID 改任务层。

max 避免弱响应第三辆车改变平均分母，也保留 t11240 类型的额外车辆主响应。但更多 actor 仍可能增加发现大响应的机会，不能声称彻底控制数量效应；需记录 actor 数、所有角色分量和最大响应来源。这是课程刻画提案，不修改模型摘要/observation，也不宣称新的预测收益。

阈值仅在 32 个 task 的 approach I 上拟合下/上三分位；不用 96 个 root、960 个分叉或开发评价回合作独立样本。噪声下限为原 approach 同动作重复分支的最大旧尺度 bound，C1 须严格超过该下限+1e-10。零/并列上分位不能自动成为明确交互。

## 3. D：机会与追近代理分开，不重新加权成难度分数

保留全部当前 D 分量。初稿 C0 的必要弱约束采用：

1. 目标前/后车都存在且为不同 actor。
2. 两个路线投影机会 gap 都严格大于 0：共同 merge-end 进展下不存在投影车长重叠。此条件只叫“开放投影机会”，不当作当前物理安全保证。
3. 目标关系最大接近速度不高于训练 task 的下三分位。非接近为 0，缺失为 null，不填任意大/小值。

初稿不把跨路径投影 TTC=0 当即时风险，也不把距离 merge-end 当 ramp 剩余距离。ETA 差、邻近 ETA 车辆数、原投影 gap、关键数和剩余距离继续完整报告，不事后为获得非空层选择权重。C1 的“少量关键交互”暂沿用已有最大三辆规则；它不是恰好一辆交互或低危险的证明。

## 4. 唯一规则与优先级

| 候选层 | 条件 |
| --- | --- |
| C0：基础控制 | 可计算 I≤下三分位，且满足上述三项弱控制代理条件 |
| C1：明确交互 | 可计算 I≥上三分位，严格超过重复噪声下限，关键 actor≤3 |
| C2：其余完整任务 | 不符合 C0/C1，或合法响应暂不可计算 |

C0 → C1 → C2 的优先级使条件并列时仍互斥完整；分位边界包含等号，gap 开放和超噪声用严格大于。禁止通过 seed/vehicle ID 打破并列以硬凑配额。训练调用无法使用这些提案记录：`proposed_level` 仅供 review，`task_level=null`、classifier=null、classification_applied=false、execution_authorized=false。

本次最低覆盖仍采用 IC01 已有的每层 10 个独立源 episode，按唯一 task 计数。它只是设计覆盖检查，不是训练有效性或统计功效保证。如空层或少于 10 项，保留结果并暂停；不得把三层比例重新归一化到剩下两层、拒收失败任务或按碰撞结果分层。

## 5. 有限池目标与 Mix 初稿

若后续正式批准，目标名称应为 `finite_train_task32_gym20_uniform_v1`：每 task 权重 1/32，层质量为候选任务数/32。这不是原连续交通状态分布，也不能将这 32 项描述为其无偏样本。

最后 20k 拟恢复该**有限池均匀 task target**，不是旧连续分布。其余 0–20k、20–60k、60–100k 的 60/30/10、20/60/20、10/30/60 比例暂保持原提案；Mix 名义配额仍按阶段长度加权。报告可以给出拟议层质量和 Mix 算术值，但 `target_distribution_pi` 仍为 null，分布未冻结。如果某层为空，前三阶段要求正采样质量，调度数学上不可执行，不能因为有限池目标里该层质量为零而忽略。

即使覆盖通过也须人工批准，之后再做独立 RNG 调度/replay 审计和有界工程 smoke。task 次数/transition 曝光和 replay 更新分布不能因此声称完全等价。

## 6. 实现、审计与交接

配置 `configs/development/ic02g_task_rule_proposal_v1.json`，纯模块 `curriculum/task_rule_proposal.py`，CLI `tools/propose_curriculum_task_rules.py`。只读核查原 32 个收据、IC02f 精确文件哈希和科学资产；输出不可覆盖的 `artifacts/ic/ic02g_rules_v1/`。

```powershell
python -B tools/propose_curriculum_task_rules.py prepare --baseline-root E:\Prediction_RL
python -B tools/propose_curriculum_task_rules.py run --request artifacts/ic/ic02g_rules_v1/request.json --confirm-request-hash <prepare打印的hash>
```

运行后检查工程完整性、候选层计数、条件交集及采样可执行性。负覆盖也应 status=complete，而不是删 seed 或调阈值重跑。真正的分层冻结、正式 manifest、调度器、SUMO smoke 和训练均不在此次授权内。

## 实际检查结果

本初稿已在干净实现 commit `15ef568…` 完成一次离线检查，508 项测试通过。候选层为 C0=1、C1=11、C2=20，工程 gate=true，但 C0 未达每层 10 项覆盖，coverage_gate=false。所有 source/输入/RNG 验证通过，0 新仿真/推理/训练；没有正式标签或分布冻结。不得改本版本规则重算成通过。具体数值、产物哈希及需用户选择的后续方向见 `reports/ic02g_task_rule_proposal_acceptance.md`。
