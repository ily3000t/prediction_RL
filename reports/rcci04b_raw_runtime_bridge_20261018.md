# RCCI04b：旧 Raw 权重运行时桥接验收

## 结论

**工程桥接通过，六个旧 Raw 策略可在已验证的纯 Gym20/DDPG 条件下复用。**
不需要为换到 RCCI 工作树而重训这些 Raw 模型；RCCI 新特征语义不同，不能加载 Raw
权重冒充已训练的 RCCI。没有训练、更新模型、修改奖励、改 seed 或覆盖历史产物。

本次仅固定一个已有开发场景（列表首个 seed=200），六个冻结模型分别加载历史/
当前 binary，共12个原生回合。它不是12个独立交通场景，也不是新的方法效果比较。

## 1. 来源、配置与实现

- 冻结契约：`docs/design/rcci04b_raw_runtime_bridge_v1.md`，设计提交 `b5fcec9`。
- 配置：`configs/development/rcci04b_raw_runtime_bridge_v1.json`，无嵌套。
- 实现/测试：`tools/bridge_rcci_raw_runtime.py`、`tests/test_rcci_raw_runtime_bridge.py`。
- 执行提交：`c3adea90af576a3e6ee539f667cd90531dbc5c69`，干净工作树。
- 只读 baseline 保持干净的 `75051acc06f6ea2ab6dda9be678f191b63d04569`。
- 父证据：RCCI04a Raw完整审计；本阶段不回写其当时的“待桥接”状态。

保留原 `train_matched_ddpg.evaluation_job` 的 code object，以 scoped globals 注入
明确的资产路径、透明记录器和审计session，不修改原函数或绕过旧workflow guards。
原strict loader核对policy/q全部参数、checkpoint contract和预测器SHA。

两种binary均读取baseline中原配置文件，其**实际字节SHA**就是旧contract的
`537423fe0bb27ea19bff87caaa4ab2388789bd2cac95c1cce3d9832beb142d8a`。
不将当前CRLF文件伪称为旧SHA；没有复制/重写binary或修改配置。
当前工作树中相同上游文本/地图承担仿真执行；所有输入路径与字节哈希保存至request。

历史extension SHA：`498a57f2096e905bb1151eb37dfde41cfb1b76668c343c16e55ae63fcebc76b3`；
当前extension SHA：`aeacc8434bb6c0cd9777105fe21f8e97d5d52908379ef8e467d3d4c986acc04a`。
每worker按绝对路径显式加载并复核真实`__file__`与SHA；无fallback或跨进程复用。

## 2. 精确比较与实际预算

| 冻结模型 | 每种binary控制步 | 每种binary仿真步 | 历史回归/跨构建 |
| --- | ---: | ---: | --- |
| ordinary seed0 | 134 | 235 | 全部精确一致 |
| ordinary seed1 | 150 | 251 | 全部精确一致 |
| ordinary seed2 | 143 | 244 | 全部精确一致 |
| conditional seed0 | 144 | 245 | 全部精确一致 |
| conditional seed1 | 150 | 251 | 全部精确一致 |
| conditional seed2 | 149 | 250 | 全部精确一致 |

每个worker与相应已封存seed200记录精确比较5份完整JSON：episode、actions、states、
raw_events、resolved_config，共60个文件级比较。包含所有动作/reward/done、观测SHA、
交通状态、原生终止事件、原配置、checkpoint与结果指标。

每个模型的新旧binary之间额外比较7份完整JSON（前五项加完整观测和实际policy输入），
共42项全通过。完整169维观测、170维policy输入（含TimeFeature）、mask、RNG、
执行审计逐步一致，terminal观测的预测块为零，时间特征与步数正确对齐。

| 项目 | 实际 | 冻结上限 |
| --- | ---: | ---: |
| sessions/episodes | 12/12 | 12/12 |
| 控制step | 1,740 | 6,000 |
| simulationStep | 2,952 | 7,224 |
| ensemble forward | 1,740 | 6,000 |
| member forward | 5,220 | 18,000 |
| policy forward | 1,752 | 6,012 |

每worker对32个本地ST/extension入口使用可恢复诊断guard，**全部0调用**。
guard若被调用会立即停止，不执行ST、不替换动作；不修改源码或安装包。
policy/q与预测器保持不变；优化器状态及replay均空，无exploration和训练调用。
六个模型均按与历史一致的自然到达终止，但不将这一个场景的结果报告成总体100%到达率。

## 3. 产物与独立复核

目录：`artifacts/rcci04b/raw_bridge_v1`。

- request digest：`82e86ec9a924149bd8ed5fed350dff1358a3272fa50afda3277904a24b9c5f17`。
- report SHA：`49d6418ac680792a0e0bb4d5de97d0297de13085f9e66fbd4c996988b85f4583`。
- index SHA：`52c897729637966d77b560026128e69769e96cb6c6ef44b67ac1787513494f49`。
- receipt SHA：`3c0409445bdd6b16fd4e42da5503281c822d4173bf540a57b0a9b83eb602ef0c`。

原入口封存后独立只读复核：receipt与request/report/index一致，120个子输出/日志文件
SHA、200个绑定输入资产SHA和42项跨构建完整JSON比较均通过。原baseline与当前
工作树结束时干净，原模型/结果不变。large artifacts、binary和日志均不进Git。

先行桥接/RCCI回归29 passed（9.79 s）；最终选定回归**168 passed（18.50 s）**，
无deselect，不称全仓测试通过。`git diff --check`通过。

## 4. 能与不能得出的结论

可以：在记录本桥接、原配置实际字节与旧训练来源的条件下，保留六个Raw权重作为
RCCI04匹配对照候选，不必重训Raw或预测器。其余开发场景/节点在新匹配请求下的
当前环境评价仍由用户执行；旧报告保留为回归参照，不伪称为新环境新运行。

不能：声称两binary字节相同、全部Cython/MPC函数等价、作者RL+MPC路径已由此验证，
或者RCCI有驾驶收益/已泛化。本次没有运行RCCI策略，也没有训练RCCI策略。

建议下一步直接进入匹配训练入口与不可变请求实现，不追加诊断场景、扩展网络、
增加奖励或延长预算。训练方案见 `docs/design/rcci04c_matched_policy_plan_v1.md`；
该方案需要人工批准，正式训练和长评价仍由用户启动。
