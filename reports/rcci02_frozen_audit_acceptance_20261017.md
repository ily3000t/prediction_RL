# RCCI02 六根冻结预测审计：结果与停止点

## 身份与范围

源码提交 `d670047d44a8f566cc1a5b5b9dc8445142edfc8a`，功能分支
`codex/road-conflict-consequence-interface-v1`，工作树干净时 prepare/run。
配置：`configs/development/rcci02_frozen_audit_v1.json`。
命令与环境完整保存在 request/report 中，本机 pytorch CPU 单线程。

- 输出：`artifacts/rcci02/rcci02_bnd_v1/`；status=complete。
- 请求 hash：`02d6730a98e3a3a1a3a3065fb9a2d66ff49967ceed10addd3d40a608313d3166`。
- report SHA256：`6c5f8f9ba0b35089d4996fc3b5fdbf487cdbc65398fb4c0685730e1bb22415a4`。
- roots SHA256：`f30cc77926e44d82648e8c227e4fe8306a5e86f86f4dcee909c3f7f938244741`。
- 固定 train 前三根 + validation 前三根，分别只来自 **1 个 episode**；
  collection seeds 分别沿用 11000 与 12000，不新增/替换 seeds。
- 每根五候选、12 个 root 物理邻车，12 次 ensemble 前向（内部各三成员）。
- 只读严格核验原 ordinary/conditional 选定三成员权重、epoch/receipt，未重选模型；
  没有新的 optimizer seed，没有 SUMO、训练、calibration/test target 读取。

运行后独立只读复核：71 个本次输入文件哈希、历史模型来源、source/environment、
请求/输出 receipts 和 root 顺序均一致；由 roots 重新 aggregate 与报告完全一致。
六根不是六个独立场景；actor/candidate/time 统计不是独立样本，以下没有置信区间
或显著性结论。不能据此选择训练 seed、阈值、模型或论文主终点。

## 1. 个体覆盖较高，但完整摘要支持下降

下表将两 split 合并，只作此有界审计的描述；分母包含全部固定 root/candidate，
不删除坏预测。跨来向候选只以 applicable 的 20 个为分母。

| 项目 | B2 ordinary | B3 conditional |
| --- | --- | --- |
| 有效未来投影点 | 8985/9000（99.83%） | 8916/9000（99.07%） |
| 完整 5 s 物理 actor 轨迹 | 355/360（98.61%） | 344/360（95.56%） |
| 完整 following 摘要候选 | 25/30（83.33%） | 21/30（70.00%） |
| 完整跨来向摘要候选 | 15/20（75.00%） | 11/20（55.00%） |
| 可计算完整 signed margin 的候选 | 0/20 | 0/20 |

这是 v1 的严格完整池掩码带来的可用性代价，不能只用 99% 的点覆盖率声称
接口已经适合在线训练。当前保持规则，不放宽半径、不忽略难算车辆、不改成有效子集
后仍称完整池。

本次无效点主要是 **projected progress 非单调**，并非 0.5 m lateral/off-path
阈值：B2 15 个 route_reversal 点；B3 83 个 route_reversal 点和 1 个 negative_speed 点。
计数包含发生一次倒退后失去完整窗口资格的后续点，不是 83 次独立倒车事件。
目前没有保存倒退幅度分布，因此不能将它直接归为浮点噪声，也不能称已确认真实倒车。

只读回查已有 source history 与逐 actor 审计显示，无效轨迹涉及 root 的
target_front / target_rear（B2 的 5 个坏 candidate-actor 轨迹均为一根中的
target_rear；B3 的 16 个中有 7 个对应这两种角色）。这些 actor 的 root speed
并非全为零，不能将其直接视作无关停车背景而丢弃。

所有六根都有 omitted actors（6、8、10 / 8、11、14）。未给 omitted token
伪造位置；这仍只是被选中物理车辆的几何支持，不是完整车辆集合的安全证明。

## 2. 误差：有积极信号，但非全面改善

以下仅在原生实际存在的连续前缀中评价。两模型 progress 误差使用相同共同有效点；
event 时间误差是各自在共同可投影前缀中匹配到事件的条件误差，必须同时看 FP/FN。
未到达、已占用/左截尾、未离开与投影失败没有被填成“安全 0”。

| MAE | train B2 | train B3 | validation B2 | validation B3 |
| --- | ---: | ---: | ---: | ---: |
| 沿路 progress / m | 0.16568 | 0.14654 | 0.23646 | 0.15231 |
| 进入时间 / s | 0.04101 | 0.04656 | 0.05777 | 0.03052 |
| 离开时间 / s | 0.01370 | 0.00849 | 0.02786 | 0.02920 |
| 观察时域内 overlap / s | 0.01099 | 0.01187 | 0.01737 | 0.00974 |

支持数分别为：train progress 3133 点、进入 31、离开 12、cross-pair overlap 112；
validation 为 3974、35、33、113。点、事件和候选高度相关，不是独立 episode。
完整 margin/进入次序支持均为 0，报告为 null 而非误差 0。

train 的进入事件：B2 TP/FP/FN/TN=31/2/1/83，B3=31/1/1/84；
validation 两者均为 35/0/0/72。观察 overlap 的 train B2=47/2/1/62、
B3=47/1/1/63；validation 两者均为 60/0/0/53。
这些是有限控制体积时序事件，不是碰撞检测准确率。

两 split 都没有完整、非左截尾、可比较的 signed margin 支持，因而也没有证据证明
该字段已有实际贡献。原生完整未来占用窗口完全落在 Raw 四采样点之间的计数本次为 0；
不能据此证明四时刻已足够或全数据不存在短暂事件。

自车原参考 rollout 在 30/30 候选上完整可投影；原生共同观测的 598 个未来点中，
progress 最大绝对误差约 2.48e-5 m。这仅说明该 pilot 自车参考几何对齐良好，
不是原生整体 runtime 或其他场景的充分验收。

## 3. 动作条件信息存在，但部分变化表现为 mask 切换

普通预测器的邻车轨迹在五候选间完全相同；固定为零 jerk 邻车响应后摘要不变，
符合 B2 不接收候选计划的设计。它自身候选摘要的差异来自自车探针及关系变化。

B3 邻车预测在候选间有非零差异，六根的最大 XY 变化约 1.91–9.26 m；
但这不证明变化正确。保持自车探针、将邻车替换为零 jerk 响应后，有三根出现
27/31/21 个 flag 改变；部分较大的摘要变化来自有效性/池化变化，不能解释成
“捕捉到更强交互”或“DDPG 已利用预测”。该替换本身也可能 OOD。

## 4. 工程验证与限制

新增 16 项合成审计测试；最终选定离线回归 **239 passed、4 deselected，13.91 s**。
4 项仍是当前工作树缺少 st_cy 的原生导入检查，未放宽/关闭旧测试或依赖。
测试输出：`artifacts/tests/rcci02_frozen_reg_v2`。
首次新测试为 15 passed、1 failed：测试夹具没有真正覆盖“5 s 内退出”，
修正合成起点/速度使其确实退出后通过；真实配置、模型和阈值未改变。
首次目录 `rcci02_frozen_unit_v1` 保留。

只新增审计代码、配置、文档和测试；Raw、预测器、reward/action/termination、
replay/训练入口及 RCCI02-A v1 均未改动。基线 main 与当前功能工作树在运行后检查
保持干净。未合并、打 tag 或推送；大权重、原始数据和结果不提交 Git。

## 5. 判断与下一步

结论是：**冻结模型可以严格复用、道路后果可以计算，但 RCCI02-A 目前尚未证明
足够稳定地提供完整数值摘要，不宜直接进入正式 DDPG 或 native 接入。**

有界样本显示 B3 在部分共同支持指标上更准确，同时完整摘要覆盖更低；两件事
必须并列保留。当前没有理由扩大网络、重训 predictor、改奖励或改 seed。

下一步优先补充诊断无效预测的**倒退幅度/发生时刻、速度一致性及字段有效性**，
区分数值性非单调、模型响应形状与完整池规则的影响；继续用冻结离线数据，
不得为让 B3 通过而临时 clamp、删除 mandatory actor 或放宽阈值。
之后再冻结是否扩大同一 train/validation 离线审计、或另建接口版本的具体方案。
本轮没有启动 RCCI03/04，没有“通过 gate 后自动训练”的行为。
