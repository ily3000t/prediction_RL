# RCCI02-A v1：离线道路后果接口契约

状态：独立纯函数原型；不是 online release，不接入 DDPG，不改变 Raw 默认路径。
研究边界沿用原始 RCCI 方案；本文件细化第一版字段与几何假设，不改写原方案。
机器可读描述由 `features/road_consequences.py::contract()` 返回。

## 1. 输入、资产与禁止项

- 当前观测历史、已审计五 lane 地图、原 actor adapter 生成的 12 个物理邻车槽位
  （不含 ego），以及 root 时刻车辆的长宽。ego 独立存在。
- 预测输入形状 `[candidate,12,25,4]`：固定 root ego 原点下的 XY、speed、acceleration。
  使用原三成员的**平均轨迹**；不改变预测器，不引入成员级后果聚合。
- 自车输入 `[candidate,25,4]`：原五探针 rollout。正式五探针约束仍由旧
  `rollout_ego` 检查：首步 jerk 为 -5/-2.5/0/2.5/5，之后 24 步零 jerk。
  adapter 接受单候选是为了核查批量一致性，不表示原 rollout 已支持任意动作。
- 不读取真实 future、未来 lane/existence/terminal mask、reward、branch labels。
  候选排序和描述性池化不选择执行动作、不吸附 DDPG 输出、不修改 observation 前缀。
- 重建 root actor slots，不能把 omitted summary 当可定位车辆。root 外新出现车辆不在本接口
  精确几何支持中；不能声称全车辆冲突检查或安全保证。
- 输入可以是 NumPy 或 detached CPU tensor；所有元素（包括 padding）必须有限。
  NaN/shape、未知 root lane、不一致观测几何等是显式工程错误，不当作低风险。

## 2. 坐标、合法路径与投影

SUMO 车辆位置按前保险杠中心处理；不是车身中心。预测 XY 加 root ego 世界原点。
root lane_position 与地图重建 XY 必须相符（1e-5 m 数值检查）。

合法投影只允许当前观测 lane 与唯一后继链：

- `ramp_0 → :mergenode_1_0 → highwayahead_0`；
- `highwayrear_0 → :mergenode_0_0 → highwayahead_0`；
- 已处于 downstream 时只允许 downstream，不能倒推原来来自哪条道路。

lane 长度按 SUMO 数值标定 polyline 弧长。projected progress 的零点是 downstream
起点，沿合法路径递增。最近点来自合法链，不遍历地图上的其他路线。

以下有限预测显式无效，而不是强制吸附或静默删车：

1. 距合法中心线残差 > 0.5 m；
2. 最近点需要 lane-chain 边界外端点 clamp；
3. 等距最近点（1e-8 m 数值容差）对应 progress 相差 > 1e-6 m；
4. progress 后退超过 1e-8 m（从该点起不可计算完整窗口）；
5. 预测 speed < 0。

0.5 m 是第一版**表示支持半径**，不是安全净距、车道宽度、测量噪声模型或按方法
胜负调出的阈值。目前未通过冻结预测器的全数据覆盖核查；不能预设此半径具有足够可用率。
异常、离路和有限时域截尾分开记录。任何正式修改必须产生新接口版本，保留旧结果。

## 3. 有限合流控制体积，不是物理碰撞区

采用固定地图中 `mergenode` polygon 的**凸包**作为有限 merge-node control volume。
凸包可能比原多边形略保守，不把共享 downstream 的整段道路当冲突区，不使用
合流终点 ETA 差冒充占用时窗。该区域覆盖整个合流节点，早段横向分离的车也可能
同时占用；因此同时占用只表示潜在时序冲突，**不表示身体接触或碰撞**。

每辆车的近似 footprint 是当前 length/width 的道路切向矩形：从前保险杠中心向后
length，横向左右各 width/2。每个中心线直线段上，使用 separating-axis clipping
求矩形与有限凸区相交时的前保险杠 progress 区间，再合并。
没有使用默认 lane width；width 来自观测车辆。

这是 road-aligned footprint approximation。原数据没有原生 heading/route ID，
不能将它称为 SUMO 原生多边形真值，不能宣称连续轨迹碰撞证明。沿曲线的切向随
polyline 段变化；当前几何要求占用 progress 支持连通，不连通时显式拒绝。

## 4. 全时域与截尾

所有量使用 t=0 root + 25 个 0.2–5.0 s 时刻。
同路径最小 route gap 取这 26 个离散时刻，**不是连续时间最小净距**。
占用进入/离开采用相邻点间 projected progress 线性插值；不是对真实 SUMO
连续运动进行无漏检认证。

每个 actor 窗口包括：valid、reached、entry/exit、left/right_censored、passed_before_root。

- root 已在区域内：entry=0，left_censored=true；真实进入可能早于 root。
- 5 s 内未进入：entry/exit=None，reached=false，right_censored=true；不等于安全。
- 进入但仍未离开：exit=None、right_censored=true。
- 恰好停在出口边界：仍算接触，不假称已经离开。
- root 已完全通过：passed_before_root=true，不是假造一个 0 s ETA。
- 任一点投影无效：窗口 valid=false；不把有效前缀说成完整 5 s 窗口，
  也不把投影失败标成时间截尾。

观察窗口内的重叠 duration 可计算时保留；右截尾出口仅在这一项中使用观察终点
5 s，绝不将 5 s 填成“真实出口”。精确 signed margin 要求两车进入未左截尾且
出口均观测到：`max(entry_ego, entry_actor) − min(exit_ego, exit_actor)`。
负数为观察到的重叠，正数为完整窗口间分离；没有校准概率、TTC 或安全阈值含义。

## 5. 149 维布局与掩码

每候选 28 维，五候选按固定次序连续排列，占 0–139；140–148 保留 Raw 的
9 个当前角色/历史覆盖/omission metadata，定义与旧实现完全一致。
两种预测器获得相同的地图、规则、尺寸和 mask 待遇。

| 每候选局部索引 | 字段 | 尺度/语义 |
| --- | --- | --- |
| 0–2 / 6–8 | front/rear 最小 route gap、对应 closing speed、时刻 | /100 m、/30 m/s、/5 s；无 clipping |
| 3 / 9 | front/rear applicable | 至少一个有效 actor-time 同路径且在对应方向 |
| 4 / 10 | complete_pool_projection | ego 与**全部选中物理车辆**26 点均可投影 |
| 5 / 11 | relation_sample_fraction | 有效对应关系 actor-time 数 / (26×选中车辆数)；空池为 0 |
| 12–15 | ego entry/exit、actor entry/exit | /5 s；缺失占位 0，由后面的 reached/censor flags 解释 |
| 16–17 | observed overlap、signed margin | /5 s，分别由 overlap_valid / margin_valid 控制 |
| 18–19 | cross applicable、complete_pool_projection | 存在跨来向车辆；全部 eligible pairs 完整可投影 |
| 20–21 | ego reached、actor reached | 窗口内进入或 root 已占用 |
| 22–23 | ego left_censored、actor left_censored | entry=0 不代表真实 0 s 进入 |
| 24–25 | ego right_censored、actor right_censored | 未进入或未离开；结合 reached 区分 |
| 26–27 | overlap_valid、margin_valid | 不可计算不是零风险 |

following 是**沿可连接路线的纵向描述**：沿用已审计的 same-approach，或涉及
downstream 的连接路线关系；不同 incoming approaches 不互算同路径 gap。
有 downstream 连接关系不表示此刻已经处于同一物理车道，弯道 route gap 也不是
欧氏 footprint clearance。front gap = progress差 − 前车length；rear gap =
反向progress差 − ego length，负值表示纵向投影重叠。

front/rear 角色在预测 actor-time 中动态池化，不等于固定 root target_front/rear。
按最小 route gap、数值平局 1e-6 m 内最早时刻、车辆 ID 选择；delta 在 1e-6 m
内按 0 归 front。任何选中车辆投影不完整时，front/rear 的三个数值归占位 0，
complete=false；不能仅对剩下“好算”的车辆给出貌似完整结果。

cross pool 只包含 root 处相对 ego 为另一 incoming approach 的物理车辆；
已经 downstream 的车辆不被虚构来向，而归 connected-route following。
对每个 eligible actor 独立提取控制体积窗口，然后按以下冻结优先级选择一个：
投影无效 → 已观测正 overlap → 窗口截尾/无法定 margin → 完整有限 separation。
overlap 类选观察重叠最长，其余有限 margin 类选最小 margin，最终按车辆 ID；
窗口池化数值平局为 1e-8 s。完整诊断保留全部 eligible pairs 与选中 ID。

存在任一无效 eligible pair 时，所有 cross 连续量及窗口 flags 归占位 0，
保留 applicable=1、complete=0；不将某一个有效 pair 冒充完整池支持。
无 eligible pair 时 16 项全 0，applicable=0。
这是描述性压缩，不是对所有邻车的安全认证，也不是自动执行优先级。

同样 149 维只匹配输入容量，**不代表旧 Raw checkpoint 可直接解释新语义**。
未来 Relation 策略必须独立训练；Raw 模型复用仍需 RCCI04 审核。

## 6. 测试、已知限制与下一道验收

测试包含单/批候选一致、共同刚体变换、稳定 ID、数值平局、负 gap、closing、
有限控制体积和前保险杠长度/宽度、窗口插值/截尾、早于四采样点之间的短事件、
无 actor、omission、历史不足、工程错误和投影失效、跨候选独立及 future poisoning。
刚体一致性仅针对 adapter，不是冻结预测网络的旋转等变性声明。

本阶段未执行 predictor 前向、SUMO 或 DDPG；未打开 calibration/test。
尚不能说明 0.5 m 投影支持对真实冻结预测足够、哪个预测器更好、占用量与真实
事件的误差、或新接口闭环是否有效。后续先在冻结 train/validation 上另建不可变
离线请求，测覆盖/截尾/偏差、B2/B3 差异及候选分解，再决定是否进入 RCCI03。
不因覆盖低就放宽半径、不因到达率饱和就自动增加场景或网络。

旧 Raw provider、预测器、reward/action/termination/replay、训练与评价协议均不改动；
未构建 st_cy、未修全局依赖、未启动正式训练、未合并 main 或推送。
