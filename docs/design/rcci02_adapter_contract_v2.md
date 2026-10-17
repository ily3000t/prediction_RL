# RCCI02-A v2：共同有效前缀、字段支持与截尾

2026-10-17，在缓存重算前冻结的独立离线契约。只替换后处理定义，不改 v1、Raw、
预测器、loss、actor slots、probe plans、环境、奖励、DDPG 或运行协议。
本阶段不是 online release、正式实验预注册或训练授权。

## 1. 冻结输入与几何

输入白名单同 v1：当前可观测 history、原 actor adapter 的 12 个物理邻车 slots
（不含 ego）、已审计地图、原 ensemble mean 轨迹和原固定五探针 ego rollout。
相同世界轴/固定 root ego 原点；SUMO 前保险杠中心、当前长宽、root lane 唯一后继链。
NaN/shape/未知 lane/错误 root 几何显式报错，不变成正常缺失。
omitted tokens 不生成虚构位置，actor 容量和 mandatory roles 不变。

复用原 `RoadConsequenceMap`，不改其投影半径 0.5 m、1e-8 m 单调性容差、
1e-6 m progress 歧义定义或有限 merge-node 凸控制体积/道路切向 footprint。
不同 incoming approaches 使用跨来向描述；涉及 downstream 时仍只描述可连接
路线的纵向 gap，不冒充即时物理同车道净空。

完整预测仍有 25 个 0.2–5.0 s 点；不缩短模型时域，不 clip/平滑/重排位置或速度。
无模型重训、成员级聚合、Risk、Shield、commitment、新奖励或策略附近新探针。

## 2. 有效前缀，不恢复坏预测

每条轨迹先调用**未修改**的 v1 `road.track`（root + 25 点）。
从 root 起找到首个无效点，支持只保留到该点之前；后面即使几何恢复也不重新接入。
所以：

- 非单调位移、负速度、off-path/端点/歧义任一出现，都会终止这条有效前缀。
- 第一未来点即无效时，支持仅 root，horizon=0；不能算有效未来预测。
- root 数据自身不合法是工程错误，不用前缀 mask 掩盖。
- 记录首个失效时刻/原因、实际支持时域和是否全 5 s 完整。
- 截断完全由当前预测及地图产生，禁止使用原生未来终止/存在 masks。

following pool 的共同前缀 = ego 和**全部已选物理邻车**前缀的交集。
cross pool 的共同前缀 = ego 和**全部 eligible 跨来向物理邻车**前缀的交集。
不筛掉无效车辆，不拿剩余较长轨迹当完整池。共同支持内每个时刻都覆盖对应池全部车辆。
未知后缀不参与数值提取，也不被宣称安全；全时域完整 flag 仍为 false。

## 3. 数值与事件

### following

在共同前缀的 root + 未来离散点内，按 v1 的连接路线和方向规则分别提取 front/rear
最小 route gap、对应 closing speed、时刻。称 **supported-prefix minimum**，不是
全 5 s 安全最小值；不提供连续时间无漏检保证。至少存在一个未来支持点且对应
关系适用时，minimum_available 才为 true。只有 root 时不得把当前 gap 当预测成功。
保留数值平局 1e-6 m、最早时刻、车辆 ID 的稳定顺序。

### cross

共同前缀终点为 T≤5 s。用相同 progress 插值和 finite control volume 求每车
entry/exit 窗口，right-censored 表示在 **T** 时尚未进入或尚未离开，而不是
默认必须观察到完整退出。

- root 已占用：entry=0、left-censored=true；它是已观察状态，不是未来预测命中。
- T 内未进入：entry=None、reached=false、right-censored=true，不是假大 ETA。
- 进入但未离开：exit=None、right-censored=true；不填假出口。
- 仅在计算 **[0,T] 内支持的重叠时长** 时，以 T 截断未知出口；不得把 T 说成真实出口。
- 无正时长重叠：duration=0、first_overlap=None；这不是碰撞或安全判断。
- 有正时长重叠：输出 duration 与 first_overlap。≤1e-8 s 的边界接触仅按数值平局
  归零，不是安全阈值，所有轨迹不改动。
- 共同前缀无未来点时，overlap_available=false，root entry 状态仍可如实记录。
- 不再将完整 signed margin、完整 actor/ego exit 强塞入策略数值通道；完整窗口及
  无法计算的原因仍在诊断中保留，不声称这些旧字段已突然有效。

所有 eligible pairs 在同一 T 上比较。池化顺序冻结为：

1. 有支持内正 overlap：选时长最长，数值平局时最早 first_overlap，再车辆 ID。
2. 无正 overlap，至少一车未在支持内进入：优先于已完整观测的窗口分离；选 T 时
   离各自入口尚欠 progress 的最大值最小者，再车辆 ID。欠距不是 ETA 或安全裕度。
3. 其余已进入但无正 overlap：选 entry 时间差绝对值最小者，再车辆 ID。

时长/时刻数值平局 1e-8 s；欠距平局 1e-6 m。规则只作描述性压缩，不选择执行动作。
保留所有 pairs 的窗口、共同支持、选中 actor 和失效原因用于审计。

## 4. 独立 149 维布局

五候选各 28 维，按固定候选顺序排列；末尾原 9 维 root metadata 完全不变。
匹配维度只是容量匹配，旧 v1/Raw checkpoint 不能直接解释新语义。
缺失数值占位 0，必须结合本表 masks 与支持时域解释；不 clipping。

| 每候选索引 | 字段 | 单位/支持 |
| --- | --- | --- |
| 0–2 | front 支持内最小 gap、closing、发生时刻 | /100 m、/30 m/s、/5 s；由 4 控制 |
| 3–6 | front applicable、minimum_available、shared-prefix horizon、full_5s | 时域 /5 s，其余 bool |
| 7–9 | rear 同样三个数值 | 同上，由 11 控制 |
| 10–13 | rear applicable、minimum_available、shared-prefix horizon、full_5s | 同上 |
| 14–15 | ego/actor entry | /5 s；由 applicable 与 reached 解释；left-censor 标明 root 已占用 |
| 16–17 | 支持内 overlap duration、first_overlap | /5 s；duration 由 26 控制，first 还要求 duration>0 |
| 18–19 | cross applicable、shared-prefix horizon | bool、/5 s |
| 20–21 | ego/actor reached | 在支持内进入或 root 已占用 |
| 22–23 | ego/actor left_censored | root 已占用 |
| 24–25 | ego/actor right_censored | 相对支持终点 T；结合 reached 区分未进入/未离开 |
| 26–27 | overlap_available、full_pool_5s | 前者要求至少一个未来支持点，后者要求池全部 26 点有效 |

无 eligible cross pair 时 14–27 全零、applicable=false。
following 即使不适用，也保留共同前缀与全时域完整信息；minimum_available=false。
这是描述性部分信息，不是 missing actor 的风险上界或 fail-closed 控制策略。

## 5. 仅缓存重算的匹配验证

只读使用 `artifacts/rcci02_diag/rcci02_diag_v1/forecasts.json` 的原两种预测器均值，
通过不可变 hash、receipt、对应 root 历史、基线和源文件身份核验。
本轮不加载模型权重、不运行模型前向、SUMO 或训练；不打开 calibration/test targets。
缓存只是六根两个 episode，已用于定义诊断，不能当独立效果验证集。

须分别报告：

- v1 与 v2 都仍可计算的**完整 5 s**数量，不把 v2 的部分支持当完整覆盖增长；
- v2 支持内可用字段数、T 分布、root-only/部分/完整支持、截尾原因；
- 先前被全池数值屏蔽的候选是否保留了早期信息，同时保留未知尾部标记；
- 同源保存预测下的 v1 特征必须精确重现，B2/B3 使用相同新规则；
- 单/批、ID 顺序、未来 poisoning、刚体变换、无车、坏数据、root-only、短前缀、
  不恢复无效后缀和正反窗口等合成测试。

覆盖率变化是**定义不同**产生的信息可用性变化，不是预测更准确、方法更安全或
性能提升的证据。该重算不决定正式 seed/预算/checkpoint，不自动进入原生接入或训练。
之后如要扩大独立冻结 train/validation 审计或进入 RCCI03，另行明确范围。
