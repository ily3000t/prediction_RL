# RCCI01：创新定位评议与离线可计算性审计契约

日期：2026-10-17。父来源为 75051acc，研究方案文档为 dca9136。
本阶段不修改原设计文档，不引入差分/课程研究分支的实现或结果。

## 客观判断

方向合理，但属于有条件的接口机制创新，不能现在认定为强算法创新。
它固定预测器、DDPG、奖励、动作和交通训练方式，将研究变量缩小到
预测后果的表达，成本和解释性优于同时更换网络、数据及控制问题。

最有价值的具体问题是：在同一路径/跨路径关系下，能否把五个相同
干预探针对应的响应表示成方向一致、带时域截尾和有效性语义的后果，
使连续策略更好利用预测，而不变成计划选择器或安全接管器。

创新风险需要如实保留：

- [ROSA-RL](https://arxiv.org/html/2606.16558v1) 已将冲突区占用预测用于
  SUMO 连续速度建议的 RL。不能以占用/TTC/冲突区增强 RL 本身宣称首次。
- [MATS](https://proceedings.mlr.press/v155/ivanovic21a.html) 已研究面向
  下游规划控制的预测表示。接口问题合理，但不是空白领域。
- [TIP](https://arxiv.org/abs/2110.08750) 讨论任务效用与预测误差的区别。
  新接口不能仅凭更小几何误差或更丰富字段就宣称改善决策。
- [Interventional Behavior Prediction](https://arxiv.org/abs/2204.08665)
  是条件与干预预测区别的先例。这里只讨论限定 SUMO 和参考后续控制。

2×2 对照可估计 Relation 整体增量及 predictor×interface 的交互：
(B3-Relation − B3-Raw) − (B2-Relation − B2-Raw)。但是 Raw→Relation 同时
涉及地图先验、25 点时间信息和摘要规则，不能由四组单独识别这三项的
闭环贡献。不自动增加 CV 或其他训练组；若未进一步控制，只报告整体
接口效应。差分还须在预先定义的具体指标尺度上解释，不泛称所有交互。

当前 B3-Raw 的已有 20 场景到达率已饱和，新接口未必能提升到达率；
效率、舒适性和信息覆盖也可能退化。道路投影可能放大预测偏差，显式
缺失通道也可能成为策略捷径。负结果应保留，不按已有输赢筛选阈值。

## 本阶段实现范围

新增独立纯离线工具 audit_rcci_feasibility.py，阶段仅为 RCCI01。
人工配置独立可读、无 extends，严格拒绝未知/改变的配置。
旧 provider、预测器、损失、环境、DDPG、Raw 权重和历史 guards 不改。
源码从本工作树导入；基线只提供被核验的文件，不导入基线运行模块。

数据只从已冻结 review 中按原顺序取 train/validation 根族。
bounded 模式各取前三个根，包括合法空监督根，不按效果/可用性筛选；
full 模式为全部 713/177 根。所有五个分支均检查；calibration/test 的
轨迹文件不打开，也不读取 200–219 闭环输赢来设计字段。

只读基线须为干净的 75051acc。核验 review 的固定 SHA、collection
request、所选 episode receipt 的 review 记录、各所用 history/labels/
c0–c4 文件哈希、root/split/episode/time/ID 对齐及原始轨迹与标签一致性。
这是所用文件的完整验证，不是重新执行全部旧 collection guard，也不是
扫描不相关 split 的整目录 inventory。receipt 中未使用文件不读取。
两个 ensemble 只核验固定 SHA；不加载权重、不执行推理，不假称 runtime
兼容验收。新产物只写本工作树 artifacts/rcci01/，短 ID、不可覆盖。
基线和本工作树的 XML 分别严格绑定 75051acc 的 Git blob，仅允许 LF/CRLF
检出差异，不允许 XML 重序列化或语义/空白变更；请求另外记录当前工作树
地图字节 SHA。此检查不改写旧请求哈希或降低任何历史 guard。

## 几何与缺失语义

SUMO getPosition 返回前保险杠中心，lanePosition 也是前保险杠沿道路的位置，
不是车身中心。依据：[SUMO TraCI 文档](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)。
本审计使用 XML polyline，并显式按 SUMO lane length 与 polyline arc length
的比例转换，不假定两者精确相等。

测量三个分布：lanePosition 重建 XY 残差、已知 lane 投影残差、投影所得
lanePosition 与原生值之差。已知 lane 来自离线标签，不允许在线选择预测
lane 时读取未来真值。结果不是在线预测可投影率，更不是安全认证。
投影需要显式 lane 白名单，报告端点 clamp/歧义；不将大误差投影判为正常。
距离 bins 只做描述性直方图，不设置可用性或安全阈值。

检查记录中的航向、route ID、长宽、缺失 actor、后续新增 actor、短历史、
容量遗漏、时域截尾和道路组关系。terminal 几何不进入轨迹统计。
四采样时刻之外的 lane 变化只叫道路转换，不能叫漏掉的碰撞/冲突事件。

现有样例没有 angle/heading。没有完整原生航向就不能把构造车身框称为
原生 footprint 真值。RCCI02 可讨论道路切向或轨迹差分近似，但必须另外
定义静止/转弯/歧义 mask 和假设；不因此自动补采数据或扩大模型。

本阶段不定义最终冲突区域、进入/离开窗口、时间裕度、排序、149 维布局，
不评估 forecast 投影/事件准确性，也不宣称已完成 RCCI02。

## 测试与运行边界

单元测试包含五 lane 往返、端点/离路、白名单、非法数值、共同刚体变换、
原始 future/base frame 对齐、terminal 截尾、split 交叉拒绝和流式统计。
共同刚体变换一致性仅针对地图后处理，不对冻结预测器提出旋转等变要求。

真实 bounded 审计最多六根、三十个已存分支，不运行 SUMO/预测推理/训练。
必要时 full 离线元数据审计可沿相同冻结规则执行，不新增仿真回合。
每根保存小型统计；全局量以数值数组流式累计，不复制大型原始轨迹。
完成后审阅可计算性和支持集，再冻结 RCCI02；不直接启动策略重训。

所有提交仅留在 codex/road-conflict-consequence-interface-v1，不合并 main、
不推送、不打可运行里程碑 tag。发布上游资产的许可仍未解决。
