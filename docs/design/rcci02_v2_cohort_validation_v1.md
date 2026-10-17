# RCCI v2：既有完整 cohort 离线验证契约

2026-10-17，在完整推理与查看新结果前冻结。本阶段只验证固定接口的信息与误差，
不是正式方法效果试验，也不是 DDPG、SUMO、模型重训或 online 接入授权。

## 范围和成本

- 仅 RCCI01 已核验的 713 train + 177 validation 根，原顺序、320 episodes。
- 所有根保留，包括四个无监督根；不读取 calibration/test，不生成新场景或 seeds。
- 设计用六根所属两个 episode 单列 `design_episode_reuse`，其余单列；
  validation 不是未使用过的测试集，不能称作独立确认或泛化试验。
- 固定复用原 B2/B3 三成员 ensemble、actor adapter、五探针及 ego rollout。
- CPU、torch threads=1，每根每模型一次 ensemble forward：上限 1,780 次；
  等价成员前向上限 5,340 次。无 batch/AMP/模型参数改动。
- 完整推理由用户运行；开发只运行合成单元测试和请求准备的只读身份检查。
- 保存逐根预测与诊断，逐文件 SHA256 和不可变 receipt；不可覆盖已启动请求。
  无 resume 或失败根跳过；失败正常保留，不改范围/seed/阈值重新做成通过。

## 不变项与防泄漏

不改 RCCI v1/v2 契约、几何/阈值、预测器或 Raw。特征只接收 root history、
固定预测和 ego rollout；原生 future 仅进入单独离线比较函数。
先构建特征，再读取分支原生标签；任何 future 存在/终止 mask 不决定策略特征。
选定模型 SHA 和成员选择清单复用已完成审查，不重新选 checkpoint。
请求绑定当前源码、环境、完整配置、地图、父报告、所有输入文件及本设计文档。

## 误差与支持，事前固定

1. 每模型分别报告 following/cross 完整 5 s、共同前缀时长、首个失效原因、
   root-only、适用池和 omitted 数量；v1/v2 完整支持逐候选仍必须相等。
2. native 可观察前缀到终止/ego 或任一池车辆消失前截止，不填充未来尾部。
   未来 lane 仅构造离线真值，不用于投影预测。
3. following 的比较池为全部已选物理邻车，cross 为全部 root eligible 跨来向邻车。
   每候选共同比较长度为 B2/B3 和 native 全池共同前缀的最小值。
   不删除短轨迹车辆；没有未来支持时只记分母与失败原因，不贡献零误差。
4. 共同前缀内比较 front/rear 最小 gap；比较全池描述而非只挑某方法选中的 actor。
   gap 误差分两种：native ego+native neighbors（总体后果误差），
   同一个 reference ego+native neighbors（隔离 ego 参考轨迹偏差）。
   同时记录 ego progress 误差；隔离比较仍是诊断，不是真实执行因果效应。
5. 固定 1/3/5 s 锚点额外比较：仅两模型和 native 全池均支持该时域才计算，
   每个锚点保留候选总分母、可观察分母和共同预测支持数量。
   不以更短前缀上的低 MAE 冒充固定 5 s 的改善。
6. cross 按每个 root eligible pair 比较 entry/exit、支持内 overlap/first-overlap。
   每对使用同一个 cross 全池共同前缀，不以各自较长窗口混比。
   reached/missed、exit 和 overlap 混淆计数与时间 MAE 同报；
   root 已占用不算 entry 预测命中。没有正 overlap 时 first 不填假时间。
   truth 分别用 native ego 和 reference ego，窗口仍只是有限控制体积占用，非碰撞真值。
7. 当前状态最小值、actor 全前缀占用、first-overlap 等于 ego entry、
   overlap 等于支持时长分别计数，不宣称它们是学到邻车响应的证据。

## 机制分解，不用它选择动作

原 B3 预测不变；另做只读诊断，保持全部车辆，在跨 B2/B3、全部五探针的
共同预测支持 T 上重算数值：

- 每候选真实预测 + 每候选 ego（总候选差异）；
- 零 jerk 候选邻车预测 + 每候选 ego（ego 计划差异）；
- 每候选邻车预测 + 零 jerk ego（邻车差异）；
- 零 jerk 邻车预测 + 零 jerk ego（共同参考）。

following/cross 的共同 T 各自固定，不能把质量/截尾改变当成响应数值变化。
分别报告 gap、entry、overlap、first-overlap 的有效性变化及共同可比较数值差异；
没有同时有效的数值不做占位 0 的差分。另报原生各候选支持与未统一前缀的差别。
这类输入替换可能离分布，既非因果识别，也不保证 DDPG 会利用这些通道。

## 汇总和决策

逐根保留，按 split × 是否设计 episode 汇总；误差同时给 pooled cell 描述与
每 episode MAE 的等权汇总。独立交通单位是 episode，不是根/候选/车辆/时间点。
不进行新显著性检验、不自动挑优模型、字段或阈值。

完整报告即使显示 RCCI 没有区分度、B3 较差或支持不足也正常完成。
只有结果审查后，才决定保留/降级接口字段、另建版本或是否批准有界闭环；
本阶段不自动训练，不扩展模型，不进入 calibration/test。
