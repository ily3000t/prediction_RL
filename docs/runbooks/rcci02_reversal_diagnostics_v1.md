# RCCI02 倒退与字段支持诊断 v1

工作目录 `E:/Prediction_RL_RCCI`；使用 pytorch Python。
配置 `configs/development/rcci02_reversal_diagnostics_v1.json`，无 extends。
只诊断既有六根 pilot，不扩大场景、不更换 seeds/预测器、不改接口或阈值。

## 请求与运行

前置：`rcci02_bnd_v1` 已完整封存，报告和逐根输出哈希与固定配置一致；
旧源码、模型/成员选择、所有所用 source data 与基线仍然冻结。
本工具创建**新请求**，不会放宽旧 loader，也不执行/覆盖已完成的旧请求。
先逐文件核验旧请求记录的全部 source 内容不变（只允许新增诊断文件），再核验
当前完整 source 集合和新配置。新请求严格绑定实际 source/environment/Git。

```powershell
python -B tools/diagnose_rcci_reversals.py prepare --run-id rcci02_diag_v1
```

复制实际打印的 hash：

```powershell
python -B tools/diagnose_rcci_reversals.py run --request artifacts/rcci02_diag/rcci02_diag_v1/request.json --confirm-request-hash HASH_FROM_PREPARE
```

run ID 必须不存在；失败/完成目录均保留，不重试或覆盖。
CPU 单线程，六根 × 两个 ensemble = 12 次 ensemble 前向（各内部三成员）；
与旧结果逐根/逐字段/聚合**精确重现**后才封存诊断。
若重现不一致，显式失败，不容忍性放宽或改用新结果替换旧结论。
不启动 SUMO、训练或校准，不打开 calibration/test target 文件。

## 诊断量及含义

1. **预测** progress 真正为负的离散增量数、幅度、时间、累计后退和最大回撤。
   与 v1 首次负增量后失效的后缀点数分开；前者不是现实车辆倒车次数。
2. 直接路网投影的有效性，与 v1 的速度/单调性检查及后缀连带屏蔽分开；
   后来仍可投影不代表后缀已经可以安全恢复，工具不恢复任何 mask。
3. 负增量发生时两端 speed、world XY 位移、`ds − speed_end × 0.2 s`；
   原模型 XY/speed 分别输出，该量是冻结运动学一致性描述，不是新 loss。
4. 预测 float32 XY 的最大坐标 ULP 尺度及后退量比值。它是存储精度参照，
   不是严格投影误差上界；不据此调节 EPS 或进行 isotonic/clamp 修复。
5. 同一 actor/candidate 实际原生前缀中的 progress 增量。超出原生支持则为 null，
   不将模型 5 s 的后退硬说成原生数据已证伪的运动。
6. mandatory roles 与坏轨迹 ID，following/cross 全池失效的传播关系。
   不为了提高覆盖率删掉关键前后车或其他难算 actor。
7. 12 个连续字段各自的支持数；数值为 0 不自动代表无信息。
   entry=0 且 left_censored 是已观察占用，不是新预测命中。
8. 所有 eligible pairs 与被池化选中 pair 的 margin 无效原因：投影失败、
   未到达、左截尾、右截尾等（非互斥计数）。区分“根本无完整窗口”与“池化屏蔽”。
   保留节点控制体积长度；不把这一区域等同物理碰撞区域。

## 产物

`artifacts/rcci02_diag/<run_id>/`：request、preparation、started、report、receipt，另有：

- `forecasts.json`：实际冻结 ensemble mean（float32）和原 ego rollout（float64），
  固定 root-ego 原点坐标，供后续只读重算；不包含权重，不提交 Git。
- `diagnostics.json`：逐 root/method/candidate/actor 的投影、速度、负增量事件、
  原生支持及摘要字段/截尾理由；不用长期运行日志代替审计证据。

六根仍只对应两个 episode，不计算方法显著性或把 candidate/actor 当独立场景。
新源码及结果 hash、71 个所用源文件、父请求/报告/receipt 必须再次核验。

## 边界与下一步

只解释 RCCI02-A v1 的限制；不改 Raw、DDPG、预测网络/损失、reward、动作、
region、半径或单调性 EPS。不实现新接口、CV 对照、成员级聚合或任何训练。
有界诊断后给出明确保留/修订建议，不自动重复预算或增加网络。若应修订，
先冻结独立接口版本与匹配验证方案；旧负结果永久保留。
