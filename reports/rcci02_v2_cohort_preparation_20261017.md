# RCCI v2 完整既有 cohort：实现与请求准备验收

2026-10-18 后续运行已完成，见 [完整结果复盘](rcci02_v2_full_analysis_20261018.md)。
本文保留 2026-10-17 准备时的状态，不应据此重复运行。

## 已做与未做

协议先提交 `66dd825`；实现/测试/独立配置/运行入口提交
`2a4563c17f005905c35c20f939a6abdc8cdc0a9b`。
本轮只做实现、合成测试、真实输入身份核对和 prepare；**没有完整推理结果**。
真实模型权重加载=0、模型前向=0、SUMO=0、训练=0。
端到端测试使用合成场景及 stub，不是实际模型/环境运行。

## 冻结请求

- 位置：`artifacts/rcci02_v2_cohort/rcci02_v2_full_v1/request.json`。
- hash：`68b9f264739f07fd77df81cb2315eea7f058b542d184cf4d5db798a45e47ec59`。
- 原 713 train + 177 validation 根、320 episodes，无删除/新增根或换 seed。
- 6,577 个基线绑定文件、21 个父产物；固定两个设计 episode，单独标记复用。
- CPU / torch threads=1，预定上限 1,780 ensemble / 5,340 member forward。
- prepare 不加载权重；单独禁止 torch.load 再调用 load_request，运行前守卫通过。
- 检查时 `started.json`、`report.json` 均不存在，尚待用户执行。

## 实现检查

新模块 `evaluation/rcci_v2_cohort.py`、工具 `tools/validate_rcci_v2_cohort.py`。
原 v1/v2 adapter、Raw、预测器、训练入口及权重、地图、reward/action/DDPG 未改。

- 特征与机制分解先完成，再读取 native branch；模型输入只有 observation tensors。
- native 全池和两模型共同支持；固定 1/3/5 s 分母，缺失不记为零误差。
- native ego 总体误差与 reference ego 邻车误差分开；事件缺失计数与时间误差同报。
- 保留所有已选物理车辆；未来终止/存在 mask 只用于离线真值，不能缩短策略特征。
- 全五候选/两模型固定支持后分解 ego、邻车与条件增量，不混入支持变化。
- 按 split/设计 episode 分组；提供 pooled cell 与等权 episode MAE。
- 每根预测/诊断独立保存，index/receipt 封存 SHA；不可覆盖、跳过失败根或自动重试。
- 没有将 `status=complete` 绑定于预测增益；负结果正常完成，不自动训练。

## 实际验证

- 最终选定离线回归：**309 passed，4 deselected，23.60 s**。
- 其中新模块 24 项测试；覆盖共同支持、未恢复无效后缀、缺失 actor、
  root-only、空物理池、事件漏报、ego/邻车误差分离、当前最小值、
  支持匹配分解、防未来泄漏、父审查继承及完整 cohort receipt、
  不完整/过期请求拒绝、合成端到端不可变封存。
- 测试路径：`artifacts/tests/rcci02_v2_cohort_reg_v2`。
- 四项仍因当前工作树缺少 st_cy 的原生扩展而明确排除；不是全项目原生测试通过。
- 实际 CLI `--help` 和真实 prepare/load_request 检查通过；没有执行 run。

曾在只读准备检查中发现新入口误以为中间 reversal request 直接携带训练审查，
已改为沿 SHA 绑定的原 frozen-audit request 继承清单，并增加回归测试。
未修改父报告、模型选择或旧 guard，也没有因此新选 checkpoint。

## 用户下一步

直接执行 [运行说明](../docs/runbooks/rcci02_v2_cohort_validation_v1.md) 顶部已准备命令。
完成后提交 `report.json` 分析。输出只回答既有数据上的接口支持/误差与机制问题，
不是 DDPG 收益、碰撞安全证明或独立泛化确认。
结果审查后作一次明确决策，不自动增加模型、训练预算或重复同六根调字段。

全部改动仅本地 RCCI codex 分支；不合并、不推送、不打 tag。
