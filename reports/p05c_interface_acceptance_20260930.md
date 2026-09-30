# P5c 实现验收与手动诊断交接

## 本次完成范围

已实现任务相关性诊断，不修改既有预测器、actor 选择、原奖励、控制、
种子、数据、校准或 P5a/P5b 报告。没有新增事件头、策略特征或 DDPG。
旧项目只读，未改动；没有推送或新增可运行里程碑 tag。

配置：`configs/development/p05_task_relevance_v1.json`。
实现：`src/prediction_rl/prediction/task_relevance.py`。
入口：`tools/diagnose_task_relevance.py`。
定义：`docs/tasks/p05_task_relevance.md`。
运行手册：`docs/runbooks/p05_task_relevance.md`。

输出内容：

1. 原始分支奖励、终止类型与碰撞/到达事件，保留事件未知和轨迹删失。
2. Slotted Jerk 的终止、时间、jerk、非法动作奖励分解与一致性检查。
3. CV、普通、条件预测及真实距离代理与原回报/事件偏好的两两一致性。
4. 全部实际观察到车辆中最近前保险杠和最小同路径间距的 actor、时刻、
   车道及 selected / omitted-root / new-actor 身份。它不是碰撞责任判断。
5. 共同支持上的真实关键车辆位置误差、同场景候选响应差异误差，及排序
   错误候选对。所有未来信息仅用于离线归因，不进入模型 observation。
6. root/episode 等权统计、有效分母、并列与删失计数，不伪增独立样本数。

未折现分支回报来自已保存原奖励，不能称为 DDPG 的 Q 值。
仅在双方都终止或双方都有完整 5 秒时进行回报次序比较；其他长度不一的
未完成前缀不强行比较。保守事件 mask 沿用原标签，不把另一事件终止后
未知的目标事件静默改成负例。完整诊断可能发现可比较事件不足，这应
如实报告，而不是改变标签让一致性更好。

## 实际测试与审计

- 18 项新增测试覆盖奖励、删失、终止事件、车辆角色、未来隔离、受限
  split、确认/不可覆盖保护、prepare 不推理及负结果正常结束。
- 全套测试：**393 passed in 9.04 s**。
- staged diff 检查通过；仅提交源代码、配置、测试、文档。
- 实现 commit：`82f3794f3b29d1578b1c52c0bfcfe2d7b087e568`。
- 开发验收：`artifacts/p5/p5c_task_v1_01/report.json`。
- 报告 SHA256：`e961c919f210039bf2ff2cded7bd8885bab6b65fff0f70de0de09da834e0d043`。
- 干净实现版本执行，`status=complete`、`interface_gate=true`。
- 9 个既有开发 root / 3 个 episode / 45 个候选分支；无新 SUMO 运行。
- 原奖励重构最大绝对误差 **0**，无法核验的奖励分解时刻为 **0**。
- 17 个分支到达预测时域但未终止，20 个到达终止，8 个碰撞终止。
- 45 个分支的非终止最近前保险杠车辆均在选中集合；仅为这 3 个开发
  episode 的观察，不证明训练集/线上不会遗漏关键车辆。

工程通过不代表任务关联成立：开发集真实距离代理与回报的 root 等权
次序一致性约 46.0%；到达事件没有双方已知且结果不同的可比较候选对，
碰撞方向比较仅涉及 1 个 episode。不得将 9 个相关快照作为充分证据。
这些小样本结果未用于修改 score、阈值、模型或选择正式训练 seed。

## 已准备、未执行的训练集诊断

准备阶段核验 256 个 episode 收据、原始分支/重放、历史、标签及几何，
冻结 **713 个已到达 root**。准备过程未调用模型推理。推理阶段最多读取
16 个 root 家族，避免把所有原始分支长期载入内存。

请求：`artifacts/task_diagnostics/predictor_v1_task1/request.json`。
文件 SHA256：`aa2605abe59e40eea5f47d0c0f7c640d8742700b9e02876c95714fdc462a51c1`。
规范化请求哈希：`75710184bc7b12d0cc2745ba4d14429bc55f3f9fcc6d2fda5ae9e8140e64ee5f`。

用户在 `pytorch` 环境手动执行：

```powershell
cd E:\Prediction_RL
python tools/diagnose_task_relevance.py run --request artifacts/task_diagnostics/predictor_v1_task1/request.json --confirm-request-hash 75710184bc7b12d0cc2745ba4d14429bc55f3f9fcc6d2fda5ae9e8140e64ee5f
```

预计输出为同目录的 `report.json` 与 `roots.json`；本次未生成它们。
这是训练集内诊断，不是重新训练，更不是独立泛化评价。
检查任务关联、关键 actor 覆盖、误差定位和有效事件分母后，再讨论
最小可部署连续特征契约及小规模闭环探索。`ddpg_ready=false`、
`feature_contract_frozen=false` 保留，独立 test 仍封闭。
