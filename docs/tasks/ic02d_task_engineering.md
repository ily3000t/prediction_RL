# IC02d：三任务参考刻画工程验证

## 目标、前置与授权

在 IC02c 的离线设计后实现独立检查点 collector，验证 reset/early_ramp/approach、多次恢复及真实邻车响应的工程可信性。用户在明确的三任务/114 会话/12,234 控制步/90 分叉问题后回复“好的继续”；授权独立保存在 `provenance/ic02d_engineering_authorization.json`，旧 IC02c 不可执行配置不改。

原基线 main `75051acc…` 只读且干净，IC02c 原报告 hash 一致；只用训练 seed 11000/11008/11016，原 config、作者 greedy reference 及固定五候选。旧 20 个开发场景和 predictor test 不打开。

## 允许与禁止范围

允许独立 observer、collector、有限存证、离线 fixture 和本工作树三任务工程运行。最多 9 root 槽位、90 个分叉、114 新 SUMO 会话、12,234 控制步、23,862 raw tick。其余 29-task pilot 不批准；不训练或调用 B2/B3 predictor，不实例化 learner，不改环境/reward/动作/历史/replay/终止，不晋升 task 层或计算 pi。

## 实现、验收与产物

- `curriculum/task_characterization.py`：冻结 scope、同车道多检查点扫描、预执行预算检查、旧尺度描述与实际计数验证。
- `curriculum/task_collector.py`：只读钩子及包裹层，保留共享 ReplayBrancher 的 capture/restore、候选 sidecar 和 label contract。新鲜初始化复用相同序列，计数不进入 root signature、observation、reward 或控制。
- `tools/collect_curriculum_tasks.py`：单一命名 run、prepare/hash确认、三个独立串行 worker、不可覆盖 seal、纯离线重算验收与完整-only reuse。
- 配置 `configs/development/ic02d_task_engineering_v1.json`；原始 ignored 产物 `artifacts/ic/ic02d_eng_v1/`。

每次 session 都记录原 StepObserver 的真实事件、预热/插入/控制/cleanup tick；预算在外部动作前预留，失败尝试不退费。每次发现和捕获重复精确一致，五候选按正序/逆序恢复，branch 与 sidecar 精确一致。terminal 前后不补齐未来，工程 cap 不合成 done。

每个 worker 在成功或失败时先记录 reference tensor、session RNG 恢复与资产文件 postflight，再封存或报告失败；父进程同样先保存 RNG/文件收尾，避免原 IC02b 的收尾缺口。旧 predictor 只检查文件指纹，不加载，不能声称新的 tensor 检查补造了旧缺失证据。

offline 根证据用原 train-only D/关键车辆/I 支持定义和旧训练数据冻结的尺度；不重新拟合分位数，不标 C0/C1/C2。工程 gate 与层覆盖/方法效果分开。实际结果和测试数量另写 `reports/`。

## Git、暂停与交接

测试、审查并原子提交后才准备、执行有限请求。单一 run ID 不允许更名重置预算；started 无 complete 不自动重跑；失败保存并暂停，禁止换种子或降低要求。负响应或截尾仍可完成工程存证，不是应删除的场景。

运行命令见 `docs/runbooks/ic02d_task_engineering.md`。本阶段通过后先分析三个任务工程证据，再由用户单独批准其余 29-task 覆盖，不自动进入课程训练、合并 main、推送或创建 milestone tag。

## 实际状态

初次请求 `ic02d_eng_v1` 在 worker 预检误把 `started.json` 当 session 残留，启动 SUMO 前失败，0 会话/控制步/分叉。allowlist 修复及 399 项离线测试已完成；原 failure/请求不变，未重跑。下一步需要用户确认修复源码下同三任务、同上限的新命名请求，不能把本阶段描述为 runtime 已通过。见 `reports/ic02d_engineering_preflight_failure.md`。

用户随后回复“继续”，批准同三任务同预算的独立修复版 `ic02d_eng_v2`。新增 v2 配置与授权，不改旧失败记录；启动时绑定并检查旧零执行证据。新版命令与限制见 `docs/runbooks/ic02d_task_engineering_v2.md`，真实执行结果另记验收报告，尚不能预称 runtime 通过。
