# IC02：合法任务入口与覆盖可行性

## 阶段目标

建立可审查且与真实训练入口一致的 task 定义；明确原连续 reset 的限制、新协议差异与有限验证。入口一致性不等于三层任务覆盖或课程有效。

## 前置条件

- 已阅读 AGENTS.md、`docs/design/interaction_curriculum_v1.md`、IC01 验收与 `docs/design/curriculum_task_entry_v1.md`。
- 基线仍是只读、干净的 main `75051acc06f6ea2ab6dda9be678f191b63d04569`。
- IC01 报告及 root 字节指纹一致；C0 为空的原结果保留。

## 允许与禁止范围

IC02a 允许独立纯离线源码检查、source-episode 映射、设计配置、测试及文档，仅写课程工作树。禁止改 reset/奖励/动作/历史、读 test、执行 SUMO 或训练。

IC02b 只有用户明确批准命名协议与 finite smoke 后才允许实现独立 worker-entry 原型与所需本地构建。不得改旧基线或静默改既有训练脚本；不创建课程训练请求，不扩大 smoke，不采集新的完整反事实数据集。

## 实现任务

1. IC02a：读取而不导入原环境源码；验证 seed、预热、插入、历史及 TimeFeature 假设，记录 actual 配置而非单纯默认值。
2. 映射 source episode，保留 root/task 区别，不自动赋 task 层或 pi。
3. 提出四组共用的新入口协议、RNG/进程、控制责任及预算；说明与旧连续训练的差异。
4. 提交后纯离线运行核查；保存实际报告。该报告不宣称 runtime parity。
5. 用户批准 IC02b 后再实现新入口、synthetic 终止/传输测试及最多 12 episode/144 控制步 smoke。
6. 入口通过后再提出任务级覆盖/分支采集方案；需要新数据时另获批准，C0 缺口不通过事后改阈值解决。

## 测试与验收

- 严格设计不能通过设置 authorized=true 变成可运行请求；未知字段、错误版本、不同 arm 和 root-label 自动晋升均拒绝。
- 原 reset/插入/history/seed 假设变化可检测；静态分析不导入 simulator。
- 多 root 只归属于同一 source episode，不计独立 task，不变更 IC01 层。
- IC02b：同 seed/动作首回合与原 direct 对齐；持续 learner/replay 不随 task 重初始化；新入口只在健康终止边界使用。
- 所有失败正常保留，runtime gate 与方法表现分开。入口 smoke 通过也不能绕过任务覆盖与用户训练审批。

## 产物

- 离线源码：`src/prediction_rl/curriculum/entry_design.py`。
- 离线 CLI：`tools/audit_curriculum_entry.py`。
- 设计配置：`configs/development/ic02a_entry_design_v1.json`，未批准、未实现。
- 输出：本工作树 `artifacts/ic/<short-run-id>`，不覆盖。
- 设计与运行说明：`docs/design/curriculum_task_entry_v1.md`、`docs/runbooks/ic02a_entry_design.md`。
- 小型验收：`reports/`；原始运行日志、轨迹、权重、编译扩展不提交 Git。
- IC02b 原型：`src/prediction_rl/envs/fresh_task.py`；冻结预算/比较：`curriculum/entry_smoke.py`；CLI：`tools/smoke_curriculum_entry.py`。
- IC02b 批准范围：`provenance/ic02b_smoke_authorization.json`；新配置与 runbook 分别是 `configs/development/ic02b_entry_smoke_v1.json`、`docs/runbooks/ic02b_entry_smoke.md`。实际 12 回合/144 控制步已执行；原聚合失败保留，独立离线复核确认 12 项封存轨迹精确一致，不能补造原最终内存 postflight。
- IC02b 复核 CLI：`tools/review_curriculum_entry.py`；验收：`reports/ic02b_entry_parity_acceptance.md`；原运行 `artifacts/ic/ic02b_v1/` 与独立复核 `artifacts/ic/ic02b_review_v1/` 都不可覆盖。
- IC02c 已完成离线任务设计：`tools/audit_curriculum_task_design.py`、`docs/tasks/ic02c_task_characterization.md` 与 `reports/ic02c_task_design_acceptance.md`。实际报告 `artifacts/ic/ic02c_design_v1/report.json` 只保存既存证据与未授权提案；0 新会话/分支/训练。

## Git、暂停条件与交接

在当前 codex/ 分支原子提交，不合并 main、不推送、不打 runtime milestone tag。新入口的封存轨迹 parity 已通过，但完整原 postflight 未恢复；任务层退化与 target pi 未定义，仍不启动课程训练。具体已实现入口和离线复核 CLI 见 IC02b runbook；原 12 回合预算已消耗，不得重跑，新增 live 数据必须另获批准。

IC02c 的下一批准点是三任务独立参考刻画 collector：11000/11008/11016，最多 114 个新 SUMO 会话、12,234 控制步、23,862 raw tick，0 predictor/DDPG 训练。完整 32-task 覆盖 pilot 在工程验收后另获批准；不能由离线设计的 gate 推断运行授权。
