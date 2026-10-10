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
- IC02b 批准范围：`provenance/ic02b_smoke_authorization.json`；新配置与 runbook 分别是 `configs/development/ic02b_entry_smoke_v1.json`、`docs/runbooks/ic02b_entry_smoke.md`。实际 runtime 结果尚待运行验收。

## Git、暂停条件与交接

在当前 codex/ 分支原子提交，不合并 main、不推送、不打 runtime milestone tag。新入口和有限 smoke 已获用户批准；缺本地构建、入口不对齐、任务层退化或 target pi 未定义时，仍不启动课程训练。具体已实现入口 CLI 见 IC02b runbook；实现或批准不等于 runtime 已通过。
