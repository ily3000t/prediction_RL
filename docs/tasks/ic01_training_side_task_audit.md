# IC01：训练侧交互分层可行性核查

## 阶段目标

复用已有 train 分支，区分真实关键邻车响应与纯自车几何差异，判断是否能形成非退化 C0/C1/C2，并明确离线 root 如何关联到可执行任务。不是让条件预测器在某个新指标上胜出。

## 前置条件

- 从 `75051acc06f6ea2ab6dda9be678f191b63d04569` 建立的课程工作树。
- `provenance/interaction_curriculum_baseline_v1.json` 中冻结资产存在且 SHA256 一致。
- 基线目录及资产只读；test 数据仍封存。
- 阅读 `docs/design/interaction_curriculum_v1.md` 和工作树 AGENTS.md。

## 允许修改范围

在课程工作树中新增独立离线分析模块、strict-schema 设计/分析配置、单元测试及文档。所有输出使用本工作树 `artifacts/ic/<short-run-id>`，保存解析配置、来源 commit、脏状态、输入文件指纹与命令。

## 禁止修改范围

原主工作区、旧 checkpoint/结果/配置；预测器结构与权重；DDPG、奖励、终止/bootstrap、控制/观测特征和连续 SUMO reset。不得读取 test，或用开发评价回报/方法胜负定义层。不得启动完整训练或在线反事实分叉。

## 实现任务

1. 显式只读解析基线 train split，验证分支、重复及公共坐标来源；不通过修改历史报告或放宽哈希来复用数据。
2. 定义只依据 root 当前状态的关键邻车规则及道路纵坐标；保存 ID、角色、有效性与分支间对齐。
3. 对共同有效候选对/邻车/时刻计算 I，并对同动作重复分支计算同指标噪声。未观察未来、actor 缺失、终止截尾、无共同区间分别记录。
4. 分别报告 D 变量分布、I 分布、正/零/不可计算计数、根/episode 数及道路位置；不将相邻 root 或重复分支当独立样本。
5. 从 train-only 分布生成**待审核**边界及正归一化尺度；冻结规则优先级，报告 C0/C1/C2 互斥性和完整覆盖。结果不足则暂停，不自动改阈值。
6. 保存 root_record；提出 task_record 映射与任务入口需要的参数。沿中段 root 直接恢复不能伪装成原 episode reset。

## 测试与验收

- 仅改变 ego 轨迹、邻车轨迹保持相同，I 必须为零。
- 真正邻车变化产生正 I；候选顺序置换、ID 顺序置换不改变统计。
- 两分支使用同一固定坐标原点；移动的 ego-relative 误用必须被拒绝或检出。
- actor 出现/消失、无共同有效区间、不同终止时刻不填零、不错误计入均值。
- 同动作重复噪声与不同动作响应分别统计，无支持的样本不可进入 C1。
- road/edge 切换不造成伪纵向跳变；缺少映射时显式不可计算。
- 分层只使用 train，完整合法 task 保留；根到 episode 的数量和哈希可复核。
- 原代码/配置/预测器权重未修改；整个分析不启动 SUMO 或 DDPG。

这些是有效性 gate，不要求某层有预期收益，更不要求 B3 胜于 B2。

## 产物位置

- 源码：`src/prediction_rl/curriculum/interaction.py` 与 `train_reader.py`；入口 `tools/audit_curriculum_interaction.py`，未接入训练路径。
- 运行配置：`configs/development/ic01_train_interaction_v1.json`，无多层继承，不静默忽略未知字段。
- 原始离线分析及摘要：本工作树 `artifacts/ic/<short-run-id>`，Git 忽略。
- 小型验收说明：`reports/ic01_training_interaction_acceptance.md`，只保存工程结论与汇总，不提交轨迹或权重。

## Git 提交要求

独立 `feat(curriculum): audit training-side interaction strata` 与适当测试/文档原子提交，提交前审查 staged diff。留在 `codex/interaction-curriculum-v1`，不合并 main、不推送、不打已可运行课程的标签。

## 暂停条件

训练分支缺少可验证重复/几何；关键邻车响应主要低于噪声；C1 太少或依赖坐标伪差；所有可用高 I root 都无法转换成符合设计的任务入口；需要新的数据采集或改变环境协议。

## 交接命令

本阶段已运行完成，报告在 `artifacts/ic/ic01_v1/report.json`；该 run ID 不可覆盖。实际分析命令与测试步骤见 `docs/runbooks/ic01_interaction_audit.md`。

当前工程审计通过，三层任务池未通过：C0=0、C1=211、C2=502，所有 root 均不可直接作为原 reset 任务。暂停课程训练，保留负结果，不改阈值补层。下一步需要确认 IC02 的任务入口及覆盖方案；不要运行旧 120k 训练入口来代替课程实验。
