# IC00：工作树隔离与可行性验收

日期：2026-10-10。结论：**开发隔离通过；课程训练尚未就绪。**

## 已完成

- 从干净 main 的 `75051acc06f6ea2ab6dda9be678f191b63d04569` 创建 `codex/interaction-curriculum-v1`。
- 工作树为 `E:/Prediction_RL_worktrees_course_learning`。创建前目标目录已存在且为空，没有覆盖已有文件。
- 原 `E:/Prediction_RL` 仍在 main，同一 HEAD，`git status --short` 为空。
- 原 `E:/Prediction_RL_worktrees/response-difference-v1` 未切换、修改或删除。
- 审阅了上游 reset/交通生成与启动、现有预测特征封装、训练分支坐标及几何 sidecar、原 replay 审计接口。
- 记录 11 个只读基线资产引用：两个三成员预测器、历史 request/aggregate、数据 review、六个 B2/B3 的 120k checkpoint。
- `ConvertFrom-Json` 验证资产清单可解析；逐一核验全部 11 个文件字节 SHA256 一致，引用路径均位于基线根目录内，角色无重复。
- `git diff --check` 通过；本次仅增加设计、任务文档、来源清单、验收记录与本工作树 AGENTS 边界，无 Python 源码或现有实验配置修改。

## 未完成且未声称完成

- 未运行新 SUMO 回合、训练、预测测试或课程单元测试。
- 未计算新的 D/I 分布、噪声下限或分层阈值；未验证 C1 数量。
- 未建立可执行 task 池、原目标分布 pi 或新的 reset 协议。
- 未实现课程 sampler、replay 课程元数据或训练 CLI。
- 未复制基线权重/轨迹，没有可写链接或覆盖旧结果，没有打开 predictor test 数据。
- 未启动 1.44M 请求步匹配训练或 960 回合评价。
- 未合并 main、推送或创建可运行课程里程碑 tag。

## 后续验收

按 `docs/tasks/ic01_training_side_task_audit.md` 先完成 train-only 核查，再确认任务入口协议。若需要改变连续 SUMO reset，必须在所有四组匹配对照中显式采用批准的新协议；不得把旧 B3 作为新协议下的严格 Mix 控制。
