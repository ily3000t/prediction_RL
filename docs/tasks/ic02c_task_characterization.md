# IC02c：任务参考刻画离线设计

## 阶段目标与前置条件

将现有 train-root 与真实入口证据对照，提出有限参考刻画预算，不自动修复 C0、赋任务层或放行课程训练。IC01、IC02a 和 IC02b 封存复核已存在；原基线 main `75051acc06f6ea2ab6dda9be678f191b63d04569` 只读且干净。IC02b 原完整 postflight 缺失仍保留。

## 允许与禁止范围

允许课程工作树内的纯离线模块、配置、测试、文档及新的忽略审计产物。禁止新 SUMO 会话、候选分支、模型加载/推理、训练、打开 test、写旧基线或覆盖原产物。

## 实现任务

1. 验证冻结来源、旧 root/报告/映射和已封存入口 trace 的字节绑定。
2. 记录旧 root 参考前缀分布及真实 reset 的当前可观测 D/actor/history，不用未来替入口分类。
3. 从完整训练 roster 固定步长提出 32 task，明确仅三 task 工程阶段的有限预算。
4. 计入所有发现、捕获、恢复前缀、未来、预热/插入与 cleanup 成本。
5. 保存未授权、未分类提案；记录同车道检查点不能由旧 collector 直接执行。

## 测试、产物与 Git

新测试拒绝版本/授权/参数漂移、缺失/晋升 roster、错误预算和非 reset 状态；验证选择不依赖旧层结果、不补历史、不使用 trace 未来。运行既有入口/actor/data/feature/terminal 离线回归。

实现：`src/prediction_rl/curriculum/task_design.py`、`tools/audit_curriculum_task_design.py`。配置：`configs/development/ic02c_task_design_v1.json`。设计见 `docs/design/curriculum_task_characterization_v1.md`；离线命令见 `docs/runbooks/ic02c_task_design.md`。ignored 输出为 `artifacts/ic/<run-id>/report.json` 与 `task_proposals.json`，小型验收写入 `reports/`。

在当前 codex/ 分支审查、测试、原子提交后，干净树运行离线审计，再单独提交实际验收。禁止 `git add .`、merge、push、tag。

## 暂停条件与交接

hash/来源漂移即停止，不放宽标准。离线审计通过仅表示设计与已存证据绑定有效，不表示任务池覆盖、I/分层、runtime 或方法收益通过。下一批准点是三任务 collector 的实现与有限工程运行；其 114 会话/12,234 控制步/23,862 raw tick 上限需明确确认，不能由本阶段配置授权。
