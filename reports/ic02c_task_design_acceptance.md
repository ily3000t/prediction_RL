# IC02c：任务参考刻画离线设计验收

## 结论

离线设计审计完成，38 项输入指纹一致，32 个训练侧任务提案已保存；没有新 SUMO 会话、候选分支、模型推理、learner transition 或训练。**通过的是设计与既存证据的绑定，不是任务覆盖、三层分层或课程方法效果。**

实现来源为干净的 `59d0353b810324b7944a7d9fe4a92e10a0636e77`、`codex/interaction-curriculum-v1`。原基线 `E:/Prediction_RL` 前后保持干净 main/HEAD `75051acc06f6ea2ab6dda9be678f191b63d04569`，模型只核对文件指纹，未加载。原 IC01/IC02a/IC02b 结果和 IC02b 缺失 postflight 均未修改。

实际已运行的离线命令：

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/audit_curriculum_task_design.py --baseline-root E:\Prediction_RL --run-id ic02c_design_v1
```

实际审计区间为 UTC 2026-10-10 21:24:21.506638–21:24:22.228641，即北京时间 2026-10-11 05:24。报告保留解析配置、实际命令、来源 commit、依赖版本与指纹；没有用运行耗时推算未来采集时间。

| 产物 | 字节 SHA256 |
| --- | --- |
| `artifacts/ic/ic02c_design_v1/report.json` | `6b54b5b96b4fe975132d197ae4f71f357ea63fea1515397005d3f28cc353d487` |
| `artifacts/ic/ic02c_design_v1/task_proposals.json` | `9d46ea05ac5ba84bcbe5285c55e11930893a2bbef8e902a4b71c0d397d3c93a7` |

解析配置 canonical digest 为 `6e6052947d25ce8ca45fb2185d58402440fde9717eb5eb4575978a9ebe45276c`。原始产物留在 Git 忽略目录，不提交模型、轨迹或原始数据。

## 现有证据及缺口

| 旧检查点 | root 数 | 参考前缀控制步范围 |
| --- | ---: | --- |
| approach | 256 | 32–82 |
| internal_mid | 244 | 47–105 |
| downstream | 213 | 55–137 |

合计 713 root/256 source episode，reset root 为零。原 root 层 C0/C1/C2=0/211/502 原样保留，不能代表可执行 task 数或 pi。

从 IC02b 封存 trace 只提取三个 original-direct 首 reset 当前帧。三者剩余合流距离均为 214.10 m，观测邻车分别 12/12/11，关键车辆 3/3/2，历史均为 1 帧，无关键槽位遗漏。I 均保持 null，task 层均保持 null；没有未来候选支持，不能称它们为 C0。沿公共路线投影的负 gap/零 TTC 是道路代理，不证明起步已发生物理碰撞，也不直接作为安全判定。

固定步长提案从完整 11000–11255 train roster 选出 32 个 seed（11000、11008、…、11248）。旧 root 的 C1/C2 数量或策略结果不影响选择。三任务工程子集 11000/11008/11016 包含在该 pilot 内；这是覆盖提案，不是代表性总体分布估计。

## 非授权采集预算

| 上限 | 三任务工程提案 | 32-task pilot 总提案 |
| --- | ---: | ---: |
| reference root 槽位 | 9 | 96 |
| 候选分叉 | 90 | 960 |
| 新 SUMO 会话 | 114 | 1,216 |
| 总控制步 | 12,234 | 130,496 |
| 总 raw tick | 23,862 | 254,528 |

预算含两次参考发现、每检查点两次 root 捕获、五候选各两次恢复、重放前缀、25 步未来、每会话 101 个等待/插入 tick 及至多一个 cleanup tick，不只统计候选未来。32-task 总提案包含三任务工程子集，不批准自动执行余下任务。

同车道 reset/early_ramp/approach 需要独立检查点 observer，旧 collector 不能直接使用该配置。本阶段没有 runtime collector 或课程训练入口。`engineering_gate=true` 仅为离线审计，`runtime_authorized=false`、`training_ready=false`、classifier/pi=null 均保留。

## 测试与下一批准点

360 项离线测试通过（原 328 项，加 32 项设计、预算、roster、reset-only/mask 与未来隔离测试）；三个新增 Python 文件语法编译检查、CLI help 和 staged diff check 通过。测试写入新建的忽略临时 leaf，无仿真或训练，未清理既有目录。

下一步须用户明确批准：实现独立 collector、离线 fixture/回归测试，并在上述三任务上执行有限工程验收；不训练 predictor/DDPG。工程结果核查后，才另请批准剩余 29-task 覆盖 pilot、提出并冻结 task-level D/I/classifier/pi。不得以本离线报告为运行授权，也不能把新 postflight 追溯补到 IC02b 原失败报告。
