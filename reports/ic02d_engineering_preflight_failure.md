# IC02d：采集实现与零仿真预检失败

## 当前结论

三任务独立 collector 已实现并提交；首次工程请求失败在 worker 的目录预检，尚未启动 SUMO 或进入采集核心。错误已修复并通过离线回归，**没有自动重跑**。不能据此声称恢复/分叉 runtime 已通过，也没有新的 D/I、层覆盖或课程收益结果。

- 初始实现/prepare/运行：干净的 `6937439f3d5e9e9c9e45ada0b34b44f0ed81da3d`。
- 预检修复：`7400c0c2e0277b35b29a9d7654449f7b9250ae10`。
- 分支：`codex/interaction-curriculum-v1`；基线 `E:/Prediction_RL` 仍是干净 main `75051acc06f6ea2ab6dda9be678f191b63d04569`，未改动。
- 用户明确批准的范围仍是三任务/114 会话/12,234 控制步/23,862 raw tick/90 分叉，0 predictor/DDPG 训练，未批准其余 29 个任务。

## 实际请求及错误

已 prepare 并尝试执行单一 run `artifacts/ic/ic02d_eng_v1/`，canonical request hash 为 `27d2343b9baf284b0764de2d83042049d9c3bcb8cef468ba881299aa9b64c30c`。顶层记录 UTC 2026-10-10 23:53:22.627476–23:53:25.652182（北京时间 2026-10-11 07:53）。失败报告、原请求和日志不可覆盖。

原 worker 通过 `any(output.glob('s*.json'))` 判断是否已有 session 产物。父进程按设计先写 `started.json`，它也匹配 `s*.json`，因此一个正常的首次 worker 被误认为重试并拒绝。这是调度预检实现错误，与场景难度、预测模型、参考策略效果或课程分层无关。

worker 目录只有 `started.json`、`launch.json`、`worker.log`；日志退出位置在 worker marker、Torch thread 设置、采集引擎导入/调用之前。没有 `worker_started.json`、settings、session、discovery、root 或 task seal。依据固定来源控制流和文件记录：

| 实际执行项 | 数值/状态 |
| --- | --- |
| 派发的 worker 进程 | 1，首任务 t11000 |
| 新 SUMO 会话 | 0 |
| 控制步 / raw tick | 0 / 0 |
| 候选分叉 | 0 |
| reference 模型加载/推理 | 0 / 0 |
| B2/B3 predictor 加载/推理/训练 | 0 / 0 / 0 |
| DDPG / learner | 未启动 / 未实例化 |
| 完整封存任务 | 0 |
| 父进程 RNG / 输入文件收尾 | unchanged=true / passed=true |
| worker tensor/session postflight | 不适用：worker 未进入任何模型/会话；未补造 |
| `engineering_gate` / `training_ready` | false / 未放行 |

父进程的 `postflight.json` 在写失败报告前正常保存；task attempt budget inventory 为空，与采集未开始一致。没有把空 inventory 当作成功任务。

| 原产物 | 字节 SHA256 |
| --- | --- |
| `request.json` | `d54408e723a453db651427641b0072a7887ad724dcbdebffdbb2f3ac7fe9ca38` |
| `failure.json` | `e5ec35fce5608e17d70e7f5e5392c2f8efda141b31fb747a05309ea57c563c2a` |
| `postflight.json` | `1f3db5ee5a03d8230e37739f5b9300d01e321a7c53b5c4d993168bce722055d9` |
| `t11000/worker.log` | `4a640c7c17968a0181d5e5afc5b75ecf335feb329f70165492876091f7a9ac8f` |

## 修复与测试

预检改为明确的父进程 setup 文件 allowlist：`started.json`、`launch.json`、`worker.log`，并校验 binding。任何 worker marker、session、settings、discovery、seal、失败或未知产物仍拒绝再次执行。不改任务 seed、环境、奖励、候选、模型、预算、分层尺度或 immutable report 规则。

399 项离线测试通过：原 388 项，加 11 项实际预检入口回归。测试直接复现旧 wildcard 匹配 `started.json`，验证正常 setup 只读检查能通过，已有任一执行产物/错误绑定会被拒绝；mock 实际 worker 入口仅进入采集核心一次，再次进入拒绝，**没有调用真实 SUMO 或模型**。CLI help、语法及 staged diff check 已通过。

## 下一步与限制

原 failed 请求不可重跑，`--resume` 不能把失败或不完整 task 修成完成，不能移动/删除 receipt 或换名绕过协议。虽然实际仿真预算消耗为零，仍需用户确认以修复源码建立新的命名工程请求版本，继续同三个 seed 和相同上限；当前未创建或执行该新请求。

runtime 恢复/分叉、三任务 D/I 和覆盖仍待真实工程验收；任务 classifier、pi、持久 learner/replay 与课程训练均未放行。旧 IC02b 收尾限制与所有旧结果保留。不推送、不合并 main、不打课程 milestone tag。
