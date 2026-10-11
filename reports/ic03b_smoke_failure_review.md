# IC03b：首组原生回合完成、收据收尾失败与修复记录

## 结论与状态

2026-10-11 已执行用户批准的八回合上限工程 smoke，但协调器在首组 `b2_mix` 完成两个原生回合后停止。**总工程验收未通过，剩余三组未启动，正式课程训练未启动。** 原 `failure.json`、错误 `complete.json`、日志与轨迹保持原样，不能将文件名 complete 或子进程成功退出解释为整个请求通过。

- 执行来源：干净 `codex/interaction-curriculum-v1`，`c64006a55a9b2a388d3b4204a338796950bebd8e`。
- 原请求 canonical hash：`86a5830a2f4a365eebfbd260cbb284d7febf331875a3d60b4712a4d5dc7f53d2`。
- 修复提交：`5cd69c1edf2e14b0f5d3782879a99fe59e24b73a`。
- 请求与失败证据：`artifacts/ic/ic03b_smoke_v1/`；没有覆盖或重封存该目录。
- 基线 `E:/Prediction_RL` 仍是干净 main `75051acc06f6ea2ab6dda9be678f191b63d04569`。

## 实际消耗与已取得的有限证据

| 路径 / 原生回合 | task / 层 | 控制步 | raw ticks | 预测特征调用 | 终止 |
| --- | --- | ---: | ---: | ---: | --- |
| B2-Mix / 0 | t11024 / H | 75 | 176 | 75 | upstream_collision |
| B2-Mix / 1 | t11200 / M | 60 | 161 | 60 | upstream_collision |
| 合计 | 两个独立 fresh-task 会话 | 135 | 337 | 135 | 两个正常原生终止 |

这两个回合是从头初始化的 learner 在原 warmup 内执行，零 optimizer 更新、零真实 minibatch；137 次原 agent.act 包含两次 reset act。135 条实际 transition 留在同一 replay 中，H/M 分别为 75/60，reset bridge=0。子进程 summary 记录同一 learner/optimizer/replay 持续存续、终止 store 后才 reset、逐包 reward 独立重算最大绝对误差为 0、原执行与历史/特征对齐、environment/features RNG neutral、policy/critic/targets 哈希不变、冻结 predictor 不变。

上述是**首组子进程结束前通过的工程检查**，并非四组整体验收，也不能事后重建没有运行的三组。碰撞作为原生结果正常保留，不是文件收据失败原因，不评价未训练策略的性能。

父 failure 中 jobs=[] 是因为 verify 位于 append 前；不能据此报告零回合消耗。剩余尚未执行的明确范围为 B2-Curriculum、B3-Mix、B3-Curriculum，各两回合：最多六个新 SUMO 会话、3,000 控制步、3,612 raw ticks、3,006 特征调用。这个数字只是尚未执行的范围，**不是自动重试或新授权**。不重跑 B2-Mix，不将已执行的两个回合从预算中抹去。

## 失败原因的字节级证据

原子进程先调用 seal_episode，把父进程仍持有的 worker.log 纳入不可变 inventory；随后又打印 `smoke_arm_complete=b2_mix steps=135`。父进程等待退出并关闭日志后 verify，自然发现内容变化。只读比对发现十六个封存文件中只有 worker.log 不一致，没有缺失或新增文件；日志变化严格等于最后的 37 字节完成提示（含 CRLF）。

| worker.log 版本 | SHA256 |
| --- | --- |
| 原收据中的封存字节 | d87f486b4f1df692234cf27821cbce6785ee08551ca4d560bdd8d9e68ba96215 |
| 运行后的完整 1,717 字节日志 | 4709b240041cd68c9a294d5fda588b9061f80d160255a6225fe4fab183340bd7 |
| 去除末尾 37 字节后的前缀 | d87f486b4f1df692234cf27821cbce6785ee08551ca4d560bdd8d9e68ba96215 |

其它保留证据的**实际文件字节哈希**（不是 canonical request hash）：

| 文件 | SHA256 |
| --- | --- |
| request.json | 4e804e1a6432dab1edb51d0c3f5fa43114cb2308b35c749e61ec7e4c8b3f1415 |
| failure.json | 1957f17265459e13d37dee9370c1422cd00bf7e0b4e97f5006c5923973064bae |
| b2_mix/complete.json | 9b4888db571300a0c01923dc4273adb0aee230ba1327c4e425cbcb25d0300766 |
| b2_mix/summary.json | c4bf787f7e55fe3c93e62dc7f8e57f56ce7af73a73f740734d19e10455d7dca4 |

## 修复与实际测试

只修改新 smoke 编排器及其测试，没有修改科学源码、课程阈值/比例/seed、原 reward/action/termination、DDPG preset、预测器或正式配置。子进程不再封存整个目录；父进程必须先得到成功退出，完成子进程 finally cleanup，并关闭 stdout 日志句柄，才读取 summary、封存全部文件、严格 verify。worker.log 仍进入收据；没有排除可变文件来放宽完整性检查，也没有给旧失败报告补写通过。

新增两个离线回归覆盖：封存必须晚于模拟 child cleanup 和 stdout close，后续修改日志仍被拒绝；child 失败即使留下 summary，也不得产生收据或启动下一组，第二次 execute 必须拒绝。没有为测试新启动 SUMO 或真实 predictor 推理。

- 定向：58 passed，9.30 秒。
- 全项目离线回归：1,276 passed，48.44 秒；两个原作者加载器 torch.load FutureWarning，不是新失败。
- git diff --check 与 staged diff --check 通过。
- 十一项固定基线资产及 IC02h 四项固定文件字节哈希均不变。
- 本次五个具名 pytest 临时目录已清理；所有真实实验/失败产物保留。

实际定向命令：

```powershell
python -B -m pytest -q tests/test_curriculum_integration_smoke.py tests/test_curriculum_persistent_learner.py -p no:cacheprovider --tb=short --basetemp artifacts/ut_ic03b_receipt_v1
```

实际全回归仅在测试进程中明确只读引用并验证原作者权重：

```powershell
python -B -c "import sys; from pathlib import Path; sys.path[:0]=['tools','src']; import compare_external_baselines as author; author.MODEL=Path(r'E:\Prediction_RL\RL-MPC-LaneMerging-master\pretrained_models\ddpg_default1_extended'); author.assets(); import pytest; raise SystemExit(pytest.main(['-q','tests','-p','no:cacheprovider','--tb=short','--basetemp',r'E:\Prediction_RL_worktrees_course_learning\artifacts\ut_ic03b_receipt_all_v1']))"
```

上述 basetemp 已删除；后续测试应使用核查后不存在的新路径。测试通过不等于修复后的真实 smoke 已执行。

## 交接限制

旧 run/prepare 命令不可再次运行：旧请求绑定原 SHA/源码，失败 run 不可覆盖，不允许通过删除 failure/complete、换路径、手动重封存或 monkey-patch 解除限制。

下一步先取得用户对**剩余三组最多六个回合、零优化更新**的明确确认，再建立绑定新来源与旧消耗的独立修复版请求；保留旧失败，以独立审计说明首组证据的有限性，不把它变成新版本完整通过收据。该请求尚未实现/准备，因此当前不提供虚构命令。不自动增加两个重复回合，不进入 IC04 正式训练冻结，直到需要的真实工程证据补齐。

本轮仅本地原子提交，不 push、merge 或打 milestone tag。
