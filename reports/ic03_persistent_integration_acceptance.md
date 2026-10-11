# IC03：持久 learner 离线接入验收

## 结论

完成 fresh-task 单回合 worker 接口与持久原 ALL DDPG learner 的离线接入核心，新增 22 项测试通过，全项目 **1,240 passed、2 个原作者加载器 torch.load FutureWarning，49.38 秒**。本阶段没有真实 SUMO 会话、冻结预测器推理、正式课程训练或新实验效果结果。

- 代码提交：`e9a6bb7453052f85519c08bcc28dcb56bb3c8448`。
- 分支：`codex/interaction-curriculum-v1`；不推送、不合并、不打 tag。
- 基线仍为只读干净 main `75051acc06f6ea2ab6dda9be678f191b63d04569`。
- 实现：`envs/curriculum_stream.py`、`training/curriculum_replay.py`；设计见 `docs/design/curriculum_persistent_integration_v1.md`。

## 实际测试证据

四条 B2/B3 × Mix/Curriculum 合成路径均调用原 `SingleEnvExperiment.train`，没有改其 episode loop、DDPG `_train`、TimeFeature 或均匀 replay 算法。每条 12 个三步合成回合产生 36 条实际 transition、48 次原 act 调用；终止 sample 保留，reset 不入 replay。合成优化使用已有显式 smoke preset，仅为廉价等价性测试，不是实际课程参数。

加审计与未加审计的独立参考路径在以下内容逐位一致：原动作与回放 tuples、State/mask、policy/critic 权重、target 权重、optimizer 和 scheduler 状态，以及 Python/NumPy/PyTorch RNG。B2/B3 使用不同模式的合成 provider，不调用真实 ensemble；不能将此测试解释成预测效果评估。

另验证了：终止 store 前不能 reset、先关闭旧 worker 再创建新 worker、每回合历史从一帧开始、标签不进入 State.info、ring overwrite 的实际 slot 观察、真实原抽样索引重复计数、回合边界 overrun=2、optimizer/replay 同实例不清空、worker 的 reset/step/space/entry/close 异常不变成正常 done、对齐或元数据错误不插入伪 transition。

初次新增测试为 7 passed/10 failed：新 guard 未接受原 ALL reset State.info 中的 None。只修新观察器以兼容原表示，不改原 State 或训练器；复测 17 passed。补充五项故障测试后，新旧规则定向回归 60 passed；曾有一次测试文件名拼错，0 项运行，随后使用实际文件名。最终全项目回归包含全部 22 项新增测试，为上述 1,240 passed。

## 回归命令及资产处理

定向命令（在课程工作树、pytorch 环境）：

```powershell
python -B -m pytest -q tests/test_curriculum_persistent_learner.py tests/test_curriculum_task_rule_proposal.py -p no:cacheprovider --tb=short --basetemp artifacts/ut_ic03_core_v4
```

全回归仅在测试进程中显式只读引用基线作者权重，先使用原 assets() 验证文件集合和哈希；没有复制权重、建立链接、改 repository loader 或安装包：

```powershell
python -B -c "import sys; from pathlib import Path; sys.path[:0]=['tools','src']; import compare_external_baselines as author; author.MODEL=Path(r'E:\Prediction_RL\RL-MPC-LaneMerging-master\pretrained_models\ddpg_default1_extended'); print('explicit_read_only_author_assets=',str(author.MODEL)); author.assets(); import pytest; raise SystemExit(pytest.main(['-q','tests','-p','no:cacheprovider','--tb=short','--basetemp',r'E:\Prediction_RL_worktrees_course_learning\artifacts\ut_ic03_all_v1']))"
```

重跑测试应选新的、事先确认不存在的 basetemp；pytest 会删除已有目标。上述测试输出是本次实际结果，不代表 live worker/课程 CLI 可运行。

2026-10-11 北京时间约 18:12 的只读复核确认：provenance 中十一项基线资产字节哈希全部一致；IC02h request/report/task_proposals/postflight 四项字节哈希一致；原有 src/tools/configs 相对 IC02h 实现提交没有修改。此次只新增两项科学模块和一个测试文件，不改旧失败报告、规则或配置。`git diff --cached --check` 已通过；自身合成测试临时目录清理不涉及实验产物。

## 后续边界

IC03 仅通过**离线接入验收**，formal_training_authorized=false。未提供课程训练命令，不把旧 B3 用作新 Mix，不启动 120k，不采新场景或调整阈值/seed。

下一项先形成来源校验、完整配置、独立授权的 IC03b 真实 smoke 请求，拟四条路径各两个原生回合（总最多 8 SUMO 会话、4,000 控制步、零优化更新）。精确预算与限制见接入设计。用户确认并完成真实工程验收后，再冻结 IC04 的四组匹配训练请求；不能从本报告自动获得运行授权。
