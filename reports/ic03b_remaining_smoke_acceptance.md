# IC03b 剩余三组真实集成 smoke 验收

## 结论

2026-10-11 用户明确批准的剩余三组、最多六个新回合已完成，**新请求工程 gate=true**。六个原生回合共 1,147 控制步、1,753 raw ticks、1,147 预测特征调用、零 optimizer 更新。未重复旧 B2-Mix，未启动正式课程训练。

四条路径的真实接入证据现已齐备：新三组有严格有效完整收据，旧 B2-Mix 只通过独立只读取证复核。**旧请求仍 failed、旧收据仍无效**，其字节未修改；不能把 coverage_review 理解为替旧失败报告改判通过。

- 实现与执行来源：干净 `codex/interaction-curriculum-v1`，`83440d967f64269a18a42d503aa5e8fd20ec6b09`。
- 新请求 canonical hash：`3f5f910adbda5ed1242e19d7630fe85d03a3d2e1d507f242f076bb95e9d98ec4`。
- 实际执行时间：19:11:18–19:13:39，Asia/Shanghai。
- 独立配置：`configs/development/ic03b_remaining_smoke_v1.json`；授权：`provenance/ic03b_remaining_smoke_authorization.json`。
- 新结果：`artifacts/ic/ic03b_remaining_v1/report.json` 与 `coverage_review.json`。
- 基线仍为只读干净 main `75051acc06f6ea2ab6dda9be678f191b63d04569`。

## 实际工程结果与累计预算

| 路径 | 来源与验收方式 | task seeds / 层 | 控制步 | raw ticks | 原生终止 |
| --- | --- | --- | ---: | ---: | --- |
| B2-Mix | 旧运行；独立只读取证，非修复收据 | 11024/H、11200/M | 75+60=135 | 337 | 碰撞、碰撞 |
| B2-Curriculum | 新运行；完整收据有效 | 11016/M、11152/L | 51+455=506 | 708 | 到达、到达 |
| B3-Mix | 新运行；完整收据有效 | 11024/H、11200/M | 75+60=135 | 337 | 碰撞、碰撞 |
| B3-Curriculum | 新运行；完整收据有效 | 11016/M、11152/L | 51+455=506 | 708 | 到达、到达 |
| 旧加新合计 | 八个实际会话，不是八个新回合 | 四条路径各两回合 | 1,282 | 2,090 | 全部保留 |

累计特征调用 1,282、optimizer 更新 0。新六回合和旧加新总量均低于各自冻结上限；没有重抽任务、缩短 warmup、截断未终止回合、扩预算或重跑 B2-Mix。

上表**不是课程或 B3 有效性的实验结果**。learner 从头初始化且没有优化，Mix/Curriculum 此处采到不同任务，不能由到达/碰撞差异宣称课程提升。两次原生到达中的 455 步也不作为修改课程、分层或 seed 的依据。所有方法效果仍待正式匹配训练与固定节点开发评价。

## 独立验收证据

执行结束、尚未修改或提交文档时，以冻结 request hash 调用新 loader 重验 clean HEAD、全部来源/依赖/预测器、21-file 旧失败目录 inventory 与其原科学输入。旧日志变化严格等于已记录的末尾 37 字节提示，其余原收据文件一致；原失效收据不被 verify 接受，不重封存旧目录，不事后重建旧 tensor 或 RNG。

对新三组分别 verify 整个目录，metadata 与 summary/report 一致；再独立读取压缩 native/replay 日志，而非只相信顶层 gate：

- 逐包重新核查原 reset/20 秒预热/ego 插入、raw tick、当前 frame、history、连续 jerk 原执行与 Slotted Jerk reward；六回合 reward 最大绝对误差均为 0。
- 每组恰有两次原生 terminal，135/506/506 条真实 transition 与实际控制步一致；每组 actual_environment_step 严格从 1 连续递增。
- Replay 大小分别为 506/135/506，reset bridge=0；原 agent.act 分别为 508/137/508，包括每回合 reset act。
- 同一 learner、optimizer、target、replay 跨各自两个 worker 存续；标签不进入 observation、TimeFeature 或 State.info；sampler、原控制与 agent 调用时钟分开。
- 三组均记录零真实 minibatch、零 optimizer state/更新、policy/critic/target 初末 tensor 哈希一致；冻结 B2/B3 三成员预测器无梯度、权重不变。
- Native transport/冻结推理保持 learner RNG 不变；原探索正常使用其 RNG，父协调器 RNG 不变；不是要求整个 learner RNG 恒定。
- 成功 child 退出与 finally cleanup 完成、父 worker.log 关闭之后才封存，日志包含在有效 inventory 中。没有借排除日志或接受旧错误收据放宽 gate。

本次真实 smoke 不到 20k，没有声称真实覆盖阶段切换或 optimizer 更新；这些仍由既有离线原循环逐位等价性测试覆盖。实际 minibatch 分布将在正式训练记录，不能以这里的零采样证明课程学习收益。

## 来源指纹与测试

以下是实际文件字节 SHA256，不是 canonical request digest：

| 文件（相对新 run 目录） | SHA256 |
| --- | --- |
| request.json | 97dd8cb5c216020be6403a8a7c0f73e5ccd98c8399f6ebd2405d8cbd2da6c146 |
| report.json | 6074c3b8bf3d6ab795d86638d7d9c3e31bdadcc209fd9e5d896e6a60b0806f7b |
| coverage_review.json | 35c9edf1968e253adc0b1bb3514bbc6cf2035705e07ccf857cdbb85d4669b755 |
| b2_curriculum/complete.json | 68c6f94b5aec9bc140e4369755d4fdbc542788847bd5a6bfbc84c54a684683d6 |
| b3_mix/complete.json | 59bbeebd0d1ec451307025d3600ea93fbd346faedc7f990c1d91cfb374920c8c |
| b3_curriculum/complete.json | beab7a19b4f15266fe5601382c9325bbe6f44fa70371e914f07c563d55e9598a |

实现新增 19 项测试；实际定向 77 passed（8.84 秒），全项目离线 1,295 passed（36.83 秒、两个原作者 torch.load FutureWarning）。git diff/staged diff --check 通过。Gym 弃维护提示与 TensorFlow oneDNN 日志未导致工程失败，不以消除 warning 修改原科学条件。

实际执行命令（已完成的历史命令，不可重跑）：

```powershell
python -B tools/smoke_remaining_curriculum.py prepare --baseline-root E:\Prediction_RL
python -B tools/smoke_remaining_curriculum.py run --request artifacts/ic/ic03b_remaining_v1/request.json --confirm-request-hash 3f5f910adbda5ed1242e19d7630fe85d03a3d2e1d507f242f076bb95e9d98ec4
```

## 下一步与限制

可以进入 **IC04 正式请求冻结/人工审查**，不是立即训练：需明确四组×seeds 0/1/2、每模型 120k 请求步、原生回合结束超出记录、固定 20k/40k/60k/120k checkpoint 与 Gym20 开发场景 200–219、完整 resolved config/source/assets、统一任务池/比例和真实 replay/minibatch 审计。正式训练共 1.44M 请求步，评价 960 回合仍只有二十个开发交通场景；由用户批准后启动。

本阶段 formal_training_authorized=false、training_ready=false，没有 live formal task manifest 或正式训练命令；不假称已经实现。旧 B3 仅历史参照，不能替代新 B3-Mix。无 author50/泛化、新模型/奖励/Shield、测试数据开启、seed 删除、checkpoint warm-start、自动加预算。保留所有旧失败和新结果，不 push/merge/tag。
