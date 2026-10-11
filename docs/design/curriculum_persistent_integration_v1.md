# IC03：持久 learner 与回合任务流的离线接入

## 1. 范围与状态

本阶段接入 IC02h 的 L/M/H 算法核心，研究定义、32-task 候选清单、交互阈值、Mix 算术不改变。实现已通过离线合成测试，**不是可执行的正式训练协议**：没有课程训练 CLI、live task manifest、SUMO smoke 请求或新增运行授权。

新模块仅接受 `offline_integration_test_only`；测试中的 worker 复用 `FreshGym20TaskEnv` 原父进程 packet/reset/step 路径，但运输回复为合成 fixture，没有启动子进程或 SUMO。不能将合成路径结果描述成真实交通环境接入已验收。

## 2. 接入结构

`envs/curriculum_stream.py` 的 `CurriculumTaskStream` 在原始 20 维空间与连续 jerk 接口之外管理任务生命周期。构造时不启动 worker；reset 时从私有采样器取得任务，调用注入的单回合 worker factory。每个 worker 必须保持原入口 `seeded_fresh_gym20_task_v1`、阻塞执行契约与完整空间边界。

其外部仍使用原 `PredictionFeatureEnv` 和 `make_all_environment`。普通/条件预测仍为 149 个通道，与原始 observation 拼成 169 维；TimeFeature 仍由原 ALL body 追加。新回合历史清空，从当前一个 frame 开始，终止状态保留原零预测通道语义。task/level/采集阶段不得进入 observation 或 replay State.info。

`training/curriculum_replay.py` 的 `PersistentTaskReplayAudit` 在新 learner 的首次 act 前安装一次。保持同一个原 `SingleEnvExperiment`、agent、TimeFeature、optimizer、target 和 replay 跨任务存续；不重写 episode/update loop。

回合边界严格遵守以下顺序：

```text
原 worker 返回终止 observation/reward/done
→ 原 agent.act 处理终止状态
→ 原 replay.store 插入最后一条 transition
→ 审计核对 observation/action/reward/TimeFeature/mask
→ 记录 outcome，结束该任务的实际步数统计
→ 原 agent._train 和未执行的终止动作采样照旧
→ 原 episode loop 结束
→ 下一 reset 关闭旧 worker，再创建新 worker
```

即使环境已经 done，在终止 store 确认前也拒绝下一个 reset。reset 本身不插入 transition；不会生成跨任务 replay bridge。合法碰撞、自然到达、期限失败均正常记录。运输、空间、特征或对齐错误必须停止运行，不转成普通 done、stale、换 seed 或重抽任务。

## 3. 时钟与真实 replay 记录

采样器使用实际 control transition 计数，不使用 SUMO 预热 tick、墙钟、reset 次数或 agent 内部调用数。原 ALL `_frames_seen` 还计入 reset act；这条原算法语义不改，只不能用它驱动课程阶段。

边界超出由原生 episode 正常结束后记录，下一 reset 才切换阶段。独立测试在 19,999 开始的合成三步回合于 20,002 结束，记录 overrun=2；网络、optimizer 和 replay 保持同一实例。该测试只移动离线 fixture 时钟，不是伪造真实运行步数。

审计在原 `ExperienceReplayBuffer._reshape` 的**当前实例**挂观察器，位于原 `np.random.choice` 之后。通过实际 tuple identity 定位真实 replay slot，保留重复抽样索引；不再抽一次、不修改 batch、不修改安装包或全局类。支持原 ring overwrite，采集累计量、近期曝光、replay 存量、实际 minibatch 及其任务/采集阶段各自分开报告。

可选事件回调用于把 transition、episode、minibatch 索引流式保存到新运行日志；只收到副本，不保留全部轨迹在观察器内。阶段切换不清空 replay。回合完成后可恢复这个 buffer 实例的 hook；此操作不是断点恢复或重置 learner。

## 4. 已验证与尚未验证

- B2/B3 × Mix/Curriculum 四条合成路径分别与未加审计的独立参考路径逐位比较：动作、State/mask、replay tuple、policy/critic 权重、target、optimizer/scheduler、NumPy/PyTorch/Python RNG 一致。
- 22 项新增测试覆盖 terminal store 顺序、每回合一帧历史、真实 ring slot、阶段超出、worker 异常、元数据泄漏及 action/reward/time/mask 对齐错误。
- 全项目离线回归 1,240 passed；未改原科学源码、上游 reward 或训练配置。

这些验证不证明新任务协议下方法更有效，也不代替真实 worker 生命周期、SUMO raw events、预测器实际推理及哈希 postflight。新模块仍没有 live authorization 路径。

## 5. 下一项有界真实 smoke 提案（尚未执行/授权）

建议独立形成 IC03b 请求，在用户确认后再运行，不能把本阶段 offline purpose 改名绕过授权。

| 范围 | 提案上限 |
| --- | ---: |
| B2/B3 × Mix/Curriculum | 4 个独立新初始化 learner |
| 每条路径原生完整回合 | 2 |
| SUMO 会话 | 8 |
| 每回合原控制步 | 500，保留原终止处理 |
| 实际控制步总上限 | 4,000 |
| raw tick 总保守上限 | 4,816（每会话 101 预热/插入 + 500 控制 + 1 cleanup） |
| 预测特征调用保守上限 | 4,008（终止零通道实际不推理） |
| optimizer 更新 / 新反事实分叉 / predictor 训练 | 0 / 0 / 0 |

每条 learner 最多 1,000 control steps 和 1,002 原 act 调用，未达到原 5,000 warmup，因此无需缩短 warmup 或改 batch 即可验证持久存续与真实 replay 插入。实际优化等价性已在合成测试中验证。真实 smoke 不通过修改实际步时钟来覆盖 20k/60k 边界；边界调度先由离线测试验证。

任务 roster 与冻结预测器只能从已验证来源只读解析；不采新数据，不打开 test，不复用旧 policy 作训练初始化。采样 seed、optimizer 初始化 seed、具体任务清单、worker source/config/geometry、依赖、请求哈希和输出位置必须在新请求中明确冻结。

工程通过标准只包括真实 transport/终止/replay/历史/原执行审计与不可变资产，**不要求未训练 learner 到达或得到正回报**。若 500 控制步内未产生原生 terminal 或发生工程错误，保留失败并停止，不伪造 done、不自动扩预算。正式 12 个 120k 训练实例仍须在后续 IC04 通过冻结及人工批准后由用户运行。
