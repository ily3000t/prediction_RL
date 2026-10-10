# IC02b：入口轨迹一致性及离线恢复验收

## 结论

原 direct 与新 `seeded_fresh_gym20_task_v1` worker-entry 的封存轨迹、当前观测、奖励、动作、原始事件、history 和两种冻结预测摘要，在全部 12 项重复/跨路径比较中精确一致。原聚合失败来自设置字典的 JSON 键表示，不是控制或预测差异。

**封存产物一致性通过；原进程的完整 postflight 未恢复；课程训练仍未获准。** 原 `failure.json` 和 `engineering_gate=false` 保持不可覆盖，单独离线报告不冒充一次成功重跑。原运行在比较阶段退出，没有保存最终内存 tensor/global-RNG 检查，不能由当前模型文件哈希或逐回合 RNG witness 补造。

## 来源与实际执行

- 原实现及 live smoke：干净的 `5bef30bac6e08249e77431fbfd29096c01ca1fc7`。
- 聚合修复及离线复核：干净的 `e773ad48ffa1ec42e611f388bb5546a8788ba602`，分支 `codex/interaction-curriculum-v1`。
- 新工作树：`E:/Prediction_RL_worktrees_course_learning`；只读基线 `E:/Prediction_RL` 的 main/HEAD 仍为 `75051acc06f6ea2ab6dda9be678f191b63d04569`，核查前后干净。
- 原 live smoke 用约 184.438 秒，实际执行 12 回合和 144 控制步，随后最终聚合失败。12 份回合 seal 均完整。
- 本次只离线读取原产物：0 新回合、0 新控制步、0 预测推理、0 optimizer update、0 DDPG、0 新真实反事实分支，不打开 predictor test。

实际离线命令（已执行，不需要重跑）：

```powershell
E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -B tools/review_curriculum_entry.py --request artifacts/ic/ic02b_v1/request.json --confirm-request-hash 7bf5fd155c48bc28b34adfff916a6bb074e3a1870b334a08be32f19784fd7a41 --run-id ic02b_review_v1
```

报告保存实际命令、解析协议、日期时间、来源 commit、依赖版本和全部输入/receipt 指纹。原运行与离线复核目录均在工作树的 Git 忽略 `artifacts/ic/` 中；不提交轨迹、日志或模型。

| 产物 | 作用 | 字节 SHA256 |
| --- | --- | --- |
| `ic02b_v1/request.json` | 原冻结请求；canonical request digest 是命令中的 `7bf5fd…`，不等于文件字节哈希 | `cbbb9dc80ae711adab8bcbb674b983e608f67556932fa09584c12db8f212e4b4` |
| `ic02b_v1/failure.json` | 保留原聚合失败及 12 行完成记录 | `4095269227d34b3cedfaf2c79010e66da0bf31535a64cbb7ac2bdaae3a6a22e7` |
| `ic02b_review_v1/report.json` | 零新增仿真的独立精确复核、计数与证据限制 | `2bd2c670858e3deae05261f2335987858b7f3aebc8cf33889f6293efe72339af` |

## 精确比较与预算

| 比较 | 比较数 | 每项状态/控制步 | 结果 |
| --- | ---: | --- | --- |
| direct 同 seed 重复 | 3 | 13 / 12 | 全部精确一致 |
| worker-entry 同 seed 重复 | 3 | 13 / 12 | 全部精确一致 |
| direct 与 worker-entry，同 seed/repeat | 6 | 13 / 12 | 全部精确一致 |

只有原协议已排除的 PID、测得的特征 latency 和路径 job 身份不参与语义比较；没有新排除字段、容差、数值修正或种子改变。121 项原输入指纹全部核查：只有 aggregation runner 发生预期修改，其原字节由历史 Git blob 验证；环境、特征、比较代码及冻结权重文件不变。

| 实际计数 | 数值 |
| --- | ---: |
| simulator seeds | 11000 / 11001 / 11002，共 3 个训练侧场景 |
| 工程回合 | 12，不是 12 个独立场景 |
| 控制步 | 144 |
| 等待/插入 tick | 1,212（每回合原 100+1） |
| raw tick | 1,356，0.2 s/tick |
| 当前状态特征调用，每 predictor mode | 156；两 mode 合计 312 |
| 自然 terminal 回合 | 0 |
| 最后状态 | 全部为原 done=false 的工程 cap 停止 |

首次预测 history 为 1 帧，按原实现增长至 11 帧，不预填历史。原观测 20 维，原预测增强观测 169 维；TimeFeature 控制步元数据按原 0.001 尺度核查。没有实例化 ALL learner/replay，因此不把该检查称为实际学习器持续状态验收。短 live trace 没有碰撞/自然到达/deadline，相关传输/cleanup 语义由已有离线 fixture 覆盖，不夸大 live 终止覆盖。

## 原因与修复边界

上游 `config.py` 的 `JERK_VALUES_DQN` 使用整数键。direct 子进程写出的 native trace 被父进程从 JSON 读取，键已变成字符串；worker 的设置仍在内存，键仍为整数。原最终聚合比较混用这两种表示，尽管保存后的完整 trace 精确一致，仍判为不一致。该设置是审计元数据，连续 jerk 动作接口未改为 DQN。

修复：所有新鲜/复用回合在 seal 后通过 `read_sealed_trace` 验证 inventory、绑定及 trace 字节哈希，再统一从 JSON 读取后聚合。原比较函数完全不改，模型/环境/奖励/动作/feature/schema/seed/预算不改。

离线诊断通过只读 AST 提取已验证的原设置 literal，单独重建键类型差异，并用未修改比较器复现失败；JSON roundtrip 后精确通过。这是**来源代码支持的表示重建**，不是原进程内存快照。复核重新检查全部 12 份 seal、原顶层哈希和输入，保留原失败。

## 权重、RNG 与缺失证据

| 证据 | 状态/含义 |
| --- | --- |
| 两种三成员 ensemble 文件 | 基线字节 SHA256 未变，只读引用，无训练或重新加载推理 |
| ordinary 文件 SHA256 | `939242f358315bd00ce89978422e4b55b8059f41837463d2e80f661c63817e8d` |
| conditional 文件 SHA256 | `daddeda8569b88c2a3482a0e8c739995785ee0f8216dda6525c86d2ce120e25f` |
| 每回合父进程 RNG witness | 12/12 为 true，来源于原 sealed metadata |
| 原最终内存权重 tensor snapshot | 不可用；原执行未到达该检查 |
| 原最终全局 RNG postflight | 不可用；逐回合 witness 不替代它 |
| `artifact_parity_gate` / `runtime_parity_verified` | true，仅确认已封存运行轨迹的精确一致性 |
| `full_original_engineering_gate_recovered` | false，不将原失败改成完整成功 |
| `training_ready` | false |

本地扩展按原 setup 在本工作树构建，Cython 保持开启，两路径使用相同扩展。实际本地 pyd SHA256 为 `5e1086f3cb5336f0c5ba428bb05bdd64b0fc6f613c63a76eb147fdb8d59ea7d9`；不宣称它与历史基线扩展字节相同。复核只检验已记录 executable/extension/dependency 指纹，不重新运行 SUMO version 命令。

## 测试与下一阶段边界

328 项离线测试通过：原 311 项入口/feature/actor/dataset/collection/terminal 回归，加 17 项序列化、seal/绑定/哈希损坏、历史来源漂移、完整 12 格复核、RNG 失败保留与零额外预算测试。CLI help 和 `git diff --check` 通过。新封存 fixture 起初未将 ndarray 转为 JSON，修正为生产路径已有的 `json_safe` 后全部通过；没有因此修改模型或共享序列化语义。

仍未解决：IC01 C0=0；root 不能自动晋升为 task；task-level D/I 规则、完整覆盖和 fresh-task target `pi` 未冻结；持久 learner/replay 接入未实施。入口轨迹一致性不能补齐这些科学/训练条件。

下一步只设计训练侧 task 参考刻画、分层覆盖与有限数据预算。若需要新 live smoke/反事实采集，必须单独批准版本与预算；不因本复核通过直接启动课程训练、重复原 smoke、合并 main、推送或打可运行课程 tag。
