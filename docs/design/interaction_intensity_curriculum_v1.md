# IC02h：交互强度课程新版本

## 研究定义与不变项

本版本研究“按参考条件下的动作条件邻车响应强度组织训练，是否帮助策略利用冻结 B3 信息”，不研究低控制难度到高难度。旧 IC02g 的 C0/C1/C2=1/11/20 和覆盖失败保持不变；新 L/M/H 不是旧规则修复或旧 gate 放宽。

复用 IC02f 全部 32 个 train tasks/96 roots。固定 approach anchor、14 ticks/2.8 秒共同有效窗、旧 dv/ds 尺度，逐关键邻车平均候选对/时间响应，再取 actor 最大值。禁止包括 ego、用不同候选移动的自车坐标、删去额外关键车、按策略输赢选任务或改变时窗。当前 I 描述作者参考策略条件下的任务交互潜力，不代表新策略每回合实际经历的强度或信息价值。

普通/条件预测器、候选 jerk、执行周期、原 reward/terminal/bootstrap、网络/optimizer/noise/均匀 replay 不变。真实候选未来仅用于离线分层，不进入在线 observation、reward、动作或 teacher。旧 B3 只作历史参照，四组新训练共用相同任务入口。

## L/M/H 与不可分类项

- L：有效 I 不高于下三分位；不意味着简单或安全。
- M：有效 I 严格处于上下三分位之间。
- H：有效 I 不低于上三分位，并严格超过全部 approach 重复噪声最大值 + 1e-10。
- 缺失完整响应/重复噪声证据、边界退化、上分位响应未超噪声均不能归入 M 或用 ID 补齐配额。保留记录但阻止冻结；每层仍至少十项唯一来源任务。

D 所有分量、关键车数量/角色和 predictor 输入覆盖独立报告，不参与分层或删除任务。投影 gap/TTC 不能解释成跨路径即时物理安全保证。H 若同时具有较高接近速度，需要在结果中说明混淆；仅四组差中之差不唯一证明信息利用机制。

## 固定调度与 Mix

| 实际控制步区间 | L | M | H |
| --- | ---: | ---: | ---: |
| 0–20k | 0.60 | 0.30 | 0.10 |
| 20–60k | 0.20 | 0.20 | 0.60 |
| 60–120k | 有限池层质量 | 有限池层质量 | 有限池层质量 |

层内均匀 task 抽样。目标是 32 个任务均匀分布，不是旧持续交通分布。若候选为 11/10/11，target=34.375/31.25/34.375%，按阶段长度加权 Mix=33.8541667/27.2916667/38.8541667%。这些是名义 reset 配额，不保证任务次数或 transition 曝光相同。

阶段由累计实际控制 transition 决定，在下一回合入口应用；不中断 episode。记录实际阶段入口、边界超出步数及请求 120k 的超出步数。采样使用独立 Random 实例，不播种或消耗网络、探索、SUMO 的 RNG。阶段切换不清空 replay。最终 60k 再接触有限池目标，不宣称更新分布立即恢复目标。

## 工程核心与未实现范围

`curriculum/interaction_intensity.py` 提供纯离线规则、调度算术、`OfflineEpisodeSampler` 和 `ExposureAudit`。采样器只允许 `offline_algorithm_test_only`，输出始终 execution_authorized=false；不能作为训练授权。审计仅观察真实 replay slot/实际 minibatch 索引，不插入或抽样 replay，不追加策略输入。记录每层/任务/采集阶段的近期采集、存量及实际抽样，保留所有终止与失败。

当前还没有持久 DDPG learner 与 fresh-task worker 的调度接入、训练 CLI 或运行授权。原 reset 无 seed 切换接口，不能直接把 sampler 结果传给旧 reset。后续 IC03/IC04 应在同一批准入口下完成接入、预测器哈希和真实 minibatch 对齐、有界工程 smoke，再由用户启动四组匹配训练。

预算仍为 B2/B3×Mix/Curriculum×seeds 0/1/2，各 120k 请求步；评价节点 20k/40k/60k/120k，场景 200–219，主终点 120k。无新泛化/author50 实验，无自动预算扩展。960 回合仍只有二十个开发场景。

## 配置、来源与离线运行

独立配置 `configs/development/ic02h_interaction_intensity_v1.json`，不 extends，不修改旧配置。CLI 检查原 baseline、预测器/原科学源码、IC02d/e 收据、IC02f 四项固定字节哈希及 IC02g 四项固定哈希。只允许新增一个明确模块，不能忽略未知科学源码。

在干净课程工作树中，以下命令仅复核新提案，不启动 SUMO/模型/训练：

```powershell
python -B tools/propose_interaction_intensity.py prepare --baseline-root E:\Prediction_RL
python -B tools/propose_interaction_intensity.py run --request artifacts/ic/ic02h_intensity_v1/request.json --confirm-request-hash <prepare打印的hash>
```

输出使用新的短 run ID 且不可覆盖，记录 Git/config/源文件哈希、命令、环境、父 RNG 与输入 postflight。coverage_gate=true 只证明这份分层提案的工程覆盖；classifier/pi/live labels 和 training_ready 仍不发布。正式训练前需审核新规则与匹配入口并通过接入 smoke。
