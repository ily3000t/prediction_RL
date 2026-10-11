# IC02h：新版交互强度规则与离线调度核心验收

## 结论与范围

新版本 L/M/H 候选层为 **11/10/11**，完整覆盖 32 个唯一训练侧源任务，每层满足原定十项覆盖要求，coverage_gate=true。工程实现、来源链和独立复算通过；这不是课程效果结论，也不代表可以启动课程训练。

旧 IC02g 的 C0/C1/C2=1/11/20 及覆盖失败保留，旧规则、配置、报告不修改。用户已明确将研究问题改为动作条件交互强度而非控制难度。D 只独立报告，不作分类或拒收条件。

- 实现提交：`5b6a2d17fd9117303ac4a357473570dcdbdd8618`，分支 `codex/interaction-curriculum-v1`。
- 只读主基线：E:/Prediction_RL，干净 main `75051acc06f6ea2ab6dda9be678f191b63d04569`，没有修改、暂存或合并。
- 独立配置：`configs/development/ic02h_interaction_intensity_v1.json`。
- 新设计：`docs/design/interaction_intensity_curriculum_v1.md`。
- 不推送、不 merge、不创建可运行里程碑 tag。

## 已实现

1. 固定 approach/2.8 秒共同窗/原尺度、逐关键邻车平均再取最大响应的 L/M/H 新规则。缺失响应或重复噪声、退化分位边界、上分位未超噪声均保留但阻止冻结，不塞入 M，不用 ID 凑数。
2. 固定 0–20k/20–60k/60–120k 三阶段调度及有限池 target；层内 task 均匀抽样，Mix 由阶段长度加权名义 reset 质量产生。
3. `OfflineEpisodeSampler`：私有 RNG、同 seed 可复现、仅回合边界换任务、实际控制步时钟、阶段入口/边界超出及总预算超出记录。明确限制为 offline_algorithm_test_only，输出执行授权始终 false。
4. `ExposureAudit`：只观察外部实际 replay slot 与实际 minibatch 索引，不存入或抽取真实 replay；报告各层、task、采集阶段的近期采集/存量/抽样分布，保留失败回合和覆盖时的旧采集总数，不追加网络输入。
5. 不可覆盖离线请求与报告。检查原科学源码和模型资产、IC02d/e 收据、IC02f 和 IC02g 固定字节哈希；仅显式允许一个新模块，没有放宽旧资产校验。

## 分层、配额和 D

| 候选层 | 唯一任务数 | I 中位数 | 最大目标接近速度中位数 | 有限池 target | 名义 Mix |
| --- | ---: | ---: | ---: | ---: | ---: |
| L | 11 | 0.1556163754 | 5.225918770 m/s | 34.375% | 33.8541667% |
| M | 10 | 2.3553624072 | 6.893702984 m/s | 31.250% | 27.2916667% |
| H | 11 | 3.9583975583 | 9.011373520 m/s | 34.375% | 38.8541667% |

下/上三分位分别为 0.3976121680515756、2.6839018875244487，重复噪声最大值 0。H 不等于困难或危险，但与接近速度关联；报告保留全部 D 分量、critical actor 数量与输入覆盖，不声称完成控制难度匹配或测量了每层预测准确率。

上述 target 是有限任务池均匀分布，不等于旧持续交通分布。Mix 是名义 reset 配额，不保证 episode 数或实际 transition 暴露相同。完整任务阶段是重访 L/M/H 全池，不是第四类任务，也不保证 replay 更新分布立即恢复 target。

## 实际执行与封存

纯离线请求在干净实现提交上运行；时间为北京时间 2026-10-11 17:47:46.959362–17:47:50.700533。新 SUMO 会话、冻结 B2/B3 预测器推理、课程 DDPG 训练均为 0；父 RNG 与不可变输入 postflight 均通过。正式 task_level 全部 null，classification_applied=false，live 标签数量 0，classifier/pi/training_ready 不发布。

从 IC02f 原逐 actor 数据用独立 PowerShell 公式复算最大平均响应、线性插值三分位和成员归属，全部 32 项一致；未重用新版分类函数、未写回源数据。

- canonical request hash：`a188fd0ab4870ca92be7ef6ae680e21835118226d59d7bf1a3c7c0aa0713d4a2`。
- request 文件 SHA256：`da4c0f0b6b75ee45cc1314233494975dbb6b6f7e13e9e2163edc88e970a17924`。
- report 文件 SHA256：`1a8694de0f3fb3f76687e08c0ce56289c809eff4d9ef4645666fb4eff7bafffb`。
- task_proposals 文件 SHA256：`a535496ad79220ca679f7cf1661403f2e16fdf1e4151a85aa575d8b151b68e17`。
- postflight 文件 SHA256：`efda7a248c43b7a0fd75142d25220517300c81f36b52f8c45ddfec898f92d5ab`。
- 产物目录：`artifacts/ic/ic02h_intensity_v1/`，不可覆盖、不提交原始数据。

## 测试与环境差异

- 定向新旧规则测试：89 passed，含 51 项新测试；测试了真实封存训练数据、异常/不可分类项、配置边界、RNG、阶段越界、任务配额及真实索引审计。
- 最终全项目离线回归：**1218 passed，2 个作者原加载器 torch.load FutureWarning，48.65 秒**。没有 SUMO 实验；测试仅使用假环境、合成更新及已验证作者模型的离线加载/查询。
- 首次全回归使用系统默认 pytest 临时目录，被 Windows 拒绝访问；未改系统权限。改用明确不存在的工作树测试目录后，1217 项通过，另一项因工作树没有被 Git 忽略的作者预训练权重失败。
- 最终全回归仅在测试进程中显式设定 `compare_external_baselines.MODEL=Path('E:/Prediction_RL/RL-MPC-LaneMerging-master/pretrained_models/ddpg_default1_extended')`，先调用原 assets() 完整检查原文件集合和哈希；其余测试、源码、加载器不改变，没有复制权重或建立可写链接。正常训练 CLI 的资产路径没有修改；不能把这次测试路径适配描述成工作树已具备全部作者资产。
- `git diff --cached --check` 通过。测试临时目录限定为本次创建的 ut_ic02h_* 三个明确目录，实验报告及旧资产保留。

## 尚未完成与下一项任务

本次只有离线采样/审计核心，**没有课程训练 CLI、持久 DDPG learner 与 fresh-task worker 的接入，没有运行真实闭环 smoke 或匹配训练**。原 reset 不支持任意 task seed 切换；不得直接对旧持续 SUMO 环境调用新 sampler 冒充合法课程。

下一项 IC03/IC04：审核并冻结新规则/有限池目标，接入同一批准 fresh Gym20 任务入口，验证 learner RNG、原 reward/action/bootstrap/TimeFeature、真实 replay/minibatch 索引和预测器哈希不变；随后提出有明确回合/控制步上限的工程 smoke，验收后交给用户运行四组×三个 seed×120k。

不增加 author50/泛化实验，不补采，不扩大预测器，不改奖励/commitment/Shield，不由 coverage_gate 自动启动训练。学习效果差也正常聚合，不删 seed、改阈值、覆盖报告或延长预算。
