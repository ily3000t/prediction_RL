# IC02a 任务入口设计验收

## 结论与实际完成范围

纯离线源码/映射核查已完成。原连续交通 reset 没有按 task/seed 选任务的入口；推荐提出独立的 `seeded_fresh_gym20_task_v1`，但它**未批准、未实现、未运行**。

该提案保留原生成器配置、Gym20 预热、ego 插入位置、原 first observation、1 帧初始预测 history、连续 jerk、reward/terminal 与 TimeFeature；新 SUMO 位于独立 worker，learner/replay/探索状态持续存在。它改变原连续训练的场景入口，所以四组必须共同使用新协议，旧 B3 只能保留为历史参照。

IC01 的 C0=0 未修成通过。入口方案本身不能证明三层可用，task-level classifier 与目标分布 pi 仍未定义；课程训练保持暂停。

## 运行与来源

- 分析源码/配置：`cc3335d858fb806ebf31f3b56dc9d9be9a72a207`，运行时工作树干净。
- 隔离工作树与分支：`E:/Prediction_RL_worktrees_course_learning`，`codex/interaction-curriculum-v1`。
- 原只读基线：`E:/Prediction_RL`，main，`75051acc06f6ea2ab6dda9be678f191b63d04569`，核查前后干净且 HEAD 不变。
- 实际命令：`E:/Programs/EnvAnaconda3/envs/pytorch/python.exe -B tools/audit_curriculum_entry.py --baseline-root E:/Prediction_RL --run-id ic02a_v1`。
- 报告保存实际解释器启动参数、开始/结束时间、OS/Python、解析设计、源码与输入指纹。
- 验证 33 项输入引用：11 项冻结基线资产、IC01 报告/root、8 个源文件在两目录的来源及本次设计代码/配置引用。
- 仅检查文件和 AST，没有导入环境模块、运行 SUMO、预测器、DDPG、补采分支或打开 predictor test。

实际产物在本工作树 `artifacts/ic/ic02a_v1/`，Git 忽略，不可覆盖：

| 文件 | 内容 | 实际字节 SHA256 |
| --- | --- | --- |
| `report.json` | 源码事实、设计、输入指纹、未授权 smoke 预算及待决定事项 | `c47dd2134446892361fa84321cfb228d155effbc07638b239dd13e19b02fb888` |
| `episode_mapping.json` | 旧 train source episode 与 root 的关联，不是可执行任务池 | `4149284510cd4d83216a00f3051091fffee9c26434d273ecb3b54fee1d69fa91` |

## 工程核查结果

| 项目 | 结果 |
| --- | --- |
| 静态源码假设 | 14 项通过，不等于 runtime parity |
| 原 reset 是否能指定 task/seed | 否 |
| 等待/插入 tick | 100 / 1，0.2 s/tick |
| 原插入条件 | rampRoute、departPos=40、arrivalPos=50、speedMode=22 |
| 学习型首个预测 history | 1 帧，不是免费完整历史 |
| 会话是否播种 Torch 等全局 RNG | 是，必须放环境 worker，不能每 task 重播种 learner |
| source episode / root | 256 / 713 |
| 含暂定 C1 root 的 source episode | 181，未赋予 episode-level C1 |
| 已赋 task 层 / 获授权任务 | 0 / 0 |
| 新入口 runtime parity | 未验证 |
| training_ready | false |

上游配置覆写被正确纳入静态解析：Slotted Jerk 与 ALT_J_WEIGHT=0.1，而非源码默认值 0.05。起步配置变量虽叫 VARIANCE，实际传入 NumPy normal 的标准差；没有修改其语义。

129 项测试通过（24 项 IC02a + 105 项 IC01/既有回归）；CLI help、工作树和 staged `git diff --check` 通过。测试拒绝把设计 authorized/training_ready 改为 true、增加前缀或补历史、root 层晋升 task 层、test seed、错误 source join、unknown fields 与改变 RNG/reset 假设。所有原运行源码仅只读核查。

另一次只读本地依赖检查发现：原基线有 `st_cy.cp310-win_amd64.pyd`（206,848 bytes），工作树没有该忽略文件。该观察不是 AST runtime gate；批准运行前需要在工作树完成同来源扩展构建/验收，不可静默关闭 Cython 或从旧目录导入源模块。

## 待批准的下一步

建议批准 IC02b 的**新入口原型和有限一致性 smoke**，不是完整课程训练：

- train seeds 11000/11001/11002；原 direct 与 worker-entry 两条路径，各重复 2 次，最多 12 个工程 episode。
- 固定 12 个连续 jerk/episode，最多 144 控制步；等待/插入另计，最多 1,212 tick。自然终止提前结束，不替换 seed。
- 普通/条件冻结 predictor 各最多 156 次当前状态特征调用；0 optimizer updates、0 新真实候选分叉。
- 验证首状态、当前交通、动作/reward/done、mask/history/摘要、重复一致性和 learner RNG 隔离；合成终止 fixture 补充传输检查，短 live smoke 不声称覆盖全部终止。
- 12-step 上限不是 MDP terminal，不作为训练数据、开发性能评价或方法优越性证据。

必须明确接受“每 task fresh 首回合”这一共同协议变化。新 target pi 不能宣称恢复了旧 continuing-traffic 分布；任务覆盖、分层、最后阶段目标权重及四组训练请求仍需后续独立验收。用户批准前不改 reset、不构建运行入口、不启动 SUMO。

设计详见 `docs/design/curriculum_task_entry_v1.md`，已实现离线命令见 `docs/runbooks/ic02a_entry_design.md`。本阶段只保留本地原子提交，不合并 main、不推送、不创建可运行课程标签。
