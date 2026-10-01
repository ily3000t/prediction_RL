# P7d 外部基线评价入口工程验收

日期：2026-10-01。阶段：开发集外部系统对照准备，不是 P8 正式方法验收。

## 已完成

- 新增 `tools/compare_external_baselines.py`：prepare/run/aggregate/audit，严格验证父 P7、模型、配置、运行环境、源码和不可变回合 receipt。
- 预先纳入作者 DDPG、ST、两版 RL＋MPC，以及当前三份冻结 conditional DDPG；没有修改或训练 predictor/DDPG。
- 所有组直接调用未修改的作者 `control.run_episode`，50s 预热、0.2s 步长、名义100s/最多501次控制；作者速度命令与虚拟策略查询顺序保留。
- 单场景记录/原始控制器回放精确对齐；不同方法初始完整交通快照哈希一致。
- 所有方法共同环境/奖励/规划器基础配置一致，只有公开声明的 supervisor 开关、模型路径和编排字段不同。CPU 推理覆盖显式记录。
- 负结果正常聚合；成功条件效率与双方均成功的配对时间分开；控制器时延报告真实样本 mean/p50/p95/p99/max，无0.5秒效果 gate。
- 未修改 `E:\WCDT_ACCVP`，未覆盖旧 P7/P7b 或模型产物，未推送远端。

代码提交：

- `b5902db` — `feat(evaluation): add frozen external RL-MPC baseline comparison`
- `f53d9dd` — `fix(evaluation): canonicalize upstream settings before request hashing`

## 测试与真实工程运行

完整单元/回归测试：**535 passed in 21.77s**；Git diff whitespace 检查通过。
新增44项检查覆盖固定 roster/未知设计拒绝、原评分规则、JSON规范化、负结果、
成功条件效率、场景配对、精确行为对齐、输入布局、终止 mask、未知pickle文件拒绝及延迟长尾。

验收目录：`artifacts/p7d/p7d_audit_v2/`。
请求哈希：`1fa2b4f263ab3783cf320c23b166d345f1caed343aaa1dae7c0ff8d7a279aca5`。
aggregate文件 SHA256：`847aa1bfdc1e88d07895c47ac9464fddb3cc3ef7eb1cfffddca9904b0fd9b006`。

| 工程检查对象 | 单次记录版控制步 | 原控制器回放对齐 | 冻结检查 |
| --- | ---: | --- | --- |
| 作者 DDPG | 144 | 精确一致 | 权重未变 |
| 作者 ST | 157 | 精确一致 | 无学习更新 |
| 作者 RL＋MPC 安全接管 | 187 | 精确一致，含接管序列 | 权重未变 |
| 作者 RL＋MPC 安全＋效率切换 | 187 | 精确一致，含接管序列 | 权重未变 |
| 当前条件预测＋DDPG（训练种子0） | 354 | 新原循环接入 smoke，不与旧P7混算 | policy/q/predictor及optimizer/replay均未更新 |

四个作者组各做两个回合，当前模型一个回合：**9 回合、1704 次控制调用**。
它们在这个工程场景均到达、无原循环碰撞事件；这不能推出任何方法优越性。
作者 hybrid 的接管 bool 序列、速度命令、位置/速度/加速度/jerk、原距离统计和终止逐项精确一致。
另一次 `--resume` 只复用全部5个已验证 job，没有新仿真；aggregate SHA256 不变。

第一次请求 `p7d_audit_v1` 在启动仿真前因整数键 JSON round-trip 哈希不一致被拒绝。
该请求保留，未覆盖、未仿真；通过规范化哈希及回归测试修复后，改用新验收 ID。
这不是模型失败或修改评分阈值。

## 已冻结、尚未执行的完整对照

- 配置：`configs/development/p07d_external_baselines_v1.json`。
- 请求：`artifacts/p7d/p7d_compare_v1/request.json`。
- 请求源码提交：`f53d9ddd97ba6d2c951491a5ad983347d3a0ba3c`；后续本次修改只增加文档，执行时另记录实际HEAD。
- 请求哈希：`18a2434c49a9f44ff27b0b0e27221b2d1914af2703373bd606609febac0a67e3`。
- 交通种子200–219，作者四种控制器各20回合；当前三个训练种子各20回合：共140回合。
- 已核查没有 `evaluate/` 目录。完整运行由用户手动启动，自动聚合。

完整命令见 [运行文档](../docs/runbooks/p07d_external_baselines.md)。

## 必须保留的解释限制

1. 作者提供一份default权重，不是三个独立训练；其实际训练seed/完整预算来源不足。本轮不匹配训练预算，不能宣称正式同预算全面超越。
2. 两版作者 hybrid 含ST监督，当前B3不含ST；对比回答完整系统效果，而不是仅预测器的因果贡献。内部B0–B3对照仍有作用。
3. 每场景独立新进程首回合，不复现作者4000连续回合统计；这是透明的开发集配对适配。
4. 原循环报告任意车辆碰撞，不能写作已核验的ego-only碰撞。最近距离只取原代码限定的合流后区段，并有100m截断。
5. `author_history_total_reward` 调用原历史评分函数，共N−1项；不是旧P7 Gym return。旧回合不进入新对照聚合。
6. 9回合只证明工程接入可用，不证明当前模型优于RL＋MPC；无新的模型改动、P8、sealed-test访问或正式效果结论。
7. 原源码发布许可仍未解决；本轮只创建本地提交，不推送来源代码或权重。
