# P7l：作者控制器 Gym20 适配验收

## 完成范围

已实现统一 Gym20 外部比较入口：冻结作者 DDPG、ST、RL+MPC safety、
RL+MPC switching，与已有 P7k 60k 的 B0/B1/B2/B3、三个训练种子配对。
保留原作者控制器、规划器、连续 jerk/速度执行、virtual rollout 查询与接管规则。
不训练模型，不改上游或 P7k 代码，不混入 Author50 历史结果。

原 Gym 20 秒预热、0.2 秒步长、500 控制步和真实事件分析保持不变。
速度控制适配器继承原 Gym reset/step，只接入原生 speed controller；
拒绝非 Slotted Jerk 或非零 INVALID_ACTION_PENALTY，不额外转换/裁剪速度。
终止 projected jerk 为占位 0，明确不是执行动作；当前原终止奖励与它无关。
ST 与组合控制器不伪造 jerk-action saturation。

## 实际测试与来源

- 相关回归首次 125 passed；修正后全套 `pytest tests -q -p no:cacheprovider`：
  **808 passed，2 个既有可信作者权重加载 FutureWarning**。
- 初次不限定目录的 pytest 扫入历史 artifacts 临时目录，出现权限/收集错误；
  随后明确限定 `tests`，未修改旧目录权限或删除历史产物。
- 首次 `p7l_audit_v1`：DDPG/ST 参考与重放完成；组合控制器参考回合在终止
  TimeFeature 检查处失败，switching 未启动。所有失败记录保留，未覆盖或当作方法结果。
- 原因：speed controller 直接 reset 底层 Gym，未触发 ALL GymEnvironment
  的 `_lazy_init`，terminal mask 为 None。通过不涉及 SUMO reset/step 的 mask
  初始化修正，并新增 uint8 mask/TimeFeature 终止重置回归测试。
- 功能提交 `55e1665`；边界修正提交 `7cb8644`。
- 修正后 `p7l_audit_v2`：四控制器各一次无 observer 参考与一次带 observer
  重放，共 **8 回合**。每组命令/轨迹/原回报/查询次数/接管序列完全一致。
- 原始 P7k request/config/执行源码与其 Git blob 校验、12 个 60k checkpoint
  来源校验、240 个冻结评价完整文件 inventory 校验及主节点指标核对通过。
  不为本阶段读取或重新声明验证未参与评价的旧训练 replay；历史训练结果保留。
- `git diff --check` 通过；模型、日志、raw traces 不进入 Git。

修正后审计请求：`artifacts/p7l/p7l_audit_v2/request.json`。
request hash：`7e2c311dc6d260d1183139a13dd82722ce9fea40a83abe95ab1aa881c5967ac6`。
aggregate：`artifacts/p7l/p7l_audit_v2/aggregate.json`。
aggregate SHA256：`a8988600111c388720b1c96c8317f541e1406e968c66b88fa056eca67c1cb0e6`。

## 有界验收记录（不是方法效果结论）

| 作者控制器 | 场景 | 实际控制步 | 原策略查询次数 | 原始事件 | 参考/重放 parity |
|---|---:|---:|---:|---|---|
| DDPG | 200 | 149 | 149 | 自然到达 | 通过 |
| ST | 200 | 57 | 0 | 自然到达 | 通过 |
| RL+MPC safety | 200 | 192 | 960 | 自然到达 | 通过 |
| RL+MPC switching | 200 | 192 | 960 | 自然到达 | 通过 |

四作者控制器与既有项目模型的初始交通哈希均为：
`99668bc8ec5b89b09e2263bcb1b76c5c3a737db0084e4e9098868038e08e0612`。
组合控制器保留每实际步五次虚拟策略查询，不人为重设成一次。
聚合使用四个带 observer 结果与同场景的 12 个旧项目结果，只有一个独立交通场景。
不将短验收成功解释为正式外部比较胜出。

## 用户下一步与限制

完整评价新跑 **80 个作者回合**，复用 **240 个已有项目回合**，
仍只有 20 个独立交通场景。作者一个冻结 checkpoint 不是三个训练重复。
原 Author50 结果、旧 20k/40k/60k 全部保留。
完整评价尚未执行；执行步骤见 `docs/runbooks/p07l_gym20_external.md`。
完成后再做一次有界信息利用检查；本阶段未实现/启动该检查或120k训练，未批准P8。
