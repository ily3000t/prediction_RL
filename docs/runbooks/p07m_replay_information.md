# P7m：只读 replay 信息利用诊断

本阶段没有训练或仿真。原奖励、连续 jerk、60k checkpoint、预测器、三个训练
种子和 Gym20 开发评价结果不变。先完成一次短信息利用检查，再人工决定是否
批准另一个固定预算实验；本工具没有训练入口。

## 查询内容与解释边界

| 扰动 | 改变的通道 | 能说明什么 |
| --- | --- | --- |
| zero_all_prediction | 149 个预测/有效性通道全部归零 | 策略对整个扩展输入的敏感性 |
| zero_values_keep_metadata | 140 个预测数值归零，9 个有效性通道不变 | 策略是否依赖数值，避免完全混入 mask 移除 |
| candidate_mean_keep_metadata | 五候选的角色/距离值各自取均值并广播，mask 不变 | 策略是否依赖候选间数值差异 |

三者均保留原 20 维观测与 replay 内原 TimeFeature，不重复追加时间。
清零/平均可能产生分布外甚至物理不一致输入，所以动作变化不等于闭环收益。
普通预测 B2 的相对几何也随候选自车轨迹变化；第三项**不能独立证明**
B3 利用了动作条件邻车响应。B2/B3 查询各自 replay，不是相同状态的因果比较。

原网络严格加载 policy 和 critic，但只查询 actor。保存动作与绝对 jerk 变化
mean/p50/p95/max、超过 0.1 m/s³ 的比例。0.1 只是预先固定的描述参考，
不是“方法成功”阈值。每个种子分别报告，不以最大敏感性选择模型。
训练全程 replay 与最终冻结策略的状态分布也不同，不能冒充独立验证。

## 用户完整运行

在项目目录、pytorch 环境、干净已提交工作树下：

```powershell
python -B tools/diagnose_replay_information.py prepare --run-id p7m_info_v1
```

检查打印出的 request 与 hash，再执行（将占位符替换为打印值）：

```powershell
python -B tools/diagnose_replay_information.py run --request artifacts/p7m/p7m_info_v1/request.json --confirm-request-hash <prepare打印的hash>
```

最多扫描六份约 60k replay（总约 360k 状态），最多查询 6144 个不同状态，
每个状态原始/三扰动只读查询；CPU 一线程、顺序执行，不启动 SUMO。
这不是新的 episode 评价，不改变任何历史产物。

完整结果已 sealed 时可用相同命令追加 `--resume` 验证复用。
失败/未完成 model 目录不能自动重跑。排查后新建 run ID，旧失败保留。
运行期间不要修改源码、配置或输入文件。

## 结果位置

- artifacts/p7m/p7m_info_v1/aggregate.json：六模型完整报告、已有失败场景与解释限制。
- models/<arm>_s<seed>/report.json：uniform 主样本与四个独立分层统计。
- selected_states.json：采样身份、实际输入和原 physical_before。
- action_rows.json：原动作与三种扰动动作，按 transition_id 对齐。
- complete.json：只读 guard 与不可变文件清单。

失败场景仅引用现有 Gym20 结果；不是采样依据。replay 中没有完整邻车未来
真值，也没有这些评价回合的完整模型输入，因此报告明确给出
`prediction_error_attribution=not_identifiable_from_saved_replay_vectors`。
不把“查不清失败归因”作为继续投入长训练或更多诊断的自动理由。

## 有界工程 smoke（开发验收，不是方法结论）

```powershell
python -B tools/diagnose_replay_information.py prepare --audit --run-id p7m_audit_v1
python -B tools/diagnose_replay_information.py run --request artifacts/p7m/p7m_audit_v1/request.json --confirm-request-hash <audit打印的hash>
```

每模型只扫描前 128 条，最多查询 24 个状态。仍严格核验全部引用文件 SHA，
扫描前缀不代表整个 replay 分布。agent 构造会按原 preset 初始化临时网络，
随后严格加载冻结权重；“无 RNG draws”指采样选择和查询，非临时网络构造。
