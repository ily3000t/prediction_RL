# P7i 训练稳定性诊断入口验收（2026-10-03）

## 已完成与边界

新增独立模块、CLI、配置、测试及运行文档；没有修改旧源码/配置/结果，
没有重新训练、更新 predictor、修改原奖励或打开封存测试集。
E:/WCDT_ACCVP 保持只读。完整诊断由用户运行，P8 未批准。
实现提交：f649c8e919dc4fee076e8508b2075e955eb31605。

本次实际验证：

- 专项及邻接回归：93 passed。
- 完整回归：738 passed，2 条既有作者 pickle 加载警告。
- git diff --check 通过；未安装或修改依赖。
- 全部 12 个训练日志的 actor/critic scalar 数与 optimizer updates 相符；
  训练回报 step/value 与原 episodes.json（float32 日志精度）精确匹配。
- 没有将 actor 的负 Q loss 当作准确率或收敛证据。

## 有界真实 SUMO smoke

artifacts/p7i/p7i_audit_01/request.json。
canonical request hash：27ab86014b98c46ddffbae8937b0d933d6bd2aa1e8a2f53a69b80cd81d3e1c56。
aggregate byte SHA256：b19e7f58810cfca8bbee8c81d4df3df6a473eb77129790a58b90274f690e8096。

四组、训练种子 2、场景 200、确定性与 noise repeat0，8 回合共 3,253 次
控制调用。全部完整封存；4 个确定性动作/奖励/native outcome 精确匹配
旧 P7；所有策略/critic/predictor 冻结，replay 和 optimizer 未更新。
原始事件、噪声序列、控制统计和完整聚合均可复算。

| 方法 | 确定性原始事件 | 加噪声原始事件 | 确定性 / 噪声控制步数 |
| --- | --- | --- | ---: |
| B0 baseline | 自车参与碰撞 | 自车参与碰撞 | 123 / 130 |
| B1 zero | 任务时间上限 | 任务时间上限 | 500 / 500 |
| B2 ordinary | 任务时间上限 | 任务时间上限 | 500 / 500 |
| B3 conditional | 任务时间上限 | 任务时间上限 | 500 / 500 |

这是失败路径覆盖的工程审计，不是噪声效果主表。没有回合被“修成成功”。
不能用一个种子、一个场景、一个噪声序列宣告噪声有效或无效。
complete-only resume 实际验证完成：8 reuse、0 run，没有新增 SUMO 回合；
现有 aggregate 和收据不覆盖。

## 只读训练日志的初步线索

以下只是原始训练日志，不是新评价或模型修改依据：

- B2 种子 2 的 q TD MSE 均值：5k–10k 为 0.262592，15k–结束为 1.532439。
  支持继续检查训练状态，但 replay 内容变化也可影响 loss，不能唯一归因。
- B3 种子 2 在 15k–结束原训练回合中的 native 到达比例为 83.33%，平均
  回报 0.893845；其本次确定性开发回放仍停车至时间上限。
  原训练是有噪声、交通连续演化；这不是相同评价场景下的对照因果证明。
- q loss 降低、actor negative-Q loss 变得更负、单个种子表现好，都不能
  自动证明价值估计正确或 DDPG 已收敛。

## 用户执行的完整请求

已准备：artifacts/p7i/p7i_diag_v1/request.json。
canonical hash：eb52a72518773039aca4b6de215d2667fdec40ba02255886293b76cdd0c7299a。
准备源码提交同上，工作树干净。原始训练曲线另存于 training_curves.json，
窗口审计另存于 training_audit.json；原日志、权重及数据不改变。

全四组 × 全三个训练种子 × 原三个场景；36 确定性 + 108 噪声 = 144 回合。
目录中没有 episodes、invocations 或完整 aggregate；未启动完整评价。
同场景/repeat 共享噪声前缀，但不会改变全局 Torch/NumPy/Python RNG。
统计先均值 noise repeats，再场景，再等权训练种子；仅三个独立交通场景。
保持 Gym20，不能与 author50 外部主表混算。

启动命令见 docs/runbooks/p07i_ddpg_stability.md。完整结束后先分析探索/
确定性执行敏感性、低速动作和 critic，不自动扩展预算、数据或网络。
新文档提交与普通 merge 不改变冻结的 Python 源码指纹。
