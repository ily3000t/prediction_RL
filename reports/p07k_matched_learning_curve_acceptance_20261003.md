# P7k 匹配 60k 学习曲线入口验收（2026-10-03）

## 已完成与未执行边界

已实现本轮用户批准的有界开发实验，尚未执行完整训练或完整评价。
固定 B0 无预测、B1 零预测通道、B2 普通预测、B3 条件预测；训练种子
0/1/2，各从头连续训练至 60k，保存 20k/40k/60k 节点。
请求预算为 12 个训练实例、720k 控制步；节点继承原回合边界，可有
至多 499 步 overshoot，保存实际步数。20k 是新运行内部参照，不承诺
与旧 20k 权重逐位相同。最终 60k 为主终点，不按开发评价选择最佳节点。

预测器直接复用旧冻结三成员 ensemble；未重新收集数据、训练预测器、
更改结构、reward、Shield、seed、reset 或连续 jerk 接口。
原 ALL 0.5.3 参数保持：gamma=.98、lr=.0002、replay warmup=5000、
batch=100、容量=1e6、原 2e6-step 调度、CPU 单 torch 线程。
训练 reset 继续原 SUMO 交通；NumPy replay 采样与随机初速的耦合保持。

每个冻结快照在独立新进程执行 Gym20、20 秒原预热、无探索、无更新，
开发场景 200–219，共 720 个回合，仍只有 20 个独立交通场景。
先在同训练种子内按 simulator seed 配对，再等权聚合三个训练种子。
不混入 Author50，不打开新测试集，不自动延长至 100k/200k。

## 前置证据

严格校验旧 P7 请求、训练/evaluation 收据与冻结预测器，并校验 P7j
全部七个 job、source/history/runtime 和 aggregate receipt。
P7j aggregate SHA256：
`8ec67ea67d9be2f29238accb6563488636322ae4426c89292a9cbc12b38de7a9`。
P7j 原有 `historical_20k_replay_coverage_gate=null` 没有被改写为通过；
新采样覆盖只说明新运行，不能补造旧 replay 或完整分布等价证据。

## 实现产物

- `configs/development/p07k_matched_60k_v1.json`：独立完整冻结配置，无 extends。
- `tools/train_matched_ddpg.py`：prepare/train/evaluate/aggregate/独立 bounded audit。
- `src/prediction_rl/training/matched_learning_curve.py`：原训练对象分段快照；
  保存实际 replay 及原 `_reshape` 接收到的采样 transition IDs，不重复采样。
- `src/prediction_rl/evaluation/matched_learning_curve.py`：严格 roster、节点、
  事件、同场景 hash 与按种子配对聚合；负结果不触发方法必须成功的 gate。
- `tests/test_matched_learning_curve.py`：配置、预算、等价性及负结果/错误证据测试。
- `docs/tasks/p07k_matched_learning_curve.md`、对应 runbook：范围、停止条件与手动命令。

新训练流式保存实际 state/action/reward/next/mask 和物理 sidecar，
以及 minibatch 实际选中的 transition IDs；reset bridge 被拒绝。
按节点累计报告低速、匝道位置、history、actor presence/omission、
采样覆盖与低速恢复转移；B0/B1 的学习特征覆盖缺失明确标为不可用。
保存上游真实 actor/critic optimizer loss；训练回报明确含 exploration，
不能把 ALL 的 `evaluation/returns` 训练 tag 当验证曲线。
另保存独立冻结策略的 `development_curves.json`。

## 实际测试

新增单元测试 23 passed；完整回归 781 passed，2 条既有作者
`torch.load(weights_only=False)` 警告。AST 和 `git diff --check` 通过。
默认 pytest 临时目录遇到 Windows ACL，改用独立 artifacts 临时目录后通过；
没有修改安装包、系统权限或已有临时目录。

四组分别比较一次性训练与同对象分段训练＋快照＋流式观察，逐位验证：

- Torch/NumPy RNG 与全部实际 replay action/state/reward/next/mask 相同；
- policy、critic、target network、Adam 与 scheduler state 相同；
- 保存快照不会重建 agent、optimizer、replay、SUMO 或插入 reset bridge。

真实 SUMO bounded smoke：`artifacts/p7k/p7k_smoke_v1`。
运行 source commit：`265b05fb42c34905d7a55222825228871c2e0ac4`。
request hash：
`ff7ddcee4ec3ef2c3c5913f5c5338f502b4b50b6135c8998337aef7946399766`。
aggregate SHA256：
`6a2a08eb4014d04a084747fb27c2b5fd97c800d2cb02805580f5779c20fd26d4`。

| 短训练实例 | 实际控制步 | 原 optimizer update 对数 | 实际 minibatch 状态次数 |
|---|---:|---:|---:|
| B0 | 69 | 62 | 496 |
| B1 | 158 | 152 | 1216 |
| B2 | 102 | 95 | 760 |
| B3 | 101 | 94 | 752 |
| 合计 | 430 | 403 | 3224 |

每实例请求 64 步，上界 563；三小节点 16/32/64 均按完整回合边界保存。
B1 的 64 节点确实继续到第二回合，其他多节点可共享同一边界。
另有 12 个独立新进程评价回合全部完成，初始场景 hash 一致，
没有 policy/critic/replay 更新。原 reward 重算最大误差为 0；
实际 ego event/route endpoint 与 native terminal 一致。
碰撞结果保留并正常聚合；`engineering_complete=true`、
`method_effect_gate=null`、`formal_claim=false`、`test_opened=false`。

`--resume` 再校验完整四训练/十二评价 jobs：全部 action=reuse，
没有新仿真；不加载 evaluation-only 权重续训。
失败/中断的训练目录仍保留并拒绝自动恢复。任何恢复实验需单独说明，
不能静默覆盖失败产物。

## 交接与限制

完整 60k 请求由 prepare 冻结，训练和 720 回合评价由用户手动启动。
见 [runbook](../docs/runbooks/p07k_matched_learning_curve.md)。
本轮 smoke 是工程证据，不用于解释方法效果或模型收敛。
真实学习效果需完成全部节点和种子后分析；critic/actor loss 仅作解释。
若出现原 reward 与独立 ego event 不一致、NaN 或错误模型接口，停止该运行，
不要用长训练、换 seed、重标 reward 或改评价协议掩盖问题。

所有旧执行源码及实验产物保留；旧 `E:/WCDT_ACCVP` 未修改。
只有本地原子提交及非 squash 集成，不推送、不创建正式复现实验 tag。
