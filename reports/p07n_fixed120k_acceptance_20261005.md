# P7n 固定 120k 四组匹配实验入口验收

日期：2026-10-05。功能分支：codex/p07n-fixed120k-matched。
实现提交：05f49c9e4162e542b18b043e6c6d6a4901e27e4a。
这是开发实验入口验收，不是 120k 方法效果报告，也不批准 P8 正式实验。

## 范围与冻结边界

根据用户批准的一次有界预算扩展，新增 P7n 请求、配置、统计和执行入口。
B0 无预测、B1 零扩展通道、B2 普通预测、B3 动作条件预测，全部保留训练
种子 0/1/2，从头训练 12 个 DDPG；旧 60k policy 不用于初始化。
普通及动作条件三成员预测器复用，不重新训练预测器。

新完整配置相对 P7k 只改变实验版本、主终点说明、checkpoint 节点和请求
训练预算四个字段。奖励、网络、特征、预测器、continuous jerk、Gym20、
训练 seeds、20 个开发场景以及原 2M 学习率计划保持不变。
四节点为 20k/40k/60k/120k；120k 是唯一主终点，不按开发结果选最好节点。

所有旧 src/tools 执行文件和上游源码保持未修改；新入口直接复用不可变
P7k training/evaluation kernel，既有 kernel 的 p7k 日志标签不是请求身份。
P7m 完整只读诊断及 P7j 核验作为来源证据，不设置正向方法效果 gate。
新 replay 容量校验确保原 1M buffer 不会在 120k 完整回合边界前绕回。
旧项目 E:/WCDT_ACCVP、历史 artifacts 与配置不变。

## 实际执行的测试

```powershell
python -B -m pytest tests/test_fixed_budget_extension.py -q -p no:cacheprovider
python -B -m pytest tests -q -p no:cacheprovider
```

- 新测试：31 passed。
- 完整回归：866 passed；两个既有作者资产 trusted-pickle FutureWarning。
- 四组 synthetic 等价性测试验证四次快照不改变动作、权重、target、
  optimizer、scheduler、replay、Torch/NumPy RNG。
- 120k 停止节点、完整 roster、负结果正常聚合、场景身份、禁止部分续训、
  replay 容量和解析配置冻结等测试通过。
- git diff --check 与 staged diff 审查通过；无权重、原始数据、日志或秘密入 Git。

## 有界真实 SUMO smoke

```powershell
python -B tools/train_fixed_budget_ddpg.py audit --run-id p7n_smoke_v1
```

产物：artifacts/p7n/p7n_smoke_v1。
请求 hash：c331a2f4e01475b3e19a1d5c6dfe3e180d47b869643a9593541b0a4a971a116b。
aggregate SHA256：dadb4f522f3042ea996c9084c173368c10b8abb3af604bb6a269ffd37b3c9dc9。
training_curves SHA256：2a071af0735e2ac769b1b3369e6c98a3c8dd793018bfebc8110556352280898c。
development_curves SHA256：59e6469eb20ef81505f42a653dae8010bb22fa5bcd45b774a83cd785f3dc0b8c。

smoke 沿用独立历史工程配置：单训练 seed，64 请求步，warmup/batch=8，
节点 16/32/64，单场景 200。不是完整 120k 配置或方法性能证据。
原执行代码在完整回合边界保存，实际训练步数如下：

| 组别 | 实际控制步 | 原 optimizer 更新（policy/critic 各自） |
| --- | ---: | ---: |
| B0 baseline | 69 | 62 |
| B1 zero | 158 | 152 |
| B2 ordinary | 102 | 95 |
| B3 conditional | 101 | 94 |

合计 430 控制步，全部 from_scratch；B1/B2/B3 初始 policy 与 critic SHA 相同。
所有组奖励核验最大误差为 0，reset bridge transitions 为 0。
训练和 12 次冻结评价均 complete；12 次评价初始交通 SHA 完全相同，
无策略更新，原终止事件一致。碰撞、自然到达、500 步时间期限均正常保留。
没有根据这些短训练结果选择 seed、checkpoint 或评价条件。

新 training_curves.json 是小型索引，引用原始 episodes.json 和
losses.jsonl.gz，并保存 SHA 与数量；每个 loss component 的行数与真实
minibatch 更新数完全一致，不再复制一份大型内嵌 loss JSON。

随后分别运行同请求 train/evaluate --resume。4 个训练实例与 12 个评价
回合均只 action=reuse，无新训练、无新仿真；aggregate SHA 保持不变。
writer.lock 已正常释放，没有覆盖或删除失败结果。

## 用户交接与限制

完整 P7n 只准备请求，不由开发验收启动：12 × 120k = 144 万请求训练步。
12 实例 × 4 节点 × 20 开发场景 = 960 个评价回合，但只有 20 个不同
交通场景，不能当成 960 个独立样本或独立确认实验。
每个节点按原完整回合边界允许最多 499 步超出。

命令见 docs/runbooks/p07n_fixed120k_matched.md。全部完成前保持源码、
配置和环境冻结，先 train，再 evaluate，最后 aggregate 一次。
--resume 只复用封存完整实例，不恢复 weights-only checkpoint 的训练状态。
中断实例保留并拒绝自动重试；审查后另建恢复请求，不能删除或替换 seed。

不混入 Author50，不启用 MPC/Shield，不修改网络/奖励，不打开测试集。
完整结果回来后比较 B2−B1、B3−B2 和全部安全/任务指标并作明确决策；
不自动延长训练预算。无 Git push 或新 milestone tag。
