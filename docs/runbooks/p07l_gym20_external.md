# P7l：60k 模型与作者控制器的 Gym20 比较

在 `E:\Prediction_RL`、`pytorch` 环境运行。先完成代码提交与有界适配验收。
不训练任何模型，原 Author50 结果独立保留，不混合回报或成功率。

## 评价内容

- 新跑作者 DDPG、ST、RL+MPC safety、RL+MPC switching：各 20 场景，共 80 回合。
- 复用 P7k 的 60k B0/B1/B2/B3、训练种子 0/1/2、场景 200–219：共 240 回合。
- 作者模型只有一个冻结训练来源，不能因与三个项目种子配对而声称有三个作者模型。
- Gym 20 秒预热、0.2 秒步长、500 控制步；初始场景逐 seed 哈希相等。
- 连续 jerk DDPG 与原速度规划器是不同控制器；不重新投影规划器速度。
- 原 Gym Slotted Jerk 计分、非法动作惩罚 0，运行步使用实测 jerk，终止奖励不依赖 projected jerk。
  适配器拒绝其他奖励或非零动作惩罚。ST 的 jerk-action saturation 为 null，而不是捏造零。
- 原 Gym reset/step/清理及真实事件分析保持不变。不是原作者50秒循环的重新命名。
- 平均耗时包含失败，不能作为效率优势；配对完成时间只使用双方自然到达的场景。
- 结果是开发集描述性比较，非匹配训练预算、非正式显著性或因果增量证明。

## 有界工程验收（最多八回合，非完整实验）

```powershell
python -B tools/compare_gym20_external.py audit --run-id p7l_audit_v1
```

仅场景 200，四控制器各一次无 observer 参考和一次带 observer 重放。
需要命令/实际轨迹/原回报/查询次数/接管序列完全相同；不把成功当作必要条件。

## 用户启动完整比较

```powershell
python -B tools/compare_gym20_external.py prepare --config configs/development/p07l_gym20_external_v1.json --run-id p7l_gym20_v1 --adapter-audit artifacts/p7l/p7l_audit_v1/aggregate.json
```

记录打印的 `confirm_request_hash`，将下面占位符替换为该值：

```powershell
python -B tools/compare_gym20_external.py run --request artifacts/p7l/p7l_gym20_v1/request.json --confirm-request-hash <打印的hash> --resume
```

`run` 自动聚合；输出 `artifacts/p7l/p7l_gym20_v1/aggregate.json`。
`--resume` 只复用完整且哈希一致的整个回合；不恢复或自动删除失败目录。
子进程失败会给出具体日志路径，先检查该文件，不重写失败结果。
如果所有回合完成但聚合中断，可以单独运行：

```powershell
python -B tools/compare_gym20_external.py aggregate --request artifacts/p7l/p7l_gym20_v1/request.json --confirm-request-hash <打印的hash>
```

完成后分析全部三个种子、四种项目方法与四个作者控制器，再决定一次短信息利用检查。
本阶段没有实现或启动 120k 扩展，也不自动批准后续训练。
