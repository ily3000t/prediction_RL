# P7m replay 信息利用诊断工程验收

日期：2026-10-05。功能分支：codex/p07m-replay-information-use。
实现提交：8bb5948。此验收不批准正式实验或 120k 训练。

## 实现范围

新增独立只读诊断，覆盖冻结 60k B2/B3 的全部三个训练种子。
原 20 维状态、已保存时间特征、连续 jerk、奖励、预测器和已有结果不变。
不重新训练、不运行 SUMO、不修改上游或旧 ACCVP 项目。
源码与数据核验沿用不可变历史来源，不放宽旧 pipeline 的 source 检查。

三扰动：全部扩展通道清零、预测数值清零但保留 metadata、候选均值广播。
采样不读取 reward/终止结果；uniform 主样本与四个分层集合分别报告。
不把候选敏感性当作动作条件邻车响应的独立收益，也不把不确定的失败归因
说成已经解释。模型加载使用原 ALL preset 和严格完整 policy/critic 权重。

## 实际运行的测试

```powershell
python -B -m pytest tests/test_replay_information.py -q -p no:cacheprovider
python -B -m pytest tests -q -p no:cacheprovider
```

- 新测试：27 passed。
- 完整回归：835 passed；两个既有作者资产 trusted-pickle FutureWarning。
- git diff --check、staged diff 检查通过；提交中无权重、原始数据、日志或秘密。

## 六模型有界真实权重 smoke

路径：artifacts/p7m/p7m_audit_v1。
请求 hash：1e5788a5f4404ec788c19a71516c7198bbabe2c0d76a22bceec9e8218c710ffc。
aggregate SHA256：55976c8eaa9d066f564d022bc2d7a2ab82e706c88da2fd40d6354785f6540856。

每模型只读取 replay 前 128 条，共 768 条；实际去重查询状态数如下：

| 冻结模型 | 状态数 |
| --- | ---: |
| B2 ordinary 0 | 9 |
| B2 ordinary 1 | 10 |
| B2 ordinary 2 | 11 |
| B3 conditional 0 | 8 |
| B3 conditional 1 | 11 |
| B3 conditional 2 | 11 |

合计 60 个不同状态。全部严格加载成功，原网络逐条/批量动作在冻结容差内一致；
policy/critic、target、optimizer、scheduler、replay、TimeFeature 和查询 RNG
未改变。状态里已有时间坐标，不调用 TimeFeature.eval 再追加时间。
没有仿真回合。完整 SHA/inventory 校验通过；追加 --resume 后只验证已封存
产物，aggregate SHA 不变，没有重新查询或覆盖报告。

## 限制与下一步

smoke 是实现验收，不是方法效果结论。完整 replay 诊断尚未执行，保留用户手动启动。
其最多扫描约 360k 状态，查询最多 6144 个不同状态，不新增交通场景。
0.1 m/s³ 是描述参考而非方法成功门槛。没有失败 seed 剔除或 checkpoint 选择。

目前 replay 缺少完整邻车未来真值和对应评价输入，无法仅凭它判断新失败由
预测误差还是策略学习波动造成；此结论明确保留为不可识别。
完整结果出来后再作一次明确的预算/模型决策，不自动延长训练。
运行命令与产物解释见 docs/runbooks/p07m_replay_information.md。
