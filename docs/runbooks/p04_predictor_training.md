# P4h — 已完成数据采集后的预测器训练

## 当前状态与边界

384 个 train/validation/calibration episodes 已采集并逐文件校验。
普通预测与动作条件预测的完整 CPU 训练入口已通过单元测试及开发数据两轮恢复审计。
正式六成员训练尚未启动；本命令不训练 DDPG、不运行 SUMO、不拟合 calibration，
也不会读取封存的 test trajectories。旧项目及原始奖励/连续 jerk 接口没有修改。

唯一实验参数来源是独立配置：
`configs/formal/p04_response_dataset_v1.json`。
请求保存其完整解析值和数据/代码/环境哈希；不要在训练中修改该配置或请求。

## 用户下一步：启动正式预测器训练

在 `pytorch` 环境、`E:\Prediction_RL` 目录执行：

```powershell
python tools/train_response_predictors.py run --request artifacts/training/predictor_v1/request.json --confirm-request-hash 5c8039f45c798e7377f21858fc98da47b3aeffaaf9eee93ca9fc5846520fada7
```

顺序训练 ordinary 的 4101、4102、4103，再训练 conditional 的同三个初始化种子。
每个成员从头初始化，不加载开发 smoke 权重；对应两组采用相同初始参数和根状态打乱顺序。
没有正式 PPO 训练：当前方案后续控制器仍是原始连续 DDPG。

训练设置：CPU float32、Torch 单线程、batch=16 个完整 root families、Adam lr=0.001、
最多 100 轮、每轮验证、patience=15、min_delta=0.0001。
有效训练根状态 710 个，每轮 45 次更新；有效验证根状态 176 个。
每个根状态的五个候选一起进入 batch，不把它们当作五个独立场景。
不使用校准或测试集选轮数。

最佳 checkpoint 保存规则为严格更低的有限 validation loss，完全相等保留较早一轮。
`min_delta` 只决定耐心计数是否清零；小于 min_delta 的真实改善仍会保存最佳权重。
训练日志的 train 是本轮更新前各 batch 的损失汇总，validation 是本轮更新后的评估损失，
两条曲线并非在同一组固定权重上测量，不能仅凭两者差距判断过拟合。

## 恢复与不可变结果

正常停止或程序报错后，先保留完整报错。明确检查后可用原命令追加 `--resume`：

```powershell
python tools/train_response_predictors.py run --request artifacts/training/predictor_v1/request.json --confirm-request-hash 5c8039f45c798e7377f21858fc98da47b3aeffaaf9eee93ca9fc5846520fada7 --resume
```

完整成员通过文件清单/哈希校验后复用；未完成成员从最后一个完整 epoch checkpoint 恢复。
恢复包含模型、Adam moments/step、采样 RNG、CPU RNG、选模历史和最佳权重。
若中断发生在尚未保存的 epoch，该轮重新计算，不换 seed。

若出现 writer.lock、缺失 checkpoint receipt、部分 selected.pt/history.json 发布、
哈希不符或未知文件状态，程序会停止，不自动删锁、不覆盖或修补报告。请先诊断，
不要通过删除产物或重新 prepare 绕过。当前不支持跨代码/环境版本的自动续训。

## 输出位置及下一阶段

全部产物位于 `artifacts/training/predictor_v1/`，Git 忽略：

- `request.json`、`preparation.json`：冻结的训练请求与准备记录。
- `ordinary_4101/` 等六目录：`e0001.pt` 等逐轮 checkpoint 与 JSON receipt、
  `selected.pt`、`history.json`、`complete.json`。
- `ordinary_ensemble.pt`、`conditional_ensemble.pt`：三个独立选模成员的严格组合；
  明确标记 `calibrated=false`、`test_evaluated=false`，尚不批准进入控制实验。
- `invocations/<id>/report.json`：实际 commit、命令、状态、失败原因及成员汇总。

完成后先审查训练/验证曲线、最佳轮次、普通与条件预测的误差和候选敏感性，
再实现/验收独立校准与 P5 离线排序指标。只有预测器和校准冻结后才释放 test 采集。
不能把训练结束直接解释为 ACCVP 有效，也不能跳过 P5 启动昂贵 DDPG 比较。

## 本阶段工程审计（已执行，不必重复）

```powershell
python tools/train_response_predictors.py review --request artifacts/data/response_v1/request.json --collection-report artifacts/data/response_v1/invocations/701dcad2a0c5/report.json --run-id response_v1_review
python tools/audit_epoch_training.py --run-id p4_epoch_v1_01
python tools/train_response_predictors.py prepare --dataset-review artifacts/data_reviews/response_v1_review/review.json --training-audit artifacts/p4/p4_epoch_v1_01/report.json --run-id predictor_v1
```

这些已存在 run IDs 不可重复创建。collection 的旧报告保持原样，
其 `formal_training_ready=false` 是采集阶段的历史状态；独立 review 与当前训练请求
记录下一阶段许可，不回写旧报告。新增代码后，旧 collector 的当前代码哈希保护仍然存在；
不要重新运行已完成的 collector。训练消费者通过 collection 实际 Git commit 的历史 blobs
验证来源，不执行历史代码，也不要求旧采集代码与新训练代码相同。
