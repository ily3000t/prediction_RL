# P4i — 完成预测器训练后进行独立校准

## 已完成／尚未完成

六个普通／条件预测成员全部完成 100 轮；最佳 checkpoint、逐轮历史、
成员文件清单和 ensemble 权重已校验。无需重跑采集或训练命令。
本阶段新增严格加载器与校准入口，已通过 336 项测试和九个开发根状态推理审计。
正式校准、P5 离线效果评估和 DDPG 均未启动，测试集仍封存。

规则来自既有完整配置 `configs/formal/p04_response_dataset_v1.json` 中的
`calibration` 段，没有修改训练轮数、数据划分、成员种子或任何既有配置。

## 用户下一步命令

在 `(pytorch) E:\Prediction_RL>` 执行一次：

```powershell
python tools/calibrate_response_predictors.py run --request artifacts/calibration/predictor_v1/request.json --confirm-request-hash 1321568a60e0f4acf3479026981b140b4a7d919c33c56bcca5a14643ffb3a7bb
```

这不是再次训练：无 optimizer、无反向传播、无 SUMO、无 DDPG。
只读取固定的两个三成员 ensemble 和 calibration 的 64 episodes／169 roots，
计算轨迹残差尺度与校准分位数，保存不可变记录。训练／验证／测试集不参与拟合 q。
不需要再次运行 prepare；请求已准备好。

## 冻结的计算与解释

1. 三成员输出使用 float32；统计计算在 float64 中重算均值及总体标准差。
2. 各坐标的尺度为 `max(ensemble population std, 0.1 × [10,3,5,2])`，
   四个通道依次为 root-relative x、y、speed、acceleration。
3. 每个有效坐标计算 `abs(truth - mean) / scale`；终止／缺失轨迹按已有 mask 排除。
4. 一个 episode 内所有根状态、候选、邻车、时间和通道的最大值只贡献一个分数。
5. 取第 `ceil((n+1) × 0.90)` 个有序分数，n 为有监督的 calibration episodes。
   若 64 个全部有效，取第 59 个，不插值；若排序位置超过 n，则区间无界，不能夹到最大观测值。
6. 普通与条件模型分别计算 q，但使用同一数据和规则。区间为 `mean ± q × scale`。

报告明确保留空 episode／root、有效坐标数和每个 episode 的分数。
`fitted_sample_coverage` 与 `coordinate_coverage` 都是校准样本内诊断，
不是独立测试结果。无界时 JSON 使用 `q: null, unbounded: true`，不会写非法 Infinity。
区间过宽也是有效负结果，程序不会因此换 seed、阈值或删去困难样本。

这些是观测到的非终止轨迹坐标区间，不是碰撞概率、可信安全保证、
任意连续 jerk 的响应保证或独立 Risk Shield。有限的三成员分歧也不是天然可信区间。

## 输出与故障处理

输出目录 `artifacts/calibration/predictor_v1/`：

- `request.json` / `preparation.json`：已存在；绑定数据、模型、源码及环境。
- `calibration.json`：运行后冻结的两组模型校准参数和模型哈希。
- `report.json`：运行后完整分数、覆盖／宽度诊断、实际 commit、命令和状态。
- `started.json`：实际启动记录。失败将保留 `report.json` 的 failure_reason。

不提供覆盖式重跑或自动删锁／修补。如果重复执行提示 already started，
先检查已有 report；不要删除失败产物或换 run ID 来隐藏问题。
新代码加入后旧训练请求的当前源码保护可能拒绝重新执行，这是预期行为；
完整模型由新消费者按训练时的 Git blobs 与哈希验收，旧训练保护不放宽。

## 后续顺序

完成后先复盘 q、区间宽度、空样本及 censoring，再继续 P5：
冻结简单预测参考与共享自车 rollout 下的同 root 候选几何／排序定义，
实现验证集离线误差与排序检查。只有相关模型、校准与评估定义全部冻结后，
再单独审核是否释放 predictor test 采集。当前命令不会解封 test。
P5 未通过之前不进入 DDPG 特征接入与昂贵闭环训练。

## 已执行的工程准备（不用重复）

```powershell
python tools/calibrate_response_predictors.py review --training-request artifacts/training/predictor_v1/request.json --training-report artifacts/training/predictor_v1/invocations/648e96637f5e/report.json --run-id predictor_v1_review
python tools/audit_calibration.py --training-review artifacts/training_reviews/predictor_v1_review/review.json --run-id p4_calibration_v1_01
python tools/calibrate_response_predictors.py prepare --training-review artifacts/training_reviews/predictor_v1_review/review.json --calibration-audit artifacts/p4/p4_calibration_v1_01/report.json --run-id predictor_v1
```
