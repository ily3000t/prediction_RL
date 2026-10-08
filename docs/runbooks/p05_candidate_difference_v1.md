# 候选响应差分损失：运行与复盘

## 位置与当前边界

请在 `pytorch` 环境下进入新工作树，不是在旧 WCDT_ACCVP 工作树运行：

```powershell
cd E:\Prediction_RL_worktrees\response-difference-v1
git branch --show-current
```

应为 `codex/p05-response-difference`。原 `E:\Prediction_RL` 只作为只读数据和
参考模型来源，不在那里启动新版本，不复制或覆盖其旧 artifacts。

独立配置：`configs/development/p05_candidate_difference_v1.json`。
首版 `lambda=0.2`，对应共同有效支持上的候选间误差权重增加 50%。这不是搜索所得
最优值；正式运行前请确认这个开发设定。旧配置不修改，新版本不在旧请求中续训。

只从头训练新 conditional ensemble 的 4101、4102、4103 三个成员。
架构、初始化、数据、root 打乱规则、CPU 单线程、batch=16、Adam 和最多 100 轮
保持原方案；patience=15、min_delta=0.0001，选模依旧使用轨迹 MSE。
训练的 `train_pre_update_loss` 是联合目标，另外保存两个损失分量和配对覆盖率。
不能把联合训练 loss 与轨迹验证 loss 当作相同指标解释。

旧 ordinary 与旧 conditional ensemble 直接复用为离线参考。不会重训作者 DDPG、
B0/B1/B2/B3，不增加泛化或作者 50 秒协议，不修改奖励和在线 149 通道的生成逻辑。
当前尚未为新 checkpoint 开放校准/闭环 DDPG 消费；旧消费者会明确拒绝新版本。

## 1. 只准备请求，不训练

确认 Git 工作树干净后：

```powershell
python -B tools/train_candidate_difference.py prepare --config configs/development/p05_candidate_difference_v1.json --reference-root E:/Prediction_RL --run-id delta_v1
```

这个步骤只审查已有训练来源、读取冻结 train/validation 数据、统计共同 mask
覆盖率并生成请求，不计算模型梯度。所有输出在新工作树：

- `artifacts/prediction_difference/delta_v1/request.json`
- `artifacts/prediction_difference/delta_v1/preparation.json`

本轮若已经准备完成，不要重复执行相同 run ID。请求冻结当前代码、配置、环境、
原模型哈希和数据 fingerprint；代码变化后不得继续使用旧请求绕过保护。

## 2. 用户确认后训练三个预测成员

先检查 preparation 的配对覆盖和配置。确认只修改了损失后执行：

```powershell
$deltaRequest = 'artifacts/prediction_difference/delta_v1/request.json'
$deltaHash = (Get-Content artifacts/prediction_difference/delta_v1/preparation.json -Raw | ConvertFrom-Json).request_hash
python -B tools/train_candidate_difference.py run --request $deltaRequest --confirm-request-hash $deltaHash
```

命令会打印真实 `difference_training_report=...` 路径，请保留。
完整训练由你手动启动，本轮工程验收不会运行此命令。

需要恢复时先检查失败原因，再使用原请求追加 `--resume`：

```powershell
python -B tools/train_candidate_difference.py run --request $deltaRequest --confirm-request-hash $deltaHash --resume
```

恢复包括模型、Adam 状态、root 采样 RNG、CPU RNG、选模和最佳权重。
完成成员只在 receipt/inventory 校验后复用。发生 writer.lock、哈希不符或部分发布时
保留现场，不删锁、不覆盖结果、不换 seed。

## 3. 配对离线验证

将下面变量设为训练实际打印的报告路径，不要使用旧 predictor_v1 报告：

```powershell
$deltaReport = '填写 difference_training_report 打印的路径'
python -B tools/train_candidate_difference.py diagnose --request $deltaRequest --confirm-request-hash $deltaHash --training-report $deltaReport
```

结果：`artifacts/prediction_difference/delta_v1/offline_validation.json`。
比较旧普通预测、旧条件预测与新差分条件预测，使用相同 validation roots/episodes，
保存 root-equal、episode-equal 和逐 root/episode 配对差值。没有新 SUMO 回合。
只读取 validation，不读 calibration/test 目标；负结果正常完成，不设置“必须赢”门槛。

优先检查：差分误差、ADE/绝对误差的权衡、各 episode 的一致性，以及共同有效
mask 覆盖。差分训练的零配对 root 仍进入监督 root 分母，而既有离线差分诊断只对
配对有效的 root 平均；两者要结合覆盖率解释，不能直接当作完全相同的数值。

## 4. 停止并审查，不自动进入 DDPG

本阶段训练完成不等于新机制有效。差分损失强调已有标签中的候选差异，不能自动
证明碰撞/到达后果或真实世界因果效应。现有训练/验证集也不是全新独立测试集。

先审查离线增量与代价，再决定是否给新预测器绑定同一在线特征生成器，并批准
与原 B3 相同预算、全部三个训练种子的从头 DDPG 比较。预测器改变 observation
映射，旧 B3 策略不能当作正式适配后的策略结果。校准量也不能从旧模型静默复用。

## 工程测试

使用一个不存在的 basetemp，避免 pytest 清理其他已有测试目录：

```powershell
New-Item -ItemType Directory -Path artifacts/tests -Force | Out-Null
python -B -m pytest tests/test_candidate_difference.py tests/test_candidate_difference_cli.py tests/test_epoch_training.py tests/test_masked_training.py -q -p no:cacheprovider --basetemp artifacts/tests/manual_delta_01
```

测试只使用小型合成数据/几何文件，覆盖 mask、归约、lambda=0 状态等价、三成员、
完整恢复、来源只读、严格导出与旧模型兼容性，不是正式训练或方法效果实验。
