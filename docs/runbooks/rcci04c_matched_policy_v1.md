# RCCI04c：手动运行匹配实验

入口已通过有界检查，见 `reports/rcci04c_entry_acceptance_20261018.md`。
以下是实际已实现命令。正式训练/长评价尚未启动，由用户逐步运行。

## 配置与成本

- 正式配置：`configs/development/rcci04c_matched_policy_v1.json`，独立完整，无extends。
- 检查配置：`configs/development/rcci04c_entry_smoke_v1.json`，**不要拿它做正式训练**。
- 新训练仅RCCI ordinary/conditional × seeds0/1/2：6实例、每实例120k请求步，共720k。
- 原Raw ordinary/conditional的6模型及四节点只读复用，不训练预测器、不warm-start。
- 四组×3 seeds×4节点×20场景=960回合，含480个Raw当前环境历史回归和480个RCCI评价。
  只有20个场景，不是独立test/generalization；主终点120k，不选中间最佳checkpoint。
- CPU、torch_threads=1，训练串行，评价最多2worker；不改奖励/连续jerk/持续交通reset。
  训练使用batch=100、replay-start=5000，其余原DDPG preset不变。

## 已准备的请求

本轮已准备 `artifacts/rcci04c/matched_v1/request.json`，不要重复prepare同目录。
请求hash：`8dfa87d77539dd7e6167127aa8a0299b8cd1a680bc62b979f58fec6f0b076b20`。
preparation应为`prepared_not_trained`、`full_training_started=false`。

使用pytorch环境，进入工作树并设置PowerShell任务变量：

```powershell
Set-Location E:\Prediction_RL_RCCI
$rcciRequest = 'artifacts/rcci04c/matched_v1/request.json'
$rcciHash = '8dfa87d77539dd7e6167127aa8a0299b8cd1a680bc62b979f58fec6f0b076b20'
```

### 1. 只启动六个新RCCI训练

```powershell
python -B tools/train_rcci_matched_ddpg.py run --request $rcciRequest --confirm-request-hash $rcciHash --stage train --resume
```

待该阶段complete、`training_index.json`和`training_index_receipt.json`存在且校验通过，
再执行评价。每job保存20k/40k/60k/120k完整episode边界snapshot、实际步数、完整receipt。
请求预算不自动加长；原Raw模型不会进入train roster。

### 2. 运行四组当前环境评价

```powershell
python -B tools/train_rcci_matched_ddpg.py run --request $rcciRequest --confirm-request-hash $rcciHash --stage evaluate --resume
```

Raw使用旧权重并逐项核对对应旧记录；RCCI用本次新权重。全部Gym20新进程首回合，
seeds200–219，无exploration。独立阻塞计算，CPU变慢不改变observation。

### 3. 全部完成后聚合

```powershell
python -B tools/train_rcci_matched_ddpg.py aggregate --request $rcciRequest --confirm-request-hash $rcciHash
```

结果：

- `artifacts/rcci04c/matched_v1/aggregate.json`：四组、各seed/节点、配对差值与交互差值。
- `development_curves.json`：冻结策略开发评价曲线；不是独立test或收敛保证。
- `training_curves.json`：训练episode/loss索引，沿用兼容schema；不是验证奖励。
- `train/<job>/interface_summary.json`及`n*.interface.json`：访问分布下的prefix/失效/latency。
- `train/<job>/training_logs/`：TensorBoard原训练奖励与loss；`episodes.json`/`losses.jsonl.gz`
  保留原训练数据。`interface_events.jsonl.gz`保存逐观测描述性诊断。
- `evaluate/<job>/episode.json`、`states.json`、`actions.json`、`raw_events.json`：逐场景证据。

负方法效果会正常聚合，method_effect_gate=null；不会删seed或“修成通过”。

## 进度、resume与限制

主终端每job打印开始，详细训练输出在`invocations/<ID>/<job>.log`。另开终端查看：

```powershell
$rcciLog = Get-ChildItem artifacts/rcci04c/matched_v1/invocations -Recurse -Filter '*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
Get-Content -LiteralPath $rcciLog -Tail 20 -Wait -Encoding UTF8
```

`--resume`仅跳过sealed complete整job，**不能恢复半个模型的优化器/replay**。
若中断/失败，保留目录、日志和failure.json供检查，不删除目录重试、换seed或拿
weights-only snapshot冒充断点续训。先完成/恢复训练阶段的整job调度，再进入评价。
aggregate一次写入，已完成结果不可覆盖。

正式运行要求工作树干净、代码/配置/依赖/原配置字节和模型/SUMO资产不变，baseline
仍为只读干净`75051acc`。请求不要求后续文档commit SHA相同，但实际代码等必须精确相同。
不要在实验运行中修改配置、安装/升级包、重编译binary或切换baseline版本。

如将来确需新请求，必须用新短run ID且先审查，不覆盖matched_v1：

```powershell
python -B tools/train_rcci_matched_ddpg.py prepare --baseline-root E:/Prediction_RL --run-id matched_02 --entry-smoke-request artifacts/rcci04c/entry_v2/request.json
```

使用那次实际打印的新hash。不要复制旧hash或重跑smoke来挑结果。入口检查不需要重做，
除非代码/配置/依赖等改变；改变后旧smoke不能冒充新实现的验收。
