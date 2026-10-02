# P7f 有界评价循环 × 预热诊断

项目根目录E:/Prediction_RL，激活pytorch；无需训练任何模型。
配置：`configs/development/p07f_driver_warmup_v1.json`，独立可读，无extends。
入口：`tools/diagnose_driver_warmup.py`。

## 冻结工作量

| 条件 | Gym、20秒 | Gym、50秒 | 作者循环、20秒 | 作者循环、50秒 |
| --- | --- | --- | --- | --- |
| 作者预训练DDPG | 复用3 | 新增3 | 新增3 | 复用3 |
| 本项目B0、三训练种子 | 复用9 | 新增9 | 新增9 | 复用9 |
| 本项目B3、三训练种子 | 复用9 | 新增9 | 新增9 | 复用9 |

场景200/210/219沿用P7b结果前的首/中/末索引选择。84格不是84个独立场景。
作者只有一份权重，训练预算不匹配，不作正式排名或显著性结论。

## 工程验收（开发者有界执行）

```powershell
python tools/diagnose_driver_warmup.py audit --run-id p7f_audit_v1
```

只在场景200用作者模型及本项目B0/B3种子0，新增6回合、引用6回合。
历史P7/P7d/P7e/P7b来源及模型链均需通过，不修改旧源码以兼容请求。

## 准备用户运行请求（不启动SUMO）

```powershell
python tools/diagnose_driver_warmup.py prepare --run-id p7f_diag_v1
```

prepare需要当前工程验收完成及干净工作树，冻结42个新任务、42份旧证据、
完整解析协议、来源哈希、模型、源码与环境；不会执行42回合。
已经准备时不要重复prepare或使用同run ID覆盖产物。

## 用户运行42回合

```powershell
python tools/diagnose_driver_warmup.py run --request artifacts/p7f/p7f_diag_v1/request.json --confirm-request-hash <prepare打印的哈希>
```

两进程，每scene新进程首回合；CPU一Torch线程，等待预测完成再推进SUMO。
自动输出`artifacts/p7f/p7f_diag_v1/aggregate.json`。
重复启动仅加`--resume`复用已验证完整回合，不重试/删除不完整或失败目录。

## 如何分析

- matrix：三模型、四条件，保留全部训练种子与成功回合分母。
- same_warmup_driver_comparisons：固定预热比较循环，逐场景共同交通、实际jerk、
  命令速度、可得的observation和TimeFeature；不计算跨driver奖励差。
- same_driver_warmup_comparisons：固定循环比较预热，原口径事件和同评分奖励变化。
- 配对轨迹只按实际索引比较共同前缀，不插值、不把状态分歧后的动作差当独立因果证据。
- P7d作者/B3旧轨迹未记录observation，明确null；TimeFeature是原单次查询序列重建。
  B0作者循环和Gym旧记录有实际输入；记录/重建来源分别保留。
- Gym命令速度是原helper重建，作者循环是原controller返回；不是线级命令拦截。
- 碰撞仍是不同原生判定；500/501调用与评分差异保留，不能直接称统一ego碰撞率。
- time_limit是仿真期限，不是wall-clock超时。负结果正常complete，不放宽gate。
- 只检查协议敏感性，不选择最好协议、不挑训练seed；训练/模型扩展/P8须另行确认。

## 本机工程验收状态（2026-10-02）

源码提交62d1798；602项测试通过（新增33项）。上述工程audit已完成：
6个新回合、6个历史引用、1473次实际控制调用，engineering_complete=true。
三模型的相同预热初始共同交通均一致；后续轨迹逐位差异仍保留，不将数值
舍入差异直接当语义错误，也不以该单场景作方法结论。
不要重复audit。完整42回合现已由用户运行完成，见下方结果复盘。
验收细节见`reports/p07f_driver_warmup_acceptance_20261002.md`。

完整请求已在干净源码上准备：`artifacts/p7f/p7f_diag_v1/request.json`。
本机这份请求已运行完成，以下保留实际启动命令供溯源，不需要重复执行：

```powershell
python tools/diagnose_driver_warmup.py run --request artifacts/p7f/p7f_diag_v1/request.json --confirm-request-hash a87464d61fcf154bbc81444267ff8bf10e13ab80c09a077d72ccd2e45d76b917
```

该哈希只对应本机这份请求。42个新回合已完成、0 reuse，加上42格历史引用后
84格复算通过；aggregate SHA256为
`9a1a37d30ec0c2181999570f9766655bc5a28179f968215ec56a57cf2f302d9e`。

## 完整结果与下一步边界（2026-10-02）

详见[完整复盘](../../reports/p07f_results_review_20261002.md)。发现9对近一致轨迹
原终止标签相反。SUMO 1.22.0碰撞移除也发布ARRIVED，上游作者循环先检查arrival，
导致native到达不能直接解释为真实成功。历史几何回查中B0种子2的20回合及
B3种子0的11回合明显在终点前结束。旧报告不覆盖、不自动改标签，不换seed或阈值。

当前先暂停按原成功率/成功条件奖励作方法结论，不重训、不扩大模型、不进入P8。
建议新增只读原始终止事件sidecar，在清理/额外step前捕获arrived IDs、collision
参与者和ego存在状态，验证冻结模型轨迹不变后重评。此阶段尚未实现或准备请求；
无需重跑P7f，不存在可交付的新CLI命令，需用户另行确认后开发。
