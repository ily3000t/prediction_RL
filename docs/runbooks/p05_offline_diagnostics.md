# P5a — 验证集预测参考比较与宽校准区间诊断

## 当前结论

用户校准已完成，普通／条件模型 q=46.1755／48.0513，均使用全部 64 个 calibration episodes。
两组样本内 episode coverage 都是 59/64=92.1875%，但平均区间全宽很大：
横向 y 约 27.71／28.83 m。校准工程完成不等于不确定性估计有实用价值。
目前不能把这些区间当作安全保证，也不能用近 100% 的坐标覆盖率替代区间宽度分析。

既有权重、训练预算、校准参数、数据划分与所有原始报告保持不变。
不需要重新采集、训练或重新校准。本步骤只读地诊断其表现。

## 下一条用户命令

在 `(pytorch) E:\Prediction_RL>` 执行：

```powershell
python tools/diagnose_response_predictors.py run --request artifacts/offline/predictor_v1_diag/request.json --confirm-request-hash 2a651b355533f1499776f6b060ee7dc87fa2c2604d04dfb446b2e7da1f1377d4
```

独立、无嵌套的配置：`configs/development/p05_offline_diagnostics_v1.json`。
它明确标为 development/validation diagnostic，不是正式独立 test 实验。
request 还冻结模型／校准／数据／源码／环境哈希，必须保持工作树干净。
没有 optimizer、SUMO 或 DDPG；不会打开 predictor test。

## 诊断内容及指标口径

### 1. Validation：64 episodes、177 roots（包括 1 个空监督 root）

同样的选中 actor、候选和未来 mask，比较：

- `lane_chain_constant_velocity`：沿固定地图的已知唯一 lane chain 保持 root speed，
  acceleration=0。只读取 root 交通状态和地图，绝不读取未来标签。
  位移以真实 root XY 锚定；地图长度与折线弧长线性对应；越出道路末端继续末段切线，
  但真实车辆消失后的标签仍然 masked，不凭空给标签补点。
- `ordinary`：冻结的普通三成员 ensemble 均值。
- `conditional`：冻结的动作条件三成员 ensemble 均值。

报告 scaled MSE、ADE、x/y/speed/acceleration MAE，以及两个不可混淆的终点指标：
`last_observed_displacement_m` 是每个 actor 最后可观测时间；
`fde_5s_m` 只包含真实观测到第 25 步的 actor branches，单独报告分母。
所有指标仅适用于 observed nonterminal cells，不解释碰撞后的缺失未来。

有效 cell 先在每个候选内汇总，再对有效候选取均值，得到 root 指标；
同时报告 root-equal 和先 root 后 episode-equal 均值。
额外输出同 root／同 episode 的 conditional−ordinary、conditional−CV 配对差值；
误差指标负值表示条件模型较低。这里只作描述性诊断，不报告显著性。
多个 actor、时间步、候选或同一 episode 的 roots 不能冒充独立样本。

### 2. 同 root 候选响应差异

五个候选的全部十对均使用，两分支都有监督的 actor-time mask 取交集：
比较预测响应差 `(pred_a-pred_b)` 与真实响应差 `(truth_a-truth_b)`，
按既定 [10,3,5,2] 缩放计算均方误差，然后先 pair、再 root／episode 汇总。
普通预测与 CV 都不接收候选，预测响应差为零，因此这一指标应完全相同。
条件预测能否降低该误差是明确的诊断问题；不会删除真实无响应的 pairs。

**它不是候选动作安全性排序**，不评价自车机会、合流成功或 Reward。
P5b 仍须明确共享自车 rollout、执行约束、几何量及终止分支处理；
不能用真实 future ego 作为在线输入来“完成”排序。

### 3. Validation + Calibration：冻结区间归因

使用已经保存的 q，不重新拟合。每个 root 输出最坏 standardized residual 对应的：
actor ID、候选、通道、预测时刻、truth、ensemble mean/std 和实际 scale。
汇总各通道 floor 生效比例、宽度、覆盖，以及每个 episode 的最坏通道／时刻。

calibration 是拟合样本内结果；validation 已参与 checkpoint selection，
也不是独立 test。两者都标注 `independent_test_evidence=false`。
这些信息用于分清长尾预测误差、分歧偏小和 episode 最大值聚合的影响，
不是据此立即换阈值、删 actor、删 seed 或改校准规则。

## 产物与下一步

所有结果保存在 `artifacts/offline/predictor_v1_diag/`：

- `report.json`：两划分汇总、三方法误差、配对差值及区间归因。
- `validation_<method>_roots.json`：逐 root 预测指标。
- `<split>_<ordinary|conditional>_attribution.json`：逐 root 最坏误差位置。
- `started.json`：真实执行记录。请求及 preparation 已存在，不必重复 prepare。

负结果也会正常 `status=complete`，不会假装方法 gate 通过。
代码错误或数据不一致则停止并保留 failure_reason，不自动重试／覆盖。
诊断完成后先复盘结果，再讨论 P5b 候选几何／排序协议。
P5 全部验收前，不启动测试采集或 DDPG。

已执行的工程检查（不用重跑）：355 项单元测试；
`p5a_interface_v1_01` 九个既有开发 roots 上的真实权重／CV／指标 smoke；
正式诊断仅 prepare，尚未 run。
