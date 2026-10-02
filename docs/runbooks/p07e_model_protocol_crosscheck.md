# P7e 冻结模型 × 原评价协议

项目根目录E:/Prediction_RL，已激活pytorch。无训练、无checkpoint挑选。

## 矩阵及工作量

| 模型 | P7 Gym、20秒预热 | P7d作者循环、50秒预热 |
| --- | --- | --- |
| author_pretrained_ddpg | 新增20回合 | 引用已有20回合 |
| project_20k_b0_ddpg | 引用已有60回合 | 新增60回合 |
| project_20k_b3_conditional_ddpg | 引用已有60回合 | 引用已有60回合 |

固定场景200–219，本项目模型保留训练种子0/1/2。新增80回合，历史200回合，
总表280单元格；不是280个独立场景，也不是作者三次独立训练。
作者policy/q哈希固定，三组P7最终checkpoint及预测器哈希固定。
旧P7b的6/9是3场景×3种子；完整P7的B0为40/60。

配置独立可读，无extends：
`configs/development/p07e_model_protocol_crosscheck_v1.json`。
入口：`tools/crosscheck_model_protocols.py`。

## 工程验收

```powershell
python tools/crosscheck_model_protocols.py audit --run-id p7e_audit_v1
```

仅场景200，作者Gym一回合＋本项目B0作者循环三回合，上限2003次实际控制调用。
验收产物：`artifacts/p7e/p7e_audit_v1/aggregate.json`。
已有完整验收时不要重复audit；接入负结果不阻止工程通过。

作者模型在Gym中直接复用原GreedyAgent＋TimeFeature和完整作者模块，不将权重
部分移植到本项目网络。只由Gym env.step执行动作；无do_control重复执行。
B0作者循环仅使用Gym观测codec，不调用Gym reset/step，保留原作者run_episode。
每次实际动作与同观测、同时间特征的原网络forward精确一致；原时间推进和
终止重置单独检查。policy/q冻结，replay/optimizer不更新，未开启探索噪声。

## 准备完整请求（不启动SUMO）

```powershell
python tools/crosscheck_model_protocols.py prepare --run-id p7e_cross_v1
```

它核验P7全部240回合、P7d全部140回合及旧工程验收链；只将交叉表需要的
200回合绑定进新请求。允许新增独立审计文件，但拒绝任何旧执行文件改动。
输出request路径和confirm_request_hash，保存完整解析协议、版本、环境、
源码/模型/历史episode与receipt哈希。

## 用户运行80回合

将下面占位哈希替换为prepare实际打印值，不能使用旧P7/P7d请求哈希。

```powershell
python tools/crosscheck_model_protocols.py run --request artifacts/p7e/p7e_cross_v1/request.json --confirm-request-hash <prepare打印的哈希>
```

两进程、新进程首回合、CPU一Torch线程、blocking-exact。无需训练任何模型。
run完成后自动聚合：`artifacts/p7e/p7e_cross_v1/aggregate.json`。

重复启动仅允许加`--resume`复用完整且哈希一致的回合：

```powershell
python tools/crosscheck_model_protocols.py run --request artifacts/p7e/p7e_cross_v1/request.json --confirm-request-hash <原哈希> --resume
```

不覆盖报告、不删除半成品、不自动重试失败任务。代码/运行时/历史证据变化会拒绝
请求。不要在运行期间修改配置。输出路径使用短run ID，旧实验目录不写入。

## 解读

- matrix：六个模型—协议单元格，含全部训练种子和成功回合分母。
- within_protocol_comparisons：作者、本项目B0、B3之间同场景比较，原口径奖励差
  只在同协议内计算；B0/B3输入维度不同，预测归因仍需原B1/B2/B3对照。
- protocol_sensitivity：同checkpoint、同seed编号的原口径到达/碰撞终止/上限差，
  不计算跨协议奖励差，不声称完全相同初始交通或单独的预热因果作用。
- 历史P7为Gym transition reward；P7d为N−1历史评分，保留两种名字。
- P7保留原500步、just_had_collision条件；P7d保留501调用和任意车辆碰撞检查。
  跨协议reported_collision不是统一责任识别后的ego碰撞指标。
- 初始交通哈希仅在同协议内要求一致；两协议的编码与初始条件不同。
- 不把停车的低jerk、早碰撞的短耗时当作舒适性或效率提升。
- 作者只有一份权重，训练预算未匹配；工程complete不代表方法获胜、收敛或正式泛化。

完整80回合未执行前，不能据工程验收评价方法。下一步需先复盘结果；
P7c噪声诊断仍未实施，P8未批准。

## 本机已准备状态（2026-10-02）

场景200工程验收已完成，四回合共940次实际控制调用，工程完整性通过。
完整80回合请求也已准备；现在不要重复audit/prepare，也不需要重新训练。
在项目根目录直接运行：

```powershell
python tools/crosscheck_model_protocols.py run --request artifacts/p7e/p7e_cross_v1/request.json --confirm-request-hash f50f32e772fa47c8195e39ecf93fb8153cf989720f2e05be5260840e82bab5de
```

该哈希仅对应这份本机冻结请求。当前80回合没有启动，也没有完整方法结果。
工程记录见`reports/p07e_crosscheck_acceptance_20261002.md`。
