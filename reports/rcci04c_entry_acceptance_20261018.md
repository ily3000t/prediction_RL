# RCCI04c 匹配策略入口验收

## 结论

独立RCCI训练/评价/聚合入口已实现并通过有界检查。六个旧Raw策略不重训；两个
冻结预测器不重训。正式720k请求已prepare，**没有启动正式训练或960回合评价**。
入口smoke不是方法效果或收敛证据，不能用其短训练结果比较成熟Raw模型。

## 实现边界

- `tools/train_rcci_matched_ddpg.py`：不可变prepare/run/aggregate、complete整job复用。
- `training/rcci_ddpg.py`：独立RCCI mode/feature/adapter/权重契约；原Raw入口不改，
  同维也不能相互加载；strict loader先验证完整policy/q，再加载，无部分权重迁移。
- 复用原ALL DDPG和原训练/评价函数code object，只注入明确的接口与审计依赖。
  相同输入的动作、replay、更新、target、RNG通过精确等价测试。
- 冻结v2特征、严格等价cached map、原观测与TimeFeature，保留短前缀/无效mask。
  新provider仅另标训练阶段release，不改原数学contract或默认provider。
- 保留原连续jerk、奖励/终止/bootstrap、持续交通reset、原2M scheduler与优化器。
  阻塞计算，不因CPU时延丢弃结果或使用stale，不加入MPC/Shield/commitment。
- 训练前核对同seed初始policy/q与历史expanded Raw一致；预测器始终冻结。
- 保存前缀/失效/actor覆盖、描述性latency与逐事件诊断；不进入replay State.info。
- 安装的ALL全部116个Python源文件、SUMO/package/native身份、配置与资产逐次校验。
  训练阶段封存完整index；评价worker轻量核对绑定receipt及所用权重，阶段边界再
  完整校验大回放/日志，避免每worker重复读未使用的大文件。
- 负效果正常complete、method_effect_gate=null，不删seed或自动加预算。

## 检查过程与实际预算

契约提交`9337cd5`，运行前计数勘误`435174b`，实现`44f5fc3`，长运行防护`92f2e6b`。
勘误补计原DDPG的优化/target-policy前向：上限5,158，不增加训练、预测或仿真预算。

首轮`artifacts/rcci04c/entry_v1`通过。补齐长运行的依赖/SUMO/index防护后，在新目录
`entry_v2`以相同参数重验，原产物不覆盖。两轮policy/q张量、训练episodes、初始化、
解压后的loss与minibatch采样索引逐项相等；不是按结果改超参或挑选seed。

最终入口执行提交：`92f2e6b4860640617da9f17798bc008e59ca820c`。

| 项目 | entry_v2实际 | 单轮上限 |
| --- | ---: | ---: |
| SUMO sessions/episodes | 6/6 | 6/36 |
| 控制step | 965 | 3,030 |
| simulationStep | 1,572 | 6,702 |
| ensemble/member forward | 965/2,895 | 3,030/9,090 |
| policy forward（含优化与target） | 1,177 | 5,158 |

ordinary短训练在52步完整episode结束，policy/q各45次更新；conditional为65步、
各58次更新。两组初始化均精确匹配历史expanded Raw，预测器保持不变，ST调用0。
原reward/动作/事件审核沿用原循环；每个非terminal观测只预测一次，terminal不预测。
四次保存后评价均完成，Raw两组与相应历史记录精确一致。驾驶失败不会阻止工程聚合。
这里只是16请求步的入口检查（batch/replay-start=8），不是正式模型已学会的证明。

最终smoke request digest：`21fa3784c8abfd525febd86a114a702d13bfeebdee5f06c3ec28a46a70b47702`。
aggregate SHA：`ac0114bffc313c1cb099fcd561b7fb36c287b140a52af479ee257e8745959b91`。
aggregate receipt SHA：`d22e4f5864d9265441236626b51610bb8cf53e081e8befad47ade675d14ba3a6`。
training index SHA：`b4629a4b2be70ef989b8b29127418583d48d5e4c6a5bbc26ab0cd9b0c32737ed`。
六个完整job inventory及receipt已独立重验。

首轮39项、补充后41项、防护后42项测试均通过；最终选定回归**230 passed，22.73s**，
无deselect，不声称全仓通过。`git diff --check`通过。原Raw/DDPG/环境/预测器文件未改，
baseline仍为干净`75051acc`；无推送、合并或tag，权重/日志/大产物不进Git。

## 正式请求与交接

`artifacts/rcci04c/matched_v1/request.json`，digest：
`8dfa87d77539dd7e6167127aa8a0299b8cd1a680bc62b979f58fec6f0b076b20`。
准备源码同为`92f2e6b`；后续仅文档提交可接受，运行仍严格绑定Python源码、配置、
设计、依赖、SUMO和资产，并分别记录准备/实际执行Git SHA，不放宽旧请求guard。

已独立确认preparation状态`prepared_not_trained`、6个新训练实例、720000请求步、
960个评价任务，正式目录不存在train/evaluate子目录。正式batch/replay-start为100/5000，
seeds0/1/2、20k/40k/60k/120k节点不变，120k为唯一主终点。

按 `docs/runbooks/rcci04c_matched_policy_v1.md` 由用户依次启动训练、评价与聚合。
没有额外诊断/泛化/作者50秒实验。960次评价只有20个交通场景，不冒充960个独立样本。
任何RCCI收益仍待该匹配实验，不由工程检查或计算加速推断。
