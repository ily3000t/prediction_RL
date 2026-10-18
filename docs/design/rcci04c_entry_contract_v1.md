# RCCI04c 入口实现与有界 smoke 契约

2026-10-18，用户“继续”确认匹配方案的实现；正式训练与长评价仍由用户启动。
本文件在入口实现/运行前冻结，不修改已完成桥接或旧训练源码/守卫。

## 正式入口

独立RCCI preset/权重contract，169维观测+ALL TimeFeature、相同400/300层DDPG。
复用原train/evaluation函数code object，以scoped globals明确注入provider、env、
权重I/O与资产校验。Raw仍用原contract，RCCI不能静默映射成Raw模式。
预测器冻结、原奖励/连续jerk/持续交通reset/原优化器与replay完全保持。

正式protocol严格为已确认方案：两组新RCCI×seeds0/1/2×120k请求步；四节点，
四组共960个当前环境评价，20个相同开发场景。配置独立可读，不允许未知字段。
prepare不启动训练；必须先有同源码已通过的入口smoke。正式运行由用户手动启动。

原Raw回放/日志已由04a完整核验。每个评价worker无需反复读取未使用的大replay：
父阶段完整校验训练receipt并封存index；worker核对绑定receipt/metadata、目录文件名、
所用snapshot/权重SHA，strict loader仍检查完整state dict。阶段结束/aggregate再完整
校验训练文件。这样不降低所用权重/契约验证，也不改任何随机采样或训练行为。

记录RCCI前缀/失效原因/候选覆盖与描述性latency，诊断不进入policy或replay State.info。
从头训练的同seed初始policy/q必须匹配历史expanded Raw，训练前即验证。
工程错误停止；方法负结果正常complete，method_effect_gate=null。

## 入口smoke（不是方法效果实验）

- 新RCCI ordinary/conditional，各seed0，节点[16]，请求16步，原完整episode结束，
  每训练job最多515控制步、16回合；minibatch/replay_start均8，仅该profile使用。
- 评价4组各1回合、simulator seed200：新RCCI用smoke权重，Raw只读原120k权重。
  两者训练预算不同，禁止汇总成方法收益表；只检验加载/执行/终止与封存。
- 共6个新进程session，上限36个episode、3,030控制步、6,702 simulationStep，
  3,030 ensemble/9,090 member forward、3,066 policy forward；每worker watchdog180s。
- 实际非terminal观测、预测调用与控制step对齐；原ST诊断guard保持0调用。
  每个session有原生step计数上限；checkpoint/optimizer/replay保存计数及初始化验证。
- Raw评价与对应旧记录精确比较；RCCI预测冻结，policy至少发生一次真实优化，
  严格save/load与evaluation-only限制必须通过。驾驶成败不作为工程通过条件。
- 干净固定源码，新短run ID、一次写入、失败不自动重试，不覆盖历史结果或变更seed。

训练run的resume仅复用sealed complete整job，不恢复中断优化器或replay。
同请求已完成的阶段可跳过完整job；失败/部分目录保留并报错。
实现、测试、smoke验收、运行文档与原子提交完成后才交接正式命令。
