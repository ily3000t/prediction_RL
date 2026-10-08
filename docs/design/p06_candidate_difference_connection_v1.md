# 差分预测器的闭环接入边界

训练结果保留在原 c3734a0 来源，不修改训练请求、验证报告或旧模型。
新消费者通过历史 Git blob、完整成员 receipt、严格 state_dict 和 checkpoint
哈希接收完成结果；旧训练/恢复与普通模型加载器仍拒绝不匹配版本。

使用原 calibration split、episode-max standardized residual、90% nominal
coverage 和相同 disagreement floor，独立拟合新 ensemble。Calibration/test
不参与选模，不把校准集覆盖称为测试覆盖。三成员分歧不自动等于可靠置信区间。

在线继续调用原 FrozenProbeFeatures：149 个预测通道、候选顺序、归一化、
actor/history mask、末端零通道均不变。既有 DDPG 特征不使用 calibration
bands；新增校准不会给策略偷偷添加不确定性特征或动作屏蔽。

后续 DDPG 仅从头训练差分 B3，三个训练种子 0/1/2；120k 主终点、
20k/40k/60k 诊断节点和开发场景 200–219 保持 P7n。旧 B0/B1/B2/B3
只读复用，直接以原条件 B3 为主要增量比较。不同预测器的 policy checkpoint
即使维度相同也必须由 predictor hash 区分，禁止复用旧 B3 权重冒充适配结果。
