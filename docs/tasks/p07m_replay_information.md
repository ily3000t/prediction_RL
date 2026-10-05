# P7m：一次有界的信息利用检查

## 目标与前置条件

已完成 P7k 60k 匹配训练与 P7l Gym20 外部比较。只使用现有 replay，
检查 B2/B3 的三个冻结训练种子是否依赖预测数值和候选差异。
不是重新审计 reward、bootstrap、终止事件，也不批准 120k 或扩大预测模型。

## 修改范围

新增独立诊断、配置、测试和文档。旧源码、配置、权重、实验输出、
上游与 E:/WCDT_ACCVP 均不修改。无 SUMO 步进、DDPG 更新或新预测数据生成。

## 实现与验收

- 来源固定为 P7k 60k 的 ordinary/conditional × 0/1/2，不挑选种子或 checkpoint。
- 验证历史 SHA、完整子报告与实际使用文件的 sealed inventory。
- 主样本为 outcome-blind SHA 优先级的最多 512 个 replay state；
  另对低速、匝道、短历史、遗漏车辆状态各取最多 128 个，合并去重后最多 1024/model。
- 分层结果不作为总体频率估计，不把采样状态数当独立场景数。
- 三种扰动只用于原策略只读查询；保留原 20 维观测与既有时间坐标。
- 原网络逐条/批量查询在冻结容差内一致；权重、target、optimizer、scheduler、
  replay、TimeFeature 与查询 RNG 不变。
- engineering smoke：每模型最多扫描 128 条、查询 24 个状态，六模型均覆盖。
- 完整诊断由用户手动启动；零敏感性或负面结论也正常聚合。

## 暂停条件

来源/输入/layout 不匹配、出现 NaN 或只读 guard 失败时停止，保留失败产物。
不静默重试、不放宽标准。已有 replay 缺少完整邻车未来标签，无法判断失败
由预测错误还是学习波动造成时，明确标记不可识别，不另开无限诊断循环。

## 交付与 Git

功能分支 codex/p07m-replay-information-use；实现、测试和验收文档原子提交。
数据放 artifacts/p7m/ 短 run ID 下且 Git 忽略。不推送、不新增 tag。
运行命令见 docs/runbooks/p07m_replay_information.md。
