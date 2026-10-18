# RCCI04a：缓存优化与 Raw 复用审计

执行源码：`8cdc2ec3bd98ad4f07deffdf43be3a59e27a0140`，干净工作树。
只读 baseline 保持 `75051acc06f6ea2ab6dda9be678f191b63d04569`，结束时仍干净。
本轮不训练、不执行策略、不启动 SUMO、不运行冻结预测器 forward。

## 1. 严格等价优化

新增独立 `CachedRoadConsequenceMap` 和显式 provider，不替换原默认入口。
只缓存 lane 折线静态量及精确 `(lane,length,width)` 控制体积区间；128 项 LRU，
key 不量化、地图只读。原几何、v2 summary、Raw、DDPG、奖励、上游文件和旧守卫未改。

验证已封存的 890 roots × 两预测器 = 1,780 个摘要，逐值比较全部 149 个 float32
特征及完整候选诊断（actor ID、tie-break、mask、prefix、occupancy、失效原因）。
另验证 RCCI03 的 18 份 history/mean/ego 缓存，比较原生特征并用未改原 map 重算
完整 summary。**1,798 个摘要全部相等**。全量区间 cache：81,146 hit、4 miss、4 项。

只用上述 18 个原生缓存计时：各实现 warmup 一轮，交替先后顺序执行三轮，
每实现 54 个样本，每次仍比较完整输出。排除文件读取、模型推理与 SUMO。

| 后处理指标 | 原实现/ms | 缓存/ms |
| --- | ---: | ---: |
| mean | 187.18 | 86.46 |
| p50 | 183.38 | 84.91 |
| p95 | 198.62 | 92.71 |
| p99 | 217.86 | 93.16 |
| max | 239.04 | 93.50 |

平均加速 2.165 倍、耗时降低约 53.8%。这是本机热静态缓存上的后处理描述性结果，
不包含初始化/首次 miss；54 个重复计时不是独立交通场景，不能直接换算训练加速，
也不能证明达到 0.5 s 部署门槛。**短前缀/route reversal/endpoint 限制原样保留**，
计算加速不等于方法有效。

## 2. Raw 复用审计

来源 `artifacts/p7n/p7n_120k_v1`，固定 120k 主终点，不选中间最佳 checkpoint。
请求 digest=`31301c64cf15606f8fc05a0280b5f7f75fd1c6e33b5cd877087284e59f31befa`；
原 aggregate SHA=`e2dd769d3943df09db5b2d46991c739cb92604cbbc5277de749e1383b7318c79`。

| Raw 方法 | seed | 主终点实际步数 | 最终权重 SHA 前12位 |
| --- | ---: | ---: | --- |
| ordinary | 0 | 120073 | `56e8c5deef8e` |
| ordinary | 1 | 120081 | `2db9b40060a1` |
| ordinary | 2 | 120093 | `5f3a52d58328` |
| conditional | 0 | 120015 | `df4579f11b6c` |
| conditional | 1 | 120109 | `aae6e45ca6be` |
| conditional | 2 | 120120 | `5e9ff175b2be` |

实际步数来自原协议完整 episode 边界。六个训练 receipt 的全部文件（含 replay、
loss、日志、四节点权重）已 streaming 校验；对应 480 个既有评价 receipt 完整，
policy SHA、actual steps、job/seed/outcome 及 no-update 标记一致。主终点为120回合，
全程只有20个独立 simulator 场景，不把跨模型/节点结果当成480个独立样本。

最终 policy/q 仅反序列化检查，未创建 agent：全部 key、float32/有限值、400/300层、
169维观测、TimeFeature=0.001、特征名哈希和三成员 predictor SHA 匹配。
从头训练、expanded/zero 初始化、预算与 ALL preset 一致；历史83个执行源文件、
25个上游文本及安装的 ALL 核心源码一致。非 extension 的运行身份相同。

两项差异须显式处理：

- 原配置字节 SHA=`537423fe0bb27ea19bff87caaa4ab2388789bd2cac95c1cce3d9832beb142d8a`，
  当前=`29ee8dcdfa8165bee21d31f7abad87937c1707e16f4055737132f70c568142f6`。
  已由历史 Git blob 证明仅 LF/CRLF 不同，无参数变化；未改旧 contract 或绕过 loader。
- 原 `st_cy` SHA=`498a57f2096e905bb1151eb37dfde41cfb1b76668c343c16e55ae63fcebc76b3`，
  重编译=`aeacc8434bb6c0cd9777105fe21f8e97d5d52908379ef8e467d3d4c986acc04a`。
  相同 pyx 来源与 RCCI03 同环境 parity，不等于跨构建行为等价。本审计没有运行策略。

结论：**历史 Raw 对照有效保留，新匹配复用待显式运行时桥接**。
不因 binary hash 不同自动重训，不删除历史结果，不把同维 Raw 权重装入 RCCI，
不将评价权重假装完整断点续训。

## 3. 产物、测试与下一步

- 缓存产物：`artifacts/rcci04a/cache_exact_v1`，report SHA：
  `d2c9de5b2e8387817e043660ebf0bf859e4bf8e0495c7baf703f6ff6b54adab1`。
- Raw产物：`artifacts/rcci04a_raw/raw_reuse_v1`，report SHA：
  `0f85dbea6694b455559e228b5e9e2168e8851c7731899b38aa3bef35b85f5afb`。
- 各目录保存 request/config/commit/命令/环境/输入哈希及 receipt；缓存另有完整
  equivalence/timing，Raw另有 verified_inventory。大型产物均忽略，不进Git。
- 最终选定回归 **162 passed，11.14 s**，无 deselect，不声称全仓通过。
  首轮151 passed、11个fixture因系统临时目录权限失败，改用新项目临时目录后
  全部重跑通过，未改断言。几何/provider先行测试14 passed。
- 原子提交：设计`2929aee`、缓存`e9d3c60`、全量核验入口`d0725f3`、Raw审计`8cdc2ec`。
  文档后续提交不改变已执行源码。无推送、合并或新tag。

下一步先做有上限的运行身份桥接并冻结 RCCI04 匹配训练方案，不立即启动大预算。
若允许复用，应保留两组Raw×三seeds，只训练 RCCI ordinary/conditional 的新策略。
具体预算/场景/checkpoint选择/统计须另行冻结、人工批准，由用户执行。
