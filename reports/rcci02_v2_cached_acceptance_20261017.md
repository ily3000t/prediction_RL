# RCCI02-A v2：实现与缓存重算验收

后续完整既有 cohort 验证入口已实现且请求已准备、尚未运行，见
[准备记录](rcci02_v2_cohort_preparation_20261017.md)。下文保留本次六根缓存验收的范围和结论。

## 身份、冻结顺序和实际范围

- 在缓存重算前，契约先独立提交：`1daa643`，见
  `docs/design/rcci02_adapter_contract_v2.md`。
- 实现提交：`8c3bb25786baef7e5f9b8ea53c6dd343e6419523`。
- 配置：`configs/development/rcci02_v2_cached_audit_v1.json`。
- 产物：`artifacts/rcci02_v2/rcci02_v2_bnd_v1/`；status=complete。
- request hash：`82cb25d7eb8e26d14c23d2fec15654f1ca1058fec5d633546eb3a760c55f5a69`。
- report SHA256：`bbc6c3494e917bc707064f4ff6152cd38ba8277ed4db70ecba3d64988b7d4362`。
- roots SHA256：`4b592620ef035006e193582a3f44f7732290f91a318649511918264d0093ea9b`。

仅读取上一轮保存的 ordinary/conditional ensemble mean 与原 ego rollout。
**模型权重加载=0，模型前向=0，SUMO=0，训练=0**；没有新 seeds，沿用六根的
collection seeds 11000/12000。没有 calibration/test target 访问。
71 个既有文件仅做已接受身份的 SHA 核验，不重新选择或加载模型。

旧 v1、Raw、预测器、loss、地图、投影阈值、actor slots、reward/action、DDPG
和训练路径均未改。v2 是独立 adapter，仍是离线原型，未注册 online provider。
v1 特征逐根精确重现，原 root metadata 与完整支持后的 following 数值也检查一致。

运行后核验 source/environment/config/设计文档/地图/父产物/receipt/71 个文件；
从保存预测重新构建 v1/v2、逐根对比并 aggregate，与已封存报告完全一致。
没有为复核新增推理、改写结果或重跑旧请求。

## 实现到底改变了什么

每车有效支持截止于首个 v1 无效点之前，不平滑/夹紧/重排任何预测，也不重新
接入后来恢复几何可投影的点。following 使用 ego+全部已选邻车的共同前缀；
cross 使用 ego+全部 eligible 邻车的共同前缀，不删除坏 actor。

按字段输出 supported-prefix minimum、进入及前缀内 overlap/first-overlap，
同时明确实际支持终点、未知后缀、左右截尾和完整 5 s flag。
这与“完整 5 s 最小值/完整通过 margin”是不同语义，不是给原模型修复后报告同一指标。
维度仍为 149，但语义已变，不能复用 Raw/v1 策略权重直接评价 v2。

## 1. 完整覆盖没有被做高

两 split 合并，仅为这六根设计样本的描述。六根来自两个 episode，不能按
30 个候选或 actor-time 扩大独立样本量。

| 项目 | B2 ordinary | B3 conditional |
| --- | ---: | ---: |
| 完整 5 s following pool（v1=v2） | 25/30 | 21/30 |
| 完整 5 s cross pool（v1=v2） | 15/20 | 11/20 |
| front 数值可用：v1完整池 → v2支持前缀 | 20 → 25 | 19 → 25 |
| rear 数值可用：v1完整池 → v2支持前缀 | 11 → 13 | 13 → 13 |
| overlap 描述可用：v1完整池窗口 → v2支持前缀窗口 | 15 → 20 | 11 → 20 |

front/rear 的候选总数均为 30，关系本身不适用的候选不强行生成数值。
cross 的分母仅为 20 个 applicable 候选。箭头左右是不同支持语义，不能写成
“预测可用率/准确率提高到 100%”。完整 5 s 数量逐候选强制相等。

B2 额外保留 front 5、rear 2、cross 5 个未知后缀候选的早期信息；
B3 分别保留 front 6、rear 0、cross 9 个。所有这些候选 full_5s 仍为 false。
B2 共同前缀为 4.4–5.0 s，B3 为 2.4–5.0 s；本批没有 root-only 支持。

支持内最小 gap 也可能在 root：两种方法均为 front 1 个、rear 2 个，不能将
这些当前状态数值当未来预测命中。其余数值发生在预测点，也不等于已经验证其正确。
质量 flag 只表示几何/数值契约支持，不是已校准置信度或误差上界。

## 2. 时间字段仍有明显的机制局限

20/20 个 applicable cross 候选都有正的支持内节点 overlap。
这并不代表 20/20 危险/碰撞；全合流节点是宽控制体积，二值 overlap 在该批
样本中没有区分度。连续 overlap 范围为：

- B2：约 3.240–5.000 s；
- B3：约 2.167–5.000 s。

这些范围包含支持长度差异，不可直接解释为 B3 预测的物理冲突更少。
两种方法各有 9/20 的 overlap duration 等于前缀终点 T，说明部分时长只是
“从 root 一直同时占用到支持终点”，可能随支持长度/质量 mask 变化。

进一步只读复查 pair 诊断：两种方法选中的 actor 都在 **20/20** 个候选中
从 root 占用至支持终点；first_overlap 在 **20/20** 个候选中等于 ego entry。
first_overlap 范围两方法相同，为 0–约 1.760 s。因此，本批这一时间通道主要由
ego 参考计划与当前占用决定，**不能拿它作为邻车动作条件响应增量的证据**。

与 v1 相比，B3 的 cross selected actor 有 5/20 发生变化，B2=0；这来自新前缀
和已冻结池化规则，并不是 DDPG 执行动作改变或策略更安全的结果。
旧完整 signed margin 在该批无支持的负结论保留；v2 没有把它重新标成有效，
而是使用不同、明确截尾的短时域描述。

## 3. 测试与实际边界

- 新增 adapter 测试 23 项、缓存审计测试 9 项：32 passed。
- 含旧 Raw/RCCI 的最终选定离线回归：**285 passed、4 deselected，18.26 s**。
- 测试目录：`artifacts/tests/rcci02_v2_reg_v1`。
- 4 项仍是缺少 st_cy 的原生导入检查，未关闭/放宽旧测试或构建扩展。

覆盖首点失效、晚期倒退、负速度/off-path 后不恢复、空池/omitted、保留全部
物理 actor、root-only、截尾、不造假退出/ETA、正窗口、批量/单个、ID/坐标变换、
future poisoning，以及缓存审计拒绝把部分支持冒充完整覆盖。
本轮未用真实 future target 给 v2 做准确性评价，没有 policy outcome 数据。

## 4. 明确决策与下一步

**v2 的实现与缓存工程验收完成，可以保留作为下一步冻结离线验证对象；
但不能因此直接启动正式 DDPG，也不能宣布 RCCI 或动作条件增量有效。**

不再用同六根反复改字段/阈值，也不以覆盖数变好为理由进一步修改区域或模型。
下一步应一次性冻结较广的**既有 train/validation** 离线验证范围，检查：

1. 共同前缀长度与失效分布是否在整个已冻结 cohort 上稳定；
2. 相同原生可观察支持上的 gap/窗口误差，不以更早失效逃避难样本；
3. 数值差异中 ego 计划、邻车响应和质量/截尾各自的贡献；
4. 宽节点 occupancy 是否仍主要由 ego/支持长度决定；如是，将该字段降级或
   放弃作为核心机制证据，不据此强行增加网络、奖励或训练预算。

已存在 cohort 为 713 train + 177 validation 根/320 episodes（含四个空监督根），
沿用其原清单与 train/validation 分组；设计使用的两个 episode 应单列标记，不
当成独立确认。这个范围说明不是已创建的运行请求或已实现的新 CLI。
正式请求、成本上限和命令仍须在下一步明确，不能静默扩展到校准、test 或新 seeds。
原生 RCCI03 smoke、Raw 复用审核和正式训练仍未开始。

## Git

契约、实现、验收分别作原子本地提交；只在当前 RCCI codex 分支，不合并、不推送、
不打 tag。旧项目、原始基线 main 与归档工作树不变；原始预测和报告留在 artifacts。
