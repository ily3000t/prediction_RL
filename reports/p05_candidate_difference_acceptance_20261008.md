# 差分训练目标工程验收（2026-10-08）

## 实现范围

基于 `75051acc06f6ea2ab6dda9be678f191b63d04569`，在
`codex/p05-response-difference` / `E:/Prediction_RL_worktrees/response-difference-v1`
实现同 root 候选响应差分回归。原 Prediction_RL 主工作树和旧 WCDT_ACCVP
源码/产物未修改。没有 push、merge、SUMO、正式 ensemble/DDPG 训练。

- 新目标：归一化轨迹 MSE + lambda × 响应差分 MSE。
- 五个候选的十个无序对；稳定 actor ID、固定 root 坐标系和共同有效时间段。
- 保留零配对 root 的绝对监督与分母，明确报告 mask/配对覆盖。
- 配置初始 lambda=0.2；三成员 4101/4102/4103 独立从头训练。
- 架构、数据、优化器/预算、在线特征及 MSE 选模/早停规则不变。
- 新 checkpoint/请求版本与旧版本严格隔离；完整 optimizer/RNG 恢复。
- 新入口只读复用冻结数据与旧普通/条件模型，输出在新工作树。
- 离线 validation 配对诊断不读取 calibration/test，不设必须胜出 gate。

## 实际执行的测试

在 `pytorch` 环境、新工作树运行以下十个文件：

```powershell
python -B -m pytest tests/test_candidate_difference.py tests/test_candidate_difference_cli.py tests/test_epoch_training.py tests/test_masked_training.py tests/test_prediction_interfaces.py tests/test_calibration.py tests/test_offline_diagnostics.py tests/test_candidate_ranking.py tests/test_probe_features.py tests/test_actor_adapter.py -q --tb=short -p no:cacheprovider --basetemp E:/Prediction_RL_worktrees/response-difference-v1/artifacts/tests/prediction_01
```

结果：**226 passed，12.85 秒**。这是相关回归子集，不是整仓库测试或真实闭环评价。
此前单独执行的 79 项核心回归及 14 项初始入口测试也通过；上述 226 项包含
后续新增的 artifact 版本/目标一致性用例，不将重复测试累计为独立覆盖量。

已验证：

1. lambda=0 的 loss/gradient、两轮模型参数、Adam、root 顺序、RNG 与选模状态完全等价。
2. 五候选方差恒等式、common-error 抵消及绝对监督锚点。
3. 真差异为零时不强迫分离；错误响应符号受到惩罚。
4. 共同 mask、缺失 actor/时间段、终止截断和 masked 有限垃圾值不产生错误梯度。
5. pair/cell/root 的均等归约和零配对 root 分母。
6. NaN、全空监督、非法 lambda、非匹配预算/种子/网络/选模配置明确拒绝。
7. 新三成员导出、完整恢复、变更 lambda/旧版本恢复拒绝。
8. 原训练、checkpoint、校准消费者、actor adapter 与候选特征回归。
9. prepare 不训练、不写参考目录；错误确认 hash 不启动训练。
10. 严格 ensemble/member 对齐、配对 validation 统计和不可变诊断输出。

ordinary 模型的差分梯度理论上为零；float32 共享路径累加可能有舍入残差，
该测试使用绝对 1e-7 数值零容差。lambda=0 等价测试仍要求精确相等，未放宽。

初次测试被系统历史 pytest 临时目录权限阻挡；显式使用新工作树内的新 basetemp
并创建其父目录后解决，没有修改模型语义或已存在测试数据目录。

## 准备记录与下一步

真实冻结数据的请求准备记录应位于
`artifacts/prediction_difference/delta_v1/preparation.json`；该记录绑定实际
source commit、请求 hash、数据 fingerprints、参考文件 hashes 和配对覆盖。
本 Markdown 记录工程测试，不将请求准备、模型训练或方法有效性混为一谈。

运行入口与检查步骤：`docs/runbooks/p05_candidate_difference_v1.md`。
完整三成员训练和离线诊断由用户确认请求后手动运行。新 checkpoint 尚未开放
校准/DDPG 消费；离线结果审查后再决定下一阶段，不自动增加模型或实验。
