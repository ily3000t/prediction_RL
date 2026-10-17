# RCCI v2 完整既有 cohort：用户运行说明

这一步不是训练，也不运行 SUMO。它复用已有两个三成员预测器，检查固定 RCCI v2
在全部既有 train/validation 数据上的支持、误差与机制局限。负结果也正常完成。

## 已冻结范围

- 配置：`configs/development/rcci02_v2_cohort_validation_v1.json`，无 extends。
- 定义：`docs/design/rcci02_v2_cohort_validation_v1.md`，先提交再实现。
- 713 train + 177 validation 根、320 episodes，原顺序，含空监督根。
- CPU / torch 单线程；最多 1,780 次 ensemble forward，即 5,340 次成员 forward。
- 同时保存预测均值和诊断；产物可能较大，请预留磁盘空间，不加入 Git。
- 两个设计 episode 单独标记，不能把其结果称作独立确认。
- 不访问 calibration/test；不换模型、seed、地图、reward、动作、DDPG 或接口阈值。

## 准备

在 `pytorch` 环境、干净 RCCI 工作树执行。原始 `E:/Prediction_RL` 必须仍为
干净 `main`/75051acc 基线；不要为了运行修改它。

```powershell
cd E:\Prediction_RL_RCCI
python -B tools/validate_rcci_v2_cohort.py prepare --baseline-root E:/Prediction_RL --run-id rcci02_v2_full_v1
```

prepare 只核对并封存身份，不加载权重、不推理。打印 request 路径和
`confirm_request_hash`。run ID 必须全新；存在时不删除/覆盖原目录。
开始前确认范围与预算；不要在源码或配置变动后继续使用旧请求。

## 执行（由用户启动）

将下方占位替换成 prepare 实际打印的 hash：

```powershell
python -B tools/validate_rcci_v2_cohort.py run --request artifacts/rcci02_v2_cohort/rcci02_v2_full_v1/request.json --confirm-request-hash <PREPARE打印的HASH>
```

每 10 根输出进度。没有 resume、失败根跳过、自动重试或覆盖已启动报告。
若失败，保留目录和错误，先诊断；不能改阈值/种子让它通过。
本实现尚未在真实完整 cohort 执行，不把单元测试当成完整运行成功。

## 完成后看什么

输出位于 `artifacts/rcci02_v2_cohort/rcci02_v2_full_v1/`：

- `request.json` / `preparation.json`：配置、源码、环境、模型和所有输入身份。
- `roots/r0000.json` 等：逐根原始预测均值、ego rollout、特征和离线诊断。
- `index.json`：逐根文件 SHA256 与原根顺序。
- `report.json`：状态、实际预算、分组汇总；`receipt.json` 封存摘要。

报告按 train/validation × 设计 episode/其他既有 episode 分组：

1. 完整 5 s 覆盖、共同前缀及失效原因，保留未知后缀；
2. 同支持的 gap、跨来向 entry/exit/overlap 误差，及缺失事件计数；
3. 1/3/5 s 固定分母的 following 误差；无支持不是零误差；
4. native ego 总体误差和 reference ego 邻车误差分别列出；
5. pool cell 描述与等权 episode MAE；不把根/候选/时间格当独立场景；
6. 固定共同支持后的 ego/邻车/条件增量分解，以及 root/截尾主导现象。

`overlap` 只是宽节点控制体积占用，不是碰撞标签。输入替换可能离分布；
数值变化不等于有效控制，也不能据此直接宣称条件模型优越。
完成后交回 `report.json` 分析，不自动进入 DDPG、原生 smoke 或 online 接入。
