# RCCI03 原生接入 smoke

仅用于工程语义，不训练、不加载策略、不评价方法效果。
规范先冻结于 `docs/design/rcci03_native_smoke_v1.md`。
固定五 arms × 两 replay，最多 10 个 SUMO session、15 episodes、1,800 受控步，
含原 reset warmup 后 simulationStep 上限 3,315；最多 1,444 ensemble forward。

当前扩展已在本工作树从固定源码构建并可导入，不复制旧 `.pyd`，不安装包。
构建 stdout 因 Windows GBK 解码缺失，保留原记录并在单独验收显式标注；
不能说完整编译日志已保存。新 worker 日志按二进制捕获，不再依赖默认文本编码。

## 新请求准备

在干净已提交 RCCI 分支、pytorch 环境运行；baseline 保持干净 75051acc。
不要覆盖或重试已经 started 的 run ID。

```powershell
cd E:\Prediction_RL_RCCI
python -B tools/smoke_rcci_native.py prepare --baseline-root E:/Prediction_RL --run-id rcci03_native_v1
```

准备不启动 SUMO。执行使用实际打印的 hash：

```powershell
python -B tools/smoke_rcci_native.py run --request artifacts/rcci03/rcci03_native_v1/request.json --confirm-request-hash <PREPARE打印的HASH>
```

父进程串行新进程运行固定脚本动作，不使用 agent、optimizer、replay 或策略 checkpoint。
最大 worker watchdog 600 s，失败保留并停止，不换 seed/cap 修成通过。

## 产物与解释

- `request.json`：源码/环境/native/SUMO/模型选择/配置与父报告身份。
- 各 replay_arm：原状态/动作/reward/done/traffic/RNG 轨迹、特征/masks、
  固定观测索引 0/1/10 的缓存示例、原始 resolved upstream settings。
- 逐次检验缓存离线特征等于在线 observation block，无额外模型 forward。
- `report.json`：实际预算及八项 learned-vs-baseline 完全语义等价检查。
- `index.json` / `receipt.json`：逐文件与摘要 SHA；日志保存原始 bytes。

RCCI provider/wrapper 是独立工程入口，mode 为 `rcci_ordinary/rcci_conditional`；
旧 Raw DDPG preset/checkpoint 会明确拒绝它们。149 维相同不意味着旧策略权重可复用。
terminal 不读取当前帧/不预测，预测块为零；合法短前缀继续显式传 mask。
工程错误会停止，不会以 stale/假零作为正常非终止 observation。

通过只表明接入语义保持。RCCI04 的 Raw 复用审查、训练预算和权重契约另行批准，
本命令不能启动它们。实际执行状态与结果需要单独验收，不将本文当通过证明。
