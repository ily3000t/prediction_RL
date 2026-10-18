# RCCI04b 桥接复查

本阶段已执行通过，无需用户重复运行。验收见
`reports/rcci04b_raw_runtime_bridge_20261018.md`。
当前只支持纯Gym20/DDPG路径的限定复用，不支持据此声称MPC全功能等价。

配置：`configs/development/rcci04b_raw_runtime_bridge_v1.json`。
入口：`tools/bridge_rcci_raw_runtime.py`。没有配置继承/extends；request保存完整配置、
Git/source/environment、旧资产与实际binary路径/SHA、旧训练来源和固定job roster。

如确需独立重做检查，在`E:/Prediction_RL_RCCI`的pytorch环境选择全新短run ID：

```powershell
python -B tools/bridge_rcci_raw_runtime.py prepare --baseline-root E:/Prediction_RL --run-id raw_bridge_02
```

只使用该次prepare实际打印的`confirm_request_hash`，不得照抄上次hash：

```powershell
python -B tools/bridge_rcci_raw_runtime.py run --request artifacts/rcci04b/raw_bridge_02/request.json --confirm-request-hash <prepare实际输出>
```

要求干净已提交源码、只读baseline仍为干净`75051acc`、原证据/权重/二进制/环境
未变。固定12回合上限，串行worker；没有resume或自动失败重试。失败目录保留，
不改seed/阈值/报告，不把失败结果改成成功，不为复查重新训练。

已完成结果：`artifacts/rcci04b/raw_bridge_v1/report.json`，engineering gate=true。
旧Raw权重仅按Raw语义加载，不能加载进同为149维但语义不同的RCCI模型。

下一阶段尚无已实现/已授权的正式RCCI训练CLI。不要用旧Raw训练入口偷偷映射RCCI
模式；待独立版本化checkpoint契约、训练请求和运行文档完成后再由用户启动。
