# RCCI04a 复查入口

本阶段已完成，无需用户重跑。结果见
`reports/rcci04a_cache_and_raw_reuse_20261018.md`。尚未开放正式RCCI训练。

新实现为 `features/road_consequence_geometry_cached.py` 与
`features/rcci_provider_cached.py`（均在`src/prediction_rl`下），原默认入口不变。
完整配置是下面两个工具的 `SETTINGS`，运行时保存至request，无嵌套配置。
设计契约在 `docs/design/rcci04a_cached_geometry_and_raw_reuse_v1.md`。

仅需要独立复查时，在`E:/Prediction_RL_RCCI`的pytorch环境使用全新短run ID：

```powershell
python -B tools/audit_rcci_cached_geometry.py --baseline-root E:/Prediction_RL --run-id cache_check_02
python -B tools/audit_rcci_raw_reuse.py --baseline-root E:/Prediction_RL --run-id raw_check_02
```

要求干净已提交源码、只读baseline仍为干净`75051acc`、所有原证据哈希不变。
不覆盖失败/完成目录，不重跑或放宽旧请求guard。

第一条只重算1,798个已有缓存摘要和有界计时，无模型推理；第二条只校验既有
训练/评价封存文件并反序列化六份最终权重检查，不创建agent。不连接SUMO、不训练。
新产物分别为`artifacts/rcci04a/<ID>`、`artifacts/rcci04a_raw/<ID>`。

已完成目录为`cache_exact_v1`与`raw_reuse_v1`。各`receipt.json`绑定request/report；
禁止手改contract、报告或SHA。缓存加速2.165倍不表示整体在线或驾驶效果改善。
Raw历史证据完整，但新native身份下严格匹配复用仍待桥接，并不意味着必须重训。
